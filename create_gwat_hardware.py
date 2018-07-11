from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import os
import pdb
import copy
import cPickle
import string
import time
import csv

import PyXFocus.transformations as tran
import PyXFocus.surfaces as surf
import PyXFocus.sources as source
import PyXFocus.analyses as anal

import arcusTrace.SPOPetal as ArcSPO
import arcusTrace.CATPetal as ArcCAT
import arcusTrace.DetectorArray as ArcDet
import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusComponents as ArcComp
import arcusTrace.arcusRays as ArcRays

import torus_maker
import emission_line_sources as emlines
import gwat_funcs as gwatf

home_directory = os.getcwd()
figure_directory = home_directory + '/Figures'
##########################################################
# PANTER Geometry Parameters

N = 10**5
# Set average yaw angle ('blaze')
alpha = 1.985*pi/180 # 1.8
# Assumes tracing to this order.
order = 8
# Constant arcsecond scale for desired MM focal length.
arcsec = 12000./3600/180*pi
# Assumes Mg Kalpha source.
wave = emlines.mgk

########################################################################################
# Setting up the XOUs.
#######################################################################################
source_dist = 119358.
spo_edge_block = 12.

xou_numbers = ['MM-0025','MM-0027']
r_o = [737.,737.]
r_i = [710.,710.]
azwidth = [66 - 2*spo_edge_block,66. - 2*spo_edge_block]
plength,slength = [40.,40.],[40.,40.]
clock_angles = radians(7.25/array([-2,2]))
# Based on xFWHM focal length, PIXI Report, Draft 1.2
focal_lengths = [11959.,11967.] 
disp_scatter = [2.00e-6,2.00e-6]  # Lorentz Parameters.
cdisp_scatter = [2.35e-5,2.35e-5]   # Empirical Parameters from SPO cross-dispersion scaling.
scatter_kind = ['Lorentz','Empirical']

def create_gwat_xous(dscatter = disp_scatter,cdscatter = cdisp_scatter,scatter_kind = scatter_kind):
    gwat_xous = dict((xou_numbers[i],ArcComp.xou(i,r_i[i],r_o[i],azwidth[i],plength[i],slength[i],clock_angles[i])) for i in range(2))
    
    for xou in gwat_xous:
        gwat_xous[xou].xou_coords.z = 0.0    # Modifying the origin for ArcSPO.SPOPetalTrace.
        gwat_xous[xou].focal_length = focal_lengths[gwat_xous[xou].xou_num]
        gwat_xous[xou].scatter = scatter_kind
        if dscatter is None:
            gwat_xous[xou].dispdir_scatter_val = disp_scatter[gwat_xous[xou].xou_num]
        else:
            gwat_xous[xou].dispdir_scatter_val = dscatter[gwat_xous[xou].xou_num]
        if cdscatter is None:
            gwat_xous[xou].crossdispdir_scatter_val = cdisp_scatter[gwat_xous[xou].xou_num]
        else:
            gwat_xous[xou].crossdispdir_scatter_val = cdscatter[gwat_xous[xou].xou_num]
            
    return gwat_xous

gwat_xous = create_gwat_xous(dscatter = disp_scatter,cdscatter = cdisp_scatter,scatter_kind = scatter_kind)

def check_xou_quality(xous = gwat_xous):
    ############
    gwat_rays = gwatf.make_gwat_source(N,wave,source_dist,clock_angles)
    spomm_rays = ArcSPO.SPOPetalTrace(gwat_rays,xous)

    m25_rays = spomm_rays.yield_object_indices(spomm_rays.xou_hit == xous['MM-0025'].xou_num)
    m27_rays = spomm_rays.yield_object_indices(spomm_rays.xou_hit == xous['MM-0027'].xou_num)
    
    focused_spo_rays,focus_pos,hpd27,fwhmX27 = gwatf.do_xou_focus(m27_rays,xous['MM-0027'])
    focused_spo_rays,focus_pos,hpd25,fwhmX25 = gwatf.do_xou_focus(m25_rays,xous['MM-0025'])
    
    focused_spo_rays,dz = gwatf.bring_rays_to_disp_focus(spomm_rays)
    hpd,fwhmX = anal.hpd(focused_spo_rays.yield_prays())/arcsec,anal.rmsX(focused_spo_rays.yield_prays())*2.35/arcsec
    print hpd27,hpd25,fwhmX27,fwhmX25,hpd,fwhmX
    return focused_spo_rays,hpd,fwhmX

##########################################################
# CAT Petal Parameters
xou_cat_zspace = 591.
xgratsize,ygratsize = 32.67,31.66
r_avg = mean([710,737])
offsets = [-(4 + 32.67/2),(4 + 32.67/2),-(9.82 + 32.67)/2,(9.82 + 32.67)/2]
facet_names = ['X10','X14','X13','X15']
facet_rolls = array([0.0,0.0,0.0,0.0]) #array([0.0,-0.7,-2.3,1.8])*pi/180/60

print 'CAT Facet Rolls: ' + facet_rolls

def roll_facets(roll_angles,facets):
    keys = facets.keys()
    for i in range(len(roll_angles)):
        x,y,z,xhat,yhat,zhat = facets[keys[i]].facet_coords.unpack()
        roll_matrix = tran.tr.rotation_matrix(roll_angles[i], zhat, point=None)[:3,:3]
        new_xhat,new_yhat = dot(roll_matrix,xhat),dot(roll_matrix,yhat)
        facets[keys[i]].facet_coords = ArcComp.coordinate_system(x,y,z,new_xhat,new_yhat,zhat)

########################################################################################
# Setting up the grating facets.
########################################################################################
def create_gwat_facets(alpha = alpha,order = order,gwat_xous = gwat_xous,N = N,
                       wave = wave,source_dist = source_dist,clock_angles = clock_angles):
    # GWAT Rays start at the infinite focus of the XOUs.
    gwat_rays = gwatf.make_gwat_source(N,wave,source_dist,clock_angles)

    #############
    # Finding the focus position of the co-alignmed SPO XOUs through raytracing
    spomm_rays = ArcSPO.SPOPetalTrace(gwat_rays,gwat_xous)
    
    # Computing the SPO focus characteristics, the focal length, and the effective radius.
    focused_spo_rays,dz = gwatf.bring_rays_to_disp_focus(spomm_rays) 
    xFoc,yFoc,zFoc = array([mean(focused_spo_rays.x),mean(focused_spo_rays.y),-dz])
    
    r_func = r_avg - yFoc
    r_func = r_func - (r_func/zFoc)*xou_cat_zspace
    
    #############
    # Studying the all gratings co-planar case. First, establishing the grating positions on the torus and putting them all on the same plane.
    facet_assembly_xpos = asarray([r_avg*sin(ca) + offset for offset,ca in zip(offsets,repeat(clock_angles,2))])
    facet_assembly_ypos = array([0.,0.,0.,0.])
    facet_assembly_zpos = array([0.,0.,0.,0.])
    facet_assembly_pos = vstack((facet_assembly_xpos,facet_assembly_ypos,facet_assembly_zpos))
    facet_assembly_center = array([0,r_func,torus_maker.solve_for_z_livetrace(0,r_func,zFoc,xou_cat_zspace,alpha)])
    
    #############
    # Next, computing the local orientation vectors on the torus to figure out how to place the gratings on a flat plate.
    disp,gbar,norm = torus_maker.make_facet_orientation_vectors(facet_assembly_center,alpha = alpha)
    facet_assembly_orientation_mat = torus_maker.make_rot_matrix(disp,gbar,norm)
    
    #############
    # Finally, making the GWAT facets on a flat plate. They should all be planar and share the coordinate system computed for the blaze.
    gwat_facets = dict((facet_names[i],ArcComp.facet(i,1.,1.,1.)) for i in range(len(facet_names)))
    facet_system = [dot(facet_assembly_orientation_mat,facet_assembly_pos[:,i]) + facet_assembly_center for i in range(4)]
    
    for i in range(len(facet_assembly_xpos)):
        key = gwat_facets.keys()[i]
        #facet_system = dot(facet_assembly_orientation_mat,facet_assembly_pos[:,i]) + facet_assembly_center
        gwat_facets[key].facet_coords = ArcComp.coordinate_system(facet_system[i][0],facet_system[i][1],facet_system[i][2],disp,gbar,norm)
        gwat_facets[key].order_select = order
        gwat_facets[key].xsize = xgratsize
        gwat_facets[key].ysize = ygratsize
    
    gwatf.set_gwat_diff_eff(gwat_facets)   
    
    roll_facets(facet_rolls,gwat_facets)
    
    return gwat_facets