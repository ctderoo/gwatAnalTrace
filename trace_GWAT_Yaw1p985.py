from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm,Normalize
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
import create_gwat_hardware as gwathw

home_directory = os.getcwd()
figure_directory = home_directory + '/Figures'
##########################################################
# PANTER Geometry Parameters

N = 10**5
# Set average yaw angle ('blaze')
alpha = 1.985*pi/180 # 1.8
# Assumes tracing to this order.
order = 14
# Constant arcsecond scale for desired MM focal length.
arcsec = 12000./3600/180*pi
# Assumes Mg Kalpha source.
wave = emlines.mgk

gwat_xous = gwathw.create_gwat_xous()
gwat_facets = gwathw.create_gwat_facets(alpha = alpha,order = order,gwat_xous = gwat_xous)

########################################################################################
# Needed functions enabling GWAT trace.
########################################################################################

def trace_gwat(gwat_xous = gwat_xous,gwat_facets = gwat_facets,order = order,wave = wave,N = 10**5,diff_focus_zloc = None):
    # Tracing from the source through the SPO MMs.
    gwat_rays = gwatf.make_gwat_source(N,wave,gwathw.source_dist,gwathw.clock_angles)
    spomm_rays = ArcSPO.SPOPetalTrace(gwat_rays,gwat_xous)
    
    # I'm now doing this rather than what's embedded in the facet calculation. Is it bad? Time will tell.
    # Computing the SPO focus characteristics, the focal length, and the effective radius.
    focused_spo_rays,dz = gwatf.bring_rays_to_disp_focus(spomm_rays) 
    xFocus,yFocus,zFocus = array([mean(focused_spo_rays.x),mean(focused_spo_rays.y),-dz])
    
    # Now creating a saved version of the SPO MM rays (spomm_rays) and readying a set of rays
    # referenced to the SPO focus for tracing.    
    system_spo_rays = copy.deepcopy(spomm_rays)
    prays = system_spo_rays.yield_prays()
    tran.transform(prays,xFocus,yFocus,-zFocus,0.,0.,0.)
    system_spo_rays.set_prays(prays)
    
    # Setting the GWAT order in accordance with this function for "order selection". 
    gwatf.set_gwat_order(order,facets = gwat_facets)
    
    # And now tracing the rays through the adjusted CAT petal.
    grat_rays = ArcCAT.CATPetalTrace(system_spo_rays,gwat_facets)
    
    ## Focusing the diffracted rays.
    #meas_rays = copy.deepcopy(grat_rays)
    #prays = meas_rays.yield_prays()
    #surf.focusX(prays)
    #meas_rays.set_prays(prays)

    # Cleaning up and naming everything appropriately.
    system_spo_rays = system_spo_rays
    system_grat_rays = grat_rays
    system_spo_focused_rays,spo_dz = gwatf.bring_rays_to_disp_focus(system_spo_rays)
    
    if diff_focus_zloc is None:
        system_diff_rays,diff_focus_zloc = gwatf.bring_rays_to_disp_focus(system_grat_rays)
        prays = system_diff_rays.yield_prays()
        tran.transform(prays,0.,0.,-diff_focus_zloc,0.,0.,0)
        system_diff_rays.set_prays(prays)
    else:
        system_diff_rays = copy.deepcopy(system_grat_rays)
        prays = system_diff_rays.yield_prays()
        tran.transform(prays,0.,0.,diff_focus_zloc,0.,0.,0)
        surf.flat(prays)
        tran.transform(prays,0.,0.,-diff_focus_zloc,0.,0.,0)
        system_diff_rays.set_prays(prays)
        
    return system_spo_rays,system_grat_rays,system_spo_focused_rays,system_diff_rays,diff_focus_zloc

def trace_gwat_orderStats(xous = gwat_xous, facets = gwat_facets,order = order,center_wave = (1240*10**-6)/1253.4,wave = wave,
                          repeats = 40,pickle_file = 'DiffRays_orderStats.pk1',kind = 'diff',N = 10**6):
    # Doing the single wave raytrace to find where the best focal plane is.
    spoRays_SW,gratRays_SW,focRays_SW,diffRays_SW,diff_dz_SW = trace_gwat(gwat_xous = xous,gwat_facets = facets,N = 10**5,order = order,wave = (1240*10**-6)/1253.4)
    #print diff_dz_SW
    order_ray_dict = {}
    
    def repeatTracer(i,order_ray_dict,select_order,kind = 'diff'):
        # Next doing the full emission line ratrace to compute what the X-ray data ought to look like. 
        spoRays,gratRays,focRays,diffRays,diff_dz = trace_gwat(gwat_xous = xous,gwat_facets = facets,
                                                               N = N,order = None,diff_focus_zloc = diff_dz_SW,wave = wave)
        
        if kind == 'diff':
            diffInd = diffRays.order == select_order
            order_ray_dict['N' + str(i)] = diffRays.yield_object_indices(diffInd)
        elif kind == 'focus':
            order_ray_dict['N' + str(i)] = focRays
        
    for i in range(repeats):
        print 'On repeat #' + str(i) + '...'
        repeatTracer(i,order_ray_dict,select_order = order,kind = kind)
        
    ray_object = ArcRays.merge_ray_object_dict(order_ray_dict)
    ray_object.pickle_me(pickle_file)
    return ray_object,order_ray_dict