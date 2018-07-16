from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.interpolate import RegularGridInterpolator as RGI
from astropy.modeling import models,fitting
import os
import pdb
import copy
import cPickle
import string
import time
import csv
from scipy.optimize import curve_fit


import PyXFocus.transformations as tran
import PyXFocus.surfaces as surf
import PyXFocus.sources as source
import PyXFocus.analyses as anal

import arcusTrace.SPOPetal as ArcSPO
import arcusTrace.CATPetal as ArcCAT
import arcusTrace.DetectorArray as ArcDet
import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusComponents as ArcComp
import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.arcusRays as ArcRays

import torus_maker
import emission_line_sources as emlines

from matplotlib.colors import *
import matplotlib
from matplotlib import gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

arcsec = 12000.*pi/180/3600
########################################################################################
# Needed functions enabling GWAT trace.
########################################################################################

def make_xou_source(clock_angle,wave,N,source_dist):
    prays = source.subannulus(700,740,0.10,N,zhat = -1.)
    tran.transform(prays,0.,0.,0.,0.,0.,clock_angle - pi/2)
    tran.pointTo(prays,0.,mean([705,745]),source_dist,reverse = 1)
    if type(wave) == float: 
        wavedist = zeros(N) + wave
    else:
        wavedist = wave.draw_waves(N)
    xou_rays = ArcRays.ArcusRays(prays,wavedist)
    return xou_rays

def make_gwat_source(N,wave,source_dist,clock_angles):
    gwat_dict = dict(('XOU' + str(i),make_xou_source(clock_angles[i],wave,N,source_dist)) for i in range(len(clock_angles)))
    gwat_rays = ArcRays.merge_ray_object_dict(gwat_dict)
    return gwat_rays

def bring_rays_to_disp_focus(ray_object):
    manipulated_rays = copy.deepcopy(ray_object)
    prays = manipulated_rays.yield_prays()
    dz = surf.focusX(prays)
    manipulated_rays.set_prays(prays)
    return manipulated_rays,dz

def calc_line_stats(prays,arcsec = arcsec):
    hpd = anal.hpd(prays)/arcsec
    fwhmX = anal.rmsX(prays)*2.35/arcsec
    return hpd,fwhmX

def do_xou_focus(input_rays,xou):
    prays = input_rays.yield_prays()
    rays = ArcUtil.undo_ray_transform_to_coordinate_system(prays,xou.xou_coords)
    ray_object = ArcRays.ArcusRays(rays,input_rays.wave)
    focused_spo_rays,dz = bring_rays_to_disp_focus(ray_object)
    focus_pos = array([mean(focused_spo_rays.x),mean(focused_spo_rays.y),dz])
    prays = focused_spo_rays.yield_prays()
    hpd = anal.hpd(prays)/arcsec
    fwhmX = anal.rmsX(prays)*2.35/arcsec
    return focused_spo_rays,focus_pos,hpd,fwhmX

def set_gwat_order(order,facets):
    for key in facets.keys():
        facets[key].order_select = order
    
def set_gwat_diff_eff(facets,diff_eff_pointer = 'C:/Users/Casey/Software/Bitbucket/caldb-inputdata/gratings/Si_4um_deep_30pct_dc_extended.csv'):
    geff_func = ArcPerf.make_geff_interp_func(diff_eff_pointer,style = 'new')
    for key in facets.keys():
        facets[key].set_geff_func(geff_func)

########################################################################################
# Plotting functions for the GWAT trace.
########################################################################################

def get_fwhm_from_image_fit(image,pix_size = 0.020,arcsec = arcsec): 
    disp_dir_counts = sum(image,axis = 1)
    fit_levmar = fitting.SimplexLSQFitter()
    lmodel = models.Lorentz1D()
    lmodel.amplitude,lmodel.x0,lmodel.fwhm = max(disp_dir_counts),float(len(disp_dir_counts))/2,2.35
    l = fit_levmar(lmodel,range(len(disp_dir_counts)),disp_dir_counts,acc = 1e-6)
    return l.fwhm*pix_size/arcsec,disp_dir_counts,l(range(len(disp_dir_counts)))

def do_gauss_fit(counts,pix_size = 0.020,arcsec = arcsec):
    def gauss_function(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
        
    popt, pcov = curve_fit(gauss_function, range(len(counts)), counts, p0 = [max(counts), argmax(counts), 5.])
    return popt[2]*pix_size/arcsec    

def get_fwhm_discrete(image,pix_size = 0.020,arcsec = arcsec): 
    disp_dir_counts = sum(image,axis = 1)
    half_max,ind_max = max(disp_dir_counts)/2,argmax(disp_dir_counts)
    ind1,ind2 = argmin(abs(disp_dir_counts[:ind_max] - half_max)),ind_max + argmin(abs(disp_dir_counts[ind_max:] - half_max))
    return (ind2 - ind1)*pix_size/arcsec

def get_hpd_from_image_fit(image,pix_size = 0.020,arcsec = arcsec): 
    cdisp_dir_counts = sum(image,axis = 0)
    fit_levmar = fitting.SimplexLSQFitter()
    gmodel = models.Gaussian1D()
    g = fit_levmar(gmodel,range(len(cdisp_dir_counts)),cdisp_dir_counts,acc = 1e-6)
    return g.stddev*2.35*pix_size/arcsec,cdisp_dir_counts,g(range(len(cdisp_dir_counts)))

def get_hpd_discrete(image,pix_size = 0.020,arcsec = arcsec): 
    cdisp_dir_counts = sum(image,axis = 0)
    prob_dist = cumsum(cdisp_dir_counts)/sum(cdisp_dir_counts)
    ind1,ind2 = argmin(abs(prob_dist - 0.25)),argmin(abs(prob_dist - 0.75))
    return (ind2 - ind1)*pix_size/arcsec

def make_binned_image(rays,pix_size = 0.020,xdiff = 10.,ydiff = 10.):
    #cx_rays,cy_rays = rays[1] - mean(rays[1]), rays[2] - mean(rays[2])
    extent = [[-xdiff/2 + mean(rays[1]),xdiff/2 + mean(rays[1])], [-ydiff/2 + mean(rays[2]),ydiff/2 + mean(rays[2])]]
    fig = plt.figure()
    counts,xedges,yedges,Image = plt.hist2d(rays[1],rays[2],bins = [xdiff/pix_size,ydiff/pix_size],range = extent)
    plt.close(fig)
    return counts

def makeGWATExampleFigure(spoRays,gratRays,diff_title,figure_title = 'ExampleRaytraceOutput.png',scale = Normalize(),diff = 10.):
    # Enable scale = LogNorm() for log plot.
    pix_size = 0.020
    
    fig = plt.figure(figsize = (12,6))
    gs = gridspec.GridSpec(1,2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ymin,ymax = min([amin(spoRays[2]),amin(gratRays[2])]),max([amax(spoRays[2]),amax(gratRays[2])])

    def do_one_axis(ax,rays,diff,title = '0th Order LSF',indices = None):
        #cx_rays,cy_rays = rays[1] - mean(rays[1]), rays[2] - mean(rays[2])
        extent = [[-diff/2 + mean(rays[1]),diff/2 + mean(rays[1])], [-diff/2 + mean(rays[2]),diff/2 + mean(rays[2])]]
        counts,xedges,yedges,Image = ax.hist2d(rays[1],rays[2],bins = diff/pix_size,range = extent,cmap = plt.get_cmap('nipy_spectral'),norm=scale)
        ax.set_xlabel('Dispersion Direction (mm)',fontsize = 16)
        ax.set_ylabel('Cross-Dispersion Direction (mm)',fontsize = 16)
        ax.set_title(title,fontsize = 16)
        
        if indices is None:
            fwhm = do_gauss_fit(sum(counts,axis = 1),pix_size = pix_size)
            hpd = do_gauss_fit(sum(counts,axis = 0),pix_size = pix_size)
        else:
            fwhm = do_gauss_fit(sum(counts[indices[0]:indices[1]],axis = 1),pix_size = pix_size)
            hpd = do_gauss_fit(sum(counts[indices[0]:indices[1]],axis = 0),pix_size = pix_size)

        ax.text(0.05,0.13,'Z Pos: ' + "{:3.1f}".format(mean(rays[3])) + ' mm.',ha = 'left',transform = ax.transAxes,fontsize = 12,color = 'k')
        ax.text(0.05,0.08,'HPD: ' + "{:3.1f}".format(hpd) + ' arcsec.',ha = 'left',transform = ax.transAxes,fontsize = 12,color = 'k')
        ax.text(0.05,0.03,'Disp. FWHM: ' + "{:3.1f}".format(fwhm) + ' arcsec.',ha = 'left',transform = ax.transAxes,fontsize = 12,color = 'k')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.10)
        cbar = plt.colorbar(Image, cax = cax)
        cbar.set_label('Intensity',fontsize = 16)
        return counts, xedges,yedges, Image
        
    foc_counts,xedges,yedges,foc_img = do_one_axis(ax1,spoRays,diff,title = '0th Order LSF',indices = [220,280])
    diff_counts,xedges,yedges,diff_img = do_one_axis(ax2,gratRays,diff,title = diff_title)

    fig.subplots_adjust(top = 0.85,hspace = 0.3,wspace = 0.4)

    plt.savefig(figure_title)
    plt.close()
    return foc_counts,diff_counts,foc_img,diff_img

