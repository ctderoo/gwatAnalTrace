from numpy import *
import matplotlib.pyplot as plt
import os
import glob
import pickle
import astropy.io.fits as pyfits
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from astropy.modeling import models,fitting
from copy import deepcopy
import pdb

import arcusTrace.arcusRays as ArcRays

import gwatAnalTrace.trace_GWAT_Yaw1p985 as traceGWAT
import gwatAnalTrace.gwat_funcs as gwatf
import gwatAnalTrace.create_gwat_hardware as gwathw

home_directory = os.getcwd()
figure_directory = home_directory + '/Figures'
arcsec = 12000.*pi/180/3600
pix_size = 0.020
disp_width = 2500
cross_width = 12500

#################################################
# Profiling and Comparison Functions
#################################################

def cut_sample(data,width):
	maxind = argmax(data)
	return data[maxind - width:maxind + width]

def filter_profile(counts,window_size,poly):
	return savgol_filter(counts,window_size,poly)

def smooth_resample(counts,width = 0,pix_size = 0.020,arcsec = arcsec,window_size = None):
	if window_size is not None:
		filt_data = filter_profile(counts,window_size,poly) #savgol_filter(counts,window_size,poly)
	else:
		filt_data = counts
	data_func = interp1d(range(len(filt_data)),filt_data,kind = 3)
	bin_size = arcsec*0.01/pix_size    # Given in pixels.
	x_samp = arange(0,len(filt_data) - 1,bin_size)
	pred_data = data_func(x_samp)
	pred_data = pred_data - median(pred_data)
	cut_data = cut_sample(pred_data,width) 
	return cut_data,bin_size

def comp_hpd_interp(pred_data,bin_size,pix_size = 0.020,arcsec = arcsec):
	#pred_data,bin_size = smooth_resample(counts)
	prob_dist = cumsum(pred_data)/sum(pred_data)
	ind1,ind2 = argmin(abs(prob_dist - 0.25)),argmin(abs(prob_dist - 0.75))
	return (ind2 - ind1)*bin_size*pix_size/arcsec#,filt_data,data_func,x_samp

def comp_fwhm_interp(pred_data,bin_size,pix_size = 0.020,arcsec = arcsec):
	#pred_data,bin_size = smooth_resample(counts)
	half_max,ind_max = max(pred_data)/2,argmax(pred_data)
	ind1,ind2 = argmin(abs(pred_data[:ind_max] - half_max)),ind_max + argmin(abs(pred_data[ind_max:] - half_max))
	return (ind2 - ind1)*bin_size*pix_size/arcsec

def CoM(profile):
	vec = profile*arange(len(profile))
	return sum(vec)/sum(profile)

def norm_for_compare(simprofile,realprofile,bin_size,pix_size = pix_size,arcsec = arcsec):
	#offset = argmax(simprofile) - argmax(realprofile)
	offset = int(CoM(simprofile) - CoM(realprofile))
	if offset > 0:
		newsimprofile = simprofile[offset:]
	else:
		newsimprofile = hstack((zeros(-offset),simprofile))

	# Cutting to the same length.
	length = min([len(newsimprofile),len(realprofile)])
	newsimprofile,realprofile,xpos = newsimprofile[:length],realprofile[:length],arange(length)*bin_size*pix_size/arcsec
	
	# Normalizing.
	newsimprofile,realprofile = newsimprofile/max(newsimprofile),realprofile/max(realprofile)
	return newsimprofile,realprofile

#################################################
# Comparing Simulation and Measurement for CSR figure.
#################################################

#def make_raytrace_measurements(order_repeats = 50,focus_repeats = 1,
#							   diff_pickle_file = pickle_directory + '/180324_DiffRays_MgK14_orderStats.pk1',
#							   foc_pickle_file = pickle_directory + '/180324_FocRays_MgK14_orderStats.pk1',wave = wave):
#	if order_repeats > 0:
#		traceGWAT.trace_gwat_orderStats(pickle_file = diff_pickle_file,repeats = order_repeats,wave  = wave)
#	if focus_repeats > 0:
#		traceGWAT.trace_gwat_orderStats(pickle_file = foc_pickle_file,repeats = focus_repeats,kind = 'focus',wave = wave)
#
#def load_raytrace_measurements(diff_pickle_file = pickle_directory + '/180324_DiffRays_MgK14_orderStats.pk1',
#							   foc_pickle_file = pickle_directory + '/180324_FocRays_MgK14_orderStats.pk1') :
#	diffRays = ArcRays.load_ray_object_from_pickle(diff_pickle_file)
#	focRays = ArcRays.load_ray_object_from_pickle(foc_pickle_file)
#	
#	diffSimImg = gwatf.make_binned_image(diffRays.yield_prays(),xdiff = 26.,ydiff = 26.8)
#	focSimImg = gwatf.make_binned_image(focRays.yield_prays(),xdiff = 26,ydiff = 26.8)
#	return diffSimImg,focSimImg,diffRays,focRays

def create_profiles(simImg,dataImg,disp_width = disp_width,cross_width = cross_width):
	'''
	Takes a simulated image and a real data image, and returns dispersion direction and cross-dispersion
	direction profiles for a side-by-side comparison.
	Returns:
	nsimdisp -- the normalized, matched simulated dispersion profile.
	nrealdisp -- the normalized, matched measured dispersion profile.
	nsimcross -- the normalized, matched simulated cross-dispersion profile.
	nrealcross -- the normalized, matched measured cross-dispersion profile.
	'''
	realdisp,bin_size = gwatprocomp.smooth_resample(sum(dataImg,axis = 1),width = disp_width)
	realcross,bin_size = gwatprocomp.smooth_resample(sum(dataImg,axis = 0),width = cross_width)
	simdisp,bin_size = gwatprocomp.smooth_resample(sum(flipud(simImg),axis = 1),width = disp_width)
	simcross,bin_size = gwatprocomp.smooth_resample(sum(flipud(simImg),axis = 0),width = cross_width)
	
	nsimdisp,nrealdisp = gwatprocomp.norm_for_compare(simdisp,realdisp,bin_size)
	nsimcross,nrealcross = gwatprocomp.norm_for_compare(simcross,realcross,bin_size)
	return nsimdisp,nrealdisp,nsimcross,nrealcross,bin_size

def make_comparison_profile_plot(sim_profile,meas_profile,bin_size,kind = 'Disp',title = 'Measured and Simulated Profiles of a Diffracted Emission Line,\n14th Order Mg K',plot_fn = None):
	step_arcsec = bin_size*pix_size/arcsec
	angle = linspace(-len(sim_profile)/2*step_arcsec,len(sim_profile)/2*step_arcsec,len(sim_profile))
	
	fig = plt.figure(figsize = (7,7))
	ax = plt.gca()
	ax.plot(angle,sim_profile,label = 'Sim. ' + kind)
	ax.plot(angle,meas_profile,label = 'Meas. ' + kind)
	ax.set_xlabel('Arcseconds')
	ax.set_ylabel('Normalized Intensity')
	
	plt.legend(loc = 'upper right')
	if kind == 'Disp':
		plt.text(0.03,0.95,'Sim. FWHM: ' + '{:3.2f}'.format(comp_fwhm_interp(sim_profile,bin_size)) + ' asec.',ha = 'left',transform = ax.transAxes)
		plt.text(0.03,0.90,'Meas. FWHM: ' + '{:3.2f}'.format(comp_fwhm_interp(meas_profile,bin_size)) + ' asec.',ha = 'left',transform = ax.transAxes)
	elif kind == 'Cross':
		plt.text(0.03,0.95,'Sim. HPD: ' + '{:3.2f}'.format(comp_hpd_interp(sim_profile,bin_size)) + ' asec.',ha = 'left',transform = ax.transAxes)
		plt.text(0.03,0.90,'Meas. HPD: ' + '{:3.2f}'.format(comp_hpd_interp(meas_profile,bin_size)) + ' asec.',ha = 'left',transform = ax.transAxes)
	ax.set_title(title)
	
	if plot_fn is not None:
		plt.savefig(plot_fn)
		plt.close()
		
def add_random_background(simimg,count_density):
	ysize,xsize = shape(simimg)
	imgsize = size(simimg)
	N_back = int(count_density*imgsize)
	
	simimg1d = simimg.ravel()
	
	for N in range(N_back):
		simimg1d[randrange(0,imgsize)] = 1
	
	return simimg1d.reshape(ysize,xsize)

def get_CoM_img(img):
	profile_x,profile_y = sum(img,axis = 0),sum(img,axis = 1)
	xCoM,yCoM = gwatprocomp.CoM(profile_x),gwatprocomp.CoM(profile_y)
	return xCoM,yCoM

def make_aligned_images(measimg,simimg,final_size = [400,200,200,200],offset = 18):
	simimg = flipud(simimg)
	simimg_padded = man.padRect(simimg,1300)
	simimg_padded[isnan(simimg_padded)] = 0.0
	
	meas_profile_x,meas_profile_y = sum(measimg,axis = 0),sum(measimg,axis = 1)
	meas_xCoM,meas_yCoM = get_CoM_img(measimg)
	sim_xCoM,sim_yCoM = get_CoM_img(simimg_padded)
	
	#pdb.set_trace()
	meas_xCoM,meas_yCoM,sim_xCoM,sim_yCoM = int(meas_xCoM),int(meas_yCoM),int(sim_xCoM),int(sim_yCoM) - offset
	
	cut_meas_img = measimg[meas_yCoM - final_size[0]:meas_yCoM + final_size[1],meas_xCoM - final_size[2]:meas_xCoM + final_size[3]]
	cut_sim_img = simimg_padded[sim_yCoM - final_size[0]:sim_yCoM + final_size[1],sim_xCoM - final_size[2]:sim_xCoM + final_size[3]]
	
	return rot90(cut_meas_img),rot90(cut_sim_img)

def make_normalized_profiles(img):
	xprofile,yprofile = sum(img,axis = 0),sum(img,axis = 1)
	norm_xprofile,norm_yprofile = xprofile/max(xprofile),yprofile/max(yprofile)
	return norm_xprofile,norm_yprofile

#################################################
# Image analysis utilities.
#################################################

def shift_image(img,xshift,yshift):
	ysize,xsize = shape(img)
	buffer_ind = max([xshift,yshift])
	expand_img = man.padRect(img,buffer_ind)
	shift_img = expand_img[buffer_ind - yshift:buffer_ind + ysize - yshift,buffer_ind - xshift:buffer_ind + xsize - xshift]
	shift_img[isnan(shift_img)] = 0.0
	return shift_img
