from numpy import *
import matplotlib.pyplot as plt
import os
import glob
import pickle
import astropy.io.fits as pyfits
from scipy.special import wofz,beta,gamma
from scipy.optimize import curve_fit,minimize
from scipy.interpolate import interp1d
from astropy.modeling import models,fitting
from copy import deepcopy

from matplotlib import gridspec
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdb

import utilities.imaging.man as man

import gwatAnalTrace.gwat_analysis_utilities as gwatutil
import gwatAnalTrace.emission_line_sources as emlines

#spo_lwidth = array(0.100)
arcsec = 12000.*pi/180/3600
pix_size = 0.020
hc = 1239.842   # hc in eV nm.
#raytrace_position = (881.118433961543)
#est_disp = hc/mg_line_energies[1]*10**-6/raytrace_position  # Computed by taking wavelength center of Mg K and dividing by the raytrace-computed best focus dispersion position of -881.118433961543 mm.
#
#mgk = emlines.emission_line(mg_line_names,mg_line_energies,mg_line_widths,mg_rel_amplitudes,energy_bounds = (1240,1270))

#########################################################
# For fitting the dispersion direction of the SPO profile and performing the resolution convolution.
#########################################################

def lorentzian(x,x0,fwhm):
	'''
	Gives the Lorentzian distribution based on a central location x0 and a FWHM. Note that
	the FWHM is NOT equal to the gamma parameter of the Lorentzian: 2*gamma = FWHM.
	'''
	gamma = fwhm/2
	return 1./(pi*gamma*(1 + ((x - x0)/gamma)**2))

def wave_gaussian(x,R,center_wave):
	sigma = center_wave/2.35/R
	return 1./sqrt(2*pi)/sigma*exp(-(x - center_wave)**2/(2*sigma**2))

#########################################################
# For comparing the line profiles.
#########################################################

def normalize_and_align_profiles(mgk_x,mgk_pdf,data_x,data_pdf,bkgd = 0.0):
	# Aligning the x data to a common grid, and normalizing by the maximum in the pdf.
	mgk_xalign = mgk_x - mgk_x[argmax(mgk_pdf)]
	mgk_pdf_norm = mgk_pdf/max(mgk_pdf)
	data_xalign = data_x - data_x[argmax(data_pdf)]
	data_pdf_norm = data_pdf/max(data_pdf)
	
	#pdb.set_trace()
	# Finally, interpolating the model onto the data coordinates.
	f = interp1d(mgk_xalign,mgk_pdf_norm,'cubic')
	mgk_resample = f(data_xalign) + bkgd
	
	return data_xalign,data_pdf_norm,mgk_resample

def normalize_and_align_subprofile(full_x,full_pdf,sub_x,sub_pdf,data_x,data_pdf,bkgd = 0.0):
	# Computing the offsets that need to be applied to the subprofile data.
	sim_offset = full_x[argmax(full_pdf)]
	sim_renorm = 1./max(full_pdf)
	
	sub_xalign = sub_x - sim_offset
	sub_pdf_norm = sub_pdf*sim_renorm
	data_xalign = data_x - data_x[argmax(data_pdf)]

	# Finally, interpolating the model onto the data coordinates.
	f = interp1d(sub_xalign,sub_pdf_norm,'cubic')
	sub_profile_resample = f(data_xalign) + bkgd
	
	return data_xalign,sub_profile_resample

#########################################################
# Fitting functions.
#########################################################
# Common merit function.
def fitting_merit_function(model,data,error = None):
	if error is None:
		error = ones(len(model))
	merit = sum(((data - model)/error)**2)
	#merit = abs(sum((data - model)/weights))
	
	if type(merit) is not float64:
		pdb.set_trace()
	else:	
		return merit

#####
# For fitting the SPO focus.
#####
def model_line_profile_focus(l,data_x,data_pdf,model_kind = lorentzian):
	bin_size = data_x[1] - data_x[0]
	model_x = arange(-len(data_x)*1.1/2,len(data_x)*1.1/2)*bin_size
	model_pdf = model_kind(model_x,0.,l)
	
	data_xalign,data_pdf_norm,mgk_resample = normalize_and_align_profiles(model_x,model_pdf,data_x,data_pdf,bkgd = 0.0)
	return data_xalign,data_pdf_norm,mgk_resample
	
def fit_line_function_focus(l,data_x,data_pdf,model_kind = lorentzian):
	data_xalign,data_pdf_norm,mgk_resample = model_line_profile_focus(l,data_x,data_pdf,model_kind)
	merit = fitting_merit_function(mgk_resample,data_pdf_norm,error = sqrt(data_pdf_norm))
	return merit

#####
# For fitting the full model.
#####

def model_line_profile_resblur(variables,data_x,data_pdf,spo_lwidth,dispersion,line_model,plotting_unpack = True):	
	# First, unpacking the variables.
	[R,bkgd] = variables
	
	# Second, making the line models by loading this from emlines package.
	mgk_model = deepcopy(line_model) 

	def do_res_convolve(pdf,pdf_wave,center_wave,R):
		# Next, making the resolution blur. We convolve this resolution with the Mg K Alpha line structure to make a blurred line structure.
		res_gaussian = wave_gaussian(pdf_wave,R,center_wave)
		bin_size = pdf_wave[1] - pdf_wave[0]
		blurred_pdf = convolve(pdf,res_gaussian,mode = 'same')
		norm = sum(convolve(ones(len(pdf)),res_gaussian,mode = 'same'))
		return blurred_pdf/norm
	
	def do_phy_space_convert(pdf_wave,dispersion):
		real_x = pdf_wave/dispersion
		return real_x 
	
	def do_spo_convolve(phy_x,profile,spo_vars):
		# Next, doing the convolution with the SPO profile.
		x_bin_size = abs(phy_x[1] - phy_x[0])
		#spo_x = arange(-len(phy_x)/2,len(phy_x)/2,1)*x_bin_size
		spo_profile = lorentzian(phy_x,mean(phy_x),spo_vars)
		convolved_profile = convolve(spo_profile,profile,mode = 'same')
		norm = sum(convolve(spo_profile,ones(len(profile)),mode = 'same'))
		return spo_x,convolved_profile/norm
	
	def do_full_convolve(line_model,R,spo_vars,dispersion):
		blurred_profile = do_res_convolve(line_model.pdf,line_model.pdf_wave,hc/line_model.line_energies[0]*10**-6,R)
		real_x = do_phy_space_convert(line_model.pdf_wave,dispersion)
		sim_x,sim_profile = do_spo_convolve(real_x,blurred_profile,spo_vars)
		return sim_x,sim_profile
	
	# Getting the full simulated profile (both lines, SPO convolution, resolution degradation.)
	full_sim_x,full_sim_profile = do_full_convolve(mgk_model,R,spo_lwidth,dispersion)
	data_xalign,data_profile_norm,full_sim_norm = normalize_and_align_profiles(full_sim_x,full_sim_profile,data_x,data_pdf,bkgd = bkgd)
	
	if plotting_unpack:
		ka1_model = emlines.emission_line([mgk_model.line_names[0]],[mgk_model.line_energies[0]],[mgk_model.line_widths[0]],[mgk_model.rel_amplitudes[0]])
		ka2_model = emlines.emission_line([mgk_model.line_names[1]],[mgk_model.line_energies[1]],[mgk_model.line_widths[1]],[mgk_model.rel_amplitudes[1]])
		
		# Getting just the underlying line profiles (i.e. no SPO, no resolution degradation) for each line component and the whole.
		ka1_sim_x,ka1_sim_profile = do_full_convolve(ka1_model,1e7,1e-6,dispersion)
		ka2_sim_x,ka2_sim_profile = do_full_convolve(ka2_model,1e7,1e-6,dispersion)
		mgk_sim_x,mgk_sim_profile = do_full_convolve(mgk_model,1e7,1e-6,dispersion)
		# Getting the line profiles with an SPO (i.e. no resolution degradation) for the whole line.
		line_spo_sim_x,line_spo_sim_profile = do_full_convolve(mgk_model,1e7,spo_lwidth,dispersion)
		
		trash,ka1_sim_norm = normalize_and_align_subprofile(full_sim_x,full_sim_profile,full_sim_x,ka1_sim_profile,data_x,data_pdf,bkgd = bkgd)
		trash,ka2_sim_norm = normalize_and_align_subprofile(full_sim_x,full_sim_profile,full_sim_x,ka2_sim_profile,data_x,data_pdf,bkgd = bkgd)
		trash,mgk_sim_norm = normalize_and_align_subprofile(full_sim_x,full_sim_profile,full_sim_x,mgk_sim_profile,data_x,data_pdf,bkgd = bkgd)
		trash,line_spo_sim_norm = normalize_and_align_subprofile(full_sim_x,full_sim_profile,full_sim_x,line_spo_sim_profile,data_x,data_pdf,bkgd = bkgd)
	
		return data_xalign,data_profile_norm,full_sim_norm,ka1_sim_norm,ka2_sim_norm,mgk_sim_norm,line_spo_sim_norm
	else:
		return data_xalign,data_profile_norm,full_sim_norm
	
def fit_line_function_resblur(variables,data_x,data_pdf,spo_lwidth,dispersion,line_model):
	data_xalign,data_profile,model_profile = model_line_profile_resblur(variables,data_x,data_pdf,spo_lwidth,dispersion,line_model,plotting_unpack = False)
	merit = fitting_merit_function(model_profile,data_profile,error = sqrt(data_profile))
	return merit

#########################################################
# Plotting functions.
#########################################################

def plot_bestResFit_diffLine(variables,diff_x,diff_disp,spo_lwidth,dispersion,line_model,
							 title = 'Diffracted 14th Order Mg K Alpha,\nMeasurement and Best Fit Model',
							 fig_name = '180409_BestFitResolution_14thOrder_MgKAlpha_Klauber1993.png'):
	data_xalign,data_profile_norm,full_sim_norm,ka1_sim_norm,ka2_sim_norm,mgk_sim_norm,line_spo_sim_norm = model_line_profile_resblur(variables,diff_x,diff_disp,spo_lwidth,dispersion,line_model,plotting_unpack = True)
	#plt.ion()
	plt.figure(figsize = (8,8))
	
	# Plot measured data.
	plt.plot(data_xalign,data_profile_norm,label = 'Measured')
	
	# Plotting line contributions.
	plt.plot(data_xalign,ka1_sim_norm,'r',linestyle = 'dotted',label = 'Mg K' + r'$\alpha_1$')
	plt.plot(data_xalign,ka2_sim_norm,'r',linestyle = 'dotted',label = 'Mg K' + r'$\alpha_2$')
	plt.plot(data_xalign,mgk_sim_norm,'r',label = 'Mg K' + r'$\alpha_{1,2}$')
	
	# Plotting SPO contribution to line.
	plt.plot(data_xalign,line_spo_sim_norm,linestyle = 'dashed',label = 'Mg K' + r'$\alpha_{1,2}$' + 'w. SPO Contribution')
	
	# Plotting full simulation.
	plt.plot(data_xalign,full_sim_norm,label = 'R = ' + "{:5.0f}".format(variables[0]))
	
	plt.xlabel('Dispersion Direction (mm)')
	plt.ylabel('Normalized Intensity')
	
	plt.legend()
	plt.title(title)
	plt.savefig(fig_name)
	
	#savetxt('180408_ResolvingPowerFitOutput_FullModel.txt',output.x)
