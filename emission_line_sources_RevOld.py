from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.interpolate import interp1d
import scipy.stats as stats
import os
import pdb
import copy
import cPickle
import string
import time
import csv

hc = 1239.842   # hc in eV nm.

## Taken from Klauber-1993 Source Functions.
#mg_line_names = ['ka1','ka2','kaprime','ka3','ka3prime','ka4']
#mg_line_energies = 1253.7 + array([0,-0.265,4.740,8.210,8.487,10.095])
#mg_line_widths = [0.26,0.26,1.1056,0.6264,0.7349,1.0007]   # Klauber-1993, with Alpha1,2 modifications to fit 14th diffracted line.
##mg_rel_intensities = [0.5,1.0,0.01027,0.06797,0.03469,0.04905]
#mg_rel_intensities = [0.5,1.0,0.02099,0.07868,0.04712,0.09071]

# Taken from Krause and Ferreira 1975.
mg_line_names = ['ka1','ka2','kadprime','kaprime','ka3','ka4']
mg_line_energies = 1253.7 + array([0,-0.3,3.6,4.6,8.5,10.1])
mg_line_widths = [0.26,0.26,0.5,0.5,0.5,0.5] # Based on best fit to the 14th order diffracted line. #[0.36,0.36,0.5,0.5,0.5,0.5]
mg_rel_intensities = [0.5,1.0,0.005,0.015,0.137,0.077]

def lorentzian(x,loc,scale):
	return 1./(pi*scale*(1 + ((x - loc)/scale)**2))

class emission_line(object):
	def __init__(self, line_names,line_energies,line_widths,rel_intensities,energy_bounds = (1240,1270)):
		self.line_names = line_names
		self.line_energies = line_energies
		self.line_widths = line_widths
		self.rel_intensities = rel_intensities
		self.energy_bounds = energy_bounds
		self.make_line_pdf_energy()
	
	def make_line_pdf_energy(self, ):
		eV = linspace(self.energy_bounds[0],self.energy_bounds[1],10000)
		pdf = zeros(len(eV))
		for i in range(len(self.line_names)):
			pdf = pdf + lorentzian(eV,self.line_energies[i],self.line_widths[i])*self.rel_intensities[i]
		self.energy = hstack((array([eV[0] - (eV[1] - eV[0])]),eV))
		self.wave = hc/self.energy*10**-6
		self.pdf = pdf
		self.pdf_wave = hc/eV*10**-6
		self.cdf = hstack((array([0]),cumsum(pdf)/sum(pdf)))
		
	def draw_waves(self,N = 10**5):
		wave_function = interp1d(self.cdf,self.wave)
		try:
			return wave_function(random.rand(N))
		except ValueError:
			pdb.set_trace()
	
mgk = emission_line(mg_line_names,mg_line_energies,mg_line_widths,mg_rel_intensities)

#mgk_central = 
