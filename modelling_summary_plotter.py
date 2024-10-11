#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:55:21 2023

@author: DhruvB
"""

#simulated_cube_summary_plot.py

import numpy as np
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
import astropy.units as u
import matplotlib as mpl
from matplotlib.patches import Ellipse
from astropy.wcs import WCS
import argparse
import scipy.ndimage as ndimage
import csv
import os
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
from astropy.io import fits
from helper_functions import ror

import sys
sys.path.insert(0,'../functions/')
from hdulister import hdulister
from kinms_param_extractor import kpe
import SB_profiles

my_parser = argparse.ArgumentParser()

my_parser.add_argument('csv_path', type=str, help='the path to the csv input file.')
args = my_parser.parse_args()
csv_path = args.csv_path

rod = ror(csv_path)

PA_guess = float(rod['PA_guess'])
i_guess = float(rod['i_guess'])
x_center_guess = float(rod['x_center_guess'])
y_center_guess = float(rod['y_center_guess'])
vsys_guess = float(rod['vsys_guess'])
num_rings = int(rod['num_rings']) #very important that num_rings be int
SB_form = str(rod['SB_form'])
SB_params = rod['SB_params']
cube_path = str(rod['cube_path'])
results_dir = str(rod['results_dir'])
run_label = str(rod['run_label'])
npz_path = results_dir+'modelling_results/'+run_label+'_results'
mask_cube_path = str(rod['mask_cube_path'])

#Second, grab the fitted parameters
output_dict = kpe(npz_path,cube_path,SB_form)

PeakFlux_exp = output_dict['PeakFlux_exp']
Rscale_exp = output_dict['Rscale_exp']
PeakFlux_gaussian = output_dict['PeakFlux_gauss']
Mean_gauss = output_dict['Mean_gauss']
sigma_gauss = output_dict['sigma_gauss']
peak_modsersic = output_dict['peak_modsersic']
mean_modsersic = output_dict['mean_modsersic']
sigma_modsersic = output_dict['sigma_modsersic']
index_modsersic = output_dict['index_modsersic']

fittedPA = output_dict['PA'][0]
fittedinc = output_dict['inc'][0]
incratio = np.sin( np.deg2rad(fittedinc) )/np.sin( np.deg2rad(i_guess))
angradius = np.array(output_dict['angradius'])
Vt = np.array(output_dict['Vt'])*incratio
eVt = np.array(output_dict['eVt'])
xc_pix = output_dict['xc_pix']
yc_pix = output_dict['yc_pix']

fig = plt.figure(figsize=(16, 4))
axRC = fig.add_subplot(131)

raw_x = np.arange(0, 100, 0.1)

cube_output = hdulister(cube_path,True)
cubeheader = cube_output['head']
cdelt1 = cube_output['cdelt1']
bmajor = cube_output['bmajor']

r_ub = num_rings*bmajor
i2c = len(raw_x)
for i in range(len(raw_x)):
    if r_ub < raw_x[i]:
        i2c = i
        break

x = raw_x[0:i2c]

#Vrot = 180.0
#velfunc = interpolate.interp1d([0, 0.5, 1, 3, 128], [0, 50, 100, Vrot, Vrot], kind='linear')
#velparams=np.array([180.0,1.0])
#vel_profile=[velocity_profs.arctan(guesses=velparams, minimums=np.array([50,0.01]), maximums=[220,10])]

#raw_vel = velfunction(180.0,0.1,raw_x)    
#vel = raw_vel[0:i2c]    
#def velfunction(x): polyex function, see Catinella
#    return 188.0*(1 - np.exp(-x/(10)))*(1 + (0.012*x)/10) 
#vel = velfunc(x)

#axRC.plot(raw_x,raw_vel,color='k',alpha=0.5)
#axRC.plot(x,vel,color='k')

radius_shift = 0.5*np.nanmean(np.diff(angradius)) #to account for binning

angradius = np.array(angradius) #+ radius_shift

axRC.scatter(angradius,Vt)
axRC.errorbar(angradius,Vt,yerr=eVt)

axRC.set_ylim([0,1.1*np.max(np.array(Vt)+np.array(eVt))])

axSB = fig.add_subplot(132)

#Plot the SB profiles:
if SB_form == 'exp' or SB_form == 'gaussian' or SB_form == 'sersic':
    I_init_max = SB_params[0]
if SB_form == 'exp_gaussian' or SB_form == 'exp_sersic':
    I_init_max = SB_params[0] + SB_params[2]
if SB_form == 'gaussian_sersic':
    I_init_max = SB_params[0] + SB_params[3]

if SB_form == 'exp':
    SB_init = SB_profiles.exp(raw_x,SB_params[0],SB_params[1])
    scaling_factor = PeakFlux_exp[0]/I_init_max
    SB_fitted = np.array(SB_profiles.exp(raw_x,PeakFlux_exp[0],Rscale_exp[0])) / scaling_factor
if SB_form == 'gaussian':
    SB_init = SB_profiles.gaussian(raw_x,SB_params[0],SB_params[1],SB_params[2])
    scaling_factor = PeakFlux_gaussian[0]/I_init_max
    SB_fitted = np.array(SB_profiles.gaussian(raw_x,PeakFlux_gaussian[0],Mean_gauss[0],sigma_gauss[0])) / scaling_factor
if SB_form == 'sersic':
    SB_init = SB_profiles.sersic(raw_x,SB_params[0],SB_params[1],SB_params[2],SB_params[3])
    SB_fitted = SB_profiles.sersic(raw_x,peak_modsersic[0],mean_modsersic[0],sigma_modsersic[0],index_modsersic[0])
if SB_form == 'exp_gaussian':
    SB_init = SB_profiles.exp_gaussian(raw_x,SB_params[0],SB_params[1],SB_params[2],SB_params[3],SB_params[4])
    scaling_factor = (PeakFlux_exp[0] + PeakFlux_gaussian[0])/I_init_max
    SB_fitted = np.array(SB_profiles.exp_gaussian(raw_x,PeakFlux_exp[0],Rscale_exp[0],PeakFlux_gaussian[0],Mean_gauss[0],sigma_gauss[0])) / scaling_factor
if SB_form == 'exp_sersic':
    SB_init = SB_profiles.exp_sersic(raw_x,SB_params[0],SB_params[1],SB_params[2],SB_params[3],SB_params[4],SB_params[5])
    SB_fitted = SB_profiles.exp_sersic(raw_x,PeakFlux_exp[0],Rscale_exp[0],peak_modsersic[0],mean_modsersic[0],sigma_modsersic[0],index_modsersic[0])
if SB_form == 'gaussian_sersic':
    SB_init = SB_profiles.gaussian_sersic(raw_x,SB_params[0],SB_params[1],SB_params[2],SB_params[3],SB_params[4],SB_params[5],SB_params[6])
    SB_fitted = SB_profiles.gaussian_sersic(raw_x,PeakFlux_gaussian[0],Mean_gauss[0],sigma_gauss[0],peak_modsersic[0],mean_modsersic[0],sigma_modsersic[0],index_modsersic[0])

SB_init = np.array(SB_init)
OGint = np.trapz(SB_init,raw_x)
func_int = np.trapz(SB_fitted,raw_x)

scaling_factor = OGint/func_int
SB_fitted = np.array(SB_fitted)*scaling_factor

axSB.plot(raw_x,SB_init,color='grey',alpha = 0.5)
axSB.scatter(raw_x,SB_fitted,color='grey',alpha=0.5,s=0.85)

axSB.plot(x,SB_init[0:i2c],color='k')
axSB.scatter(x,SB_fitted[0:i2c],color='b',s=0.85)

axMM = fig.add_subplot(133)

#Now to generate the moment 1 map.
#First, 
maskpath = mask_cube_path
mask_cube = hdulister(maskpath)['cube']
ch = hdulister(cube_path)['head']
cube_data = hdulister(cube_path)['cube']

#mask_sum = np.any(mask_cube, axis=0).astype(int) #binary_map = mask_sum.astype(int)
#    
##    mask_2d = use_mask_cube.moment(order=0) > 0
#
## Number of pixels to dilate by
##    npix_dilation = np.ceil(abs(cube_k.header["BMAJ"] / cube_k.header["CDELT1"]) * nbeam_dilation).astype(int); nbeam_dilation = 1
#npix_dilation = np.ceil(abs(ch['BMAJ'] / ch['CDELT1']) * 1).astype(int)
#
## the starting point is the 2d mask
#dilated_mask_2d = mask_sum
#
## binary_dilation of mask by n pix
#for j in range(npix_dilation):
#    struct = ndimage.generate_binary_structure(2, j)
#    dilated_mask_2d = ndimage.binary_dilation(dilated_mask_2d, structure=struct)
#
#use_mask_2d = dilated_mask_2d.astype(int)

#mask_sum = np.any(mask_cube, axis=0)

cube = SpectralCube.read(cube_path).with_spectral_unit(u.km / u.s, rest_value=ch['RESTFRQ']*u.Hz, velocity_convention="optical")
#filter the noisy cube (just as an array) by the mask
cube_f = cube_data * mask_cube
#Now turn that nsc_f into a Spectral Cube object itself:
cube_f_sc = SpectralCube(cube_f,wcs=cube.wcs,header=cube.header)
galmap = np.array(cube_f_sc.moment(order=1).data)
#
axMM.imshow(galmap,origin='lower',cmap=plt.cm.coolwarm)

#Now, plot the rings on the moment map
bmajor = ch['BMAJ']*3600.0
bminor = ch['BMIN']*3600.0
bpa = ch['BPA']
cdelt1 = np.abs(ch['CDELT1'])
cdelt2 = np.abs(ch['CDELT2'])
BPA = np.deg2rad(bpa+90.0)
arcsperpix = np.abs(cdelt1)*3600.0 #convert degrees to arcseconds
pixperarc = 1.0/arcsperpix #pixels per arcseconds
bminorpix = bminor*pixperarc
bmajorpix = bmajor*pixperarc
maxdist = 1.1*np.max([bminorpix,bmajorpix])
ellipse_xc = maxdist
ellipse_yc = maxdist
ringpixstep = bmajorpix

rings = np.arange(0,num_rings*ringpixstep,ringpixstep)
mom1axisratio = np.cos(np.deg2rad(fittedinc))
for n in range(len(rings)):
    axMM.add_patch(Ellipse((xc_pix,yc_pix),2.0*rings[n],2.0*rings[n]*mom1axisratio,fittedPA + 90.0,edgecolor='k',facecolor='none',lw=0.5))
    if n == 0:
        axMM.scatter(ch['CRPIX1']-1,ch['CRPIX2']-1,s=1,color='hotpink')
        axMM.scatter(xc_pix,yc_pix,s=1,color='k')


plt.savefig(results_dir+'modelling_results/summary.png',dpi=300,bbox_inches='tight')
plt.close()
