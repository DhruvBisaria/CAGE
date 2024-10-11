#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:35:14 2023

@author: DhruvB
"""

#simcube_noise_adder.py

from kinms import KinMS
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from astropy.io import fits
from astropy.convolution import convolve

import sys
sys.path.insert(0,'../functions/')
from hdulister import hdulister
from noise_adder_functions import error_finder
from dendrogram_functions import cubewriter

def noise_cube_gen(cubepath,error): #Works for both VERTICO and VIVA galaxies
    hdul=fits.open(cubepath)
    cubedata = hdul[0].data
    cubeheader = hdul[0].header
    beamSize=[cubeheader['BMAJ']*3600.0,cubeheader['BMIN']*3600.0,cubeheader['BPA']]
    cellSize=np.abs(cubeheader['CDELT1']*3600.0)
    psf=KinMS.makebeam(1,cubeheader['NAXIS1'], cubeheader['NAXIS2'], beamSize,cellSize=cellSize)
    noise=np.random.normal(size=cubedata.shape)
    for i in range(noise.shape[0]):
        noise[i, :, :] = convolve(noise[i, :, :], psf)
    noise_cube = noise*(error/np.nanstd(noise))
    return noise_cube

def noisy_cube_generator(cubepath,kinmscubepath,tolerance):

    co = hdulister(cubepath,True)
    input_cube = np.array(co['cube'])
    stdcube = np.nanstd(input_cube[0])
    
    kco = hdulister(kinmscubepath+'.fits',True)
    kh = kco['head']
    kinms_cube = np.array(kco['cube'])
    
    ideal_input, noise_cube = error_finder(kinmscubepath+'.fits',stdcube,tolerance)
    
    noisy_cube = kinms_cube + noise_cube
    
    noisycubepath = kinmscubepath+'_noise_added.fits'
    cubewriter(noisy_cube,kh,'simulated_cube_plus_noise '+str(ideal_input),noisycubepath)
    return noisycubepath