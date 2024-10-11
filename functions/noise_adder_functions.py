#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:14:03 2023

@author: DhruvB
"""

#noise_adder_functions.py

"""
    
Function noise_cube_gen by Tim Davis

"""

import numpy as np
from kinms import KinMS
from astropy.io import fits
from astropy.convolution import convolve

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

def PB_corrected_noise_cube_gen(noise_cube,PB_path): #only necessary for VERTICO galaxies
    hdu_pb = fits.open(PB_path)
    pb_cube = hdu_pb[0].data
    pbcorr_noise_cube = []
    for frame in noise_cube:
        pbcorr_frame = np.array(frame)/np.array(pb_cube[0])
        pbcorr_noise_cube.append(pbcorr_frame)
    return pbcorr_noise_cube

def error_finder(cubepath,desired_value,tolerance=0.01): #here, the cubepath is the simulated cube
    desired_value = desired_value
    lower_bound = 0
    upper_bound = 100  # Adjust this based on your problem domain

    output = 6000
    
    # Perform the bisection search
    while (np.abs(desired_value - output) / desired_value) > tolerance:
        mid = (lower_bound + upper_bound) / 2.0
        
        noise_cube = noise_cube_gen(cubepath,mid) # Replace `your_function` with your actual function
        output = np.nanstd(noise_cube)

        print(mid,output,np.round(100.0*np.abs(desired_value - output) / desired_value,2))
        if (np.abs(desired_value - output) / desired_value) > tolerance:
            if output < desired_value:
                lower_bound = mid
            else:
                upper_bound = mid
        else:
            break
    # Return the input within the desired tolerance
    return mid, noise_cube