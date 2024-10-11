#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:40:56 2023

@author: DhruvB
"""

#helper_functions

from pprint import pprint as pp

import os
import pandas as pd
from astropy.io import fits
from kinms import KinMS
import numpy as np
from astropy.convolution import convolve
from scipy.spatial.distance import pdist, squareform

import sys
sys.path.insert(0,'functions/')
from hdulister import hdulister
from cube_vel_finder import cubevels
from cube_vel_finder import iao

def ecc(a,default=0,data_type=None): #empty_cell_checker
    if isinstance(a, np.ndarray) and a.size == 0:
        is_empty = True
        output = default
    else:
        is_empty = False
        if data_type == int:
            output = int(a)
        if data_type == float:
            output = float(a)
        if data_type == np.ndarray:
            output = np.array(a)
    return {'output':output,'is_empty':is_empty}

def ror(csv_path): #ror = run order reader
    dft = pd.read_csv(csv_path, header=None).T
    dft.dropna(how='all', inplace=True)
    
    csv_path_T = csv_path.split('.csv')[0]+'_T.csv'
    
    dft.to_csv(csv_path_T, header=False, index=False)

    df = pd.read_csv(csv_path_T)
    column_list = list(df.columns)
    
    rod = {}
    for i in column_list:
        values = np.array(df[i].values)[~pd.isna(df[i].values)]
        if len(values) == 1:
            rod[i] = values[0]
        else:
            rod[i] = values

    os.system('rm '+csv_path_T)
    
    cube_path = rod['cube_path']
    
    cube_output = hdulister(cube_path,True)
    cubeheader = cube_output['head']
    
    ordered = iao(cubevels(cubeheader))
    
    if ordered == False:
        if ecc(rod['linefree_min'],None,int)['is_empty'] == False:
            rod['linefree_min'] = int(rod['linefree_min'] - 1)
        if ecc(rod['specmin'],None,int)['is_empty'] == False:
            rod['specmin'] = int(rod['specmin'] - 1)
        if ecc(rod['linefree_max'],None,int)['is_empty'] == False:
            rod['linefree_max'] = int(rod['linefree_max'])
        if ecc(rod['specmax'],None,int)['is_empty'] == False:
            rod['specmax'] = int(rod['specmax'])
    if ordered == True:
        if ecc(rod['linefree_max'],None,int)['is_empty'] == False:
            rod['linefree_max'] = int(rod['linefree_max'] + 1)
        if ecc(rod['specmax'],None,int)['is_empty'] == False:
            rod['specmax'] = int(rod['specmax'] + 1)
        if ecc(rod['linefree_min'],None,int)['is_empty'] == False:
            rod['linefree_min'] = int(rod['linefree_min'])
        if ecc(rod['specmin'],None,int)['is_empty'] == False:
            rod['specmin'] = int(rod['specmin'])
    
    linefree_filled_condition = (ecc(rod['linefree_min'],None,int)['is_empty'] == False and ecc(rod['linefree_max'],None,int)['is_empty'] == False)
    spec_filled_condition = (ecc(rod['specmin'],None,int)['is_empty'] == False and ecc(rod['specmax'],None,int)['is_empty'] == False)
    
    if linefree_filled_condition == True:
        linefree_indices = np.arange(rod['linefree_min'],rod['linefree_max'],1)
        rod['rms_noise'] = rms_noise_computer(cube_path,rod['linefree_min'],rod['linefree_max'])
    if spec_filled_condition == True:
        spec_indices = np.arange(rod['specmin'],rod['specmax'],1)
        
    
    if linefree_filled_condition == True and spec_filled_condition == True:
        overlap = np.intersect1d(linefree_indices, spec_indices)
        spectral_condition = isinstance(overlap, np.ndarray) and overlap.size == 0
        if spectral_condition != True:
            raise ValueError('The line-free and spectral channels overlap. Please re-check your input in '+csv_path)
    
    if spec_filled_condition == False:
        rod['specmin'] = np.nan
        rod['specmax'] = np.nan
    
    rod['thresholds'] = ecc(rod['thresholds'],np.array([rod['rms_noise'],2.0*rod['rms_noise'],3.0*rod['rms_noise']]),np.ndarray)['output']
    
    DMR_ppi_input = rod['DMR_panel_plotting_indices']

    plottings = [False for i in range(len(rod['thresholds']))] #default, to save time
    if DMR_ppi_input == 'all':
        plottings = [True for i in range(len(rod['thresholds']))]
    if DMR_ppi_input == 'none' or (isinstance(DMR_ppi_input, np.ndarray) and DMR_ppi_input.size == 0):
        plottings = [False for i in range(len(rod['thresholds']))]
    if isinstance(DMR_ppi_input,float) == True:
        plottings[int(DMR_ppi_input)] = True
    
    rod['plottings'] = plottings
    
    del rod['DMR_panel_plotting_indices']
    
    return rod

def noise_cube_gen(cubepath,error):
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

def max_dist(array):
    # Flatten the array and get indices of non-NaN elements
    non_nan_indices = np.where(~np.isnan(array))

    # Create a list of non-NaN coordinates
    coordinates = list(zip(non_nan_indices[0], non_nan_indices[1]))

    # If there are less than two non-NaN points, return None
    if len(coordinates) < 2:
        return None

    # Calculate pairwise distances between non-NaN points
    distances = squareform(pdist(coordinates))

    # Find the maximum distance
    max_distance = np.max(distances)

    return max_distance

def all_simcube_info():
    niterations = 1275 #ensure that this is an integer, not float
    PA = 46.0
    RA_deg = 188.0
    DEC_deg = 9.0
    vsys_initial = 0.1
    gas_mass = 10**8.69 #same as the molecular mass of NGC 4189 (Brown et al 2022)
    incs = [30,40,50,60,70,80]
    vertico_beam = [7.5,7.5,0]
    vertico_cell_size = 2.0
    
    viva_beam = [16.8,16.8,0]
    viva_cell_size = 8.0
    res_infos = [[vertico_beam,vertico_cell_size,'vertico'],[viva_beam,viva_cell_size,'viva']]
    SB_infos = [['exp',[10.0,15.0]], ['gaussian',[10.0,35.0,25.0]]]
    noise_levels = [0.5]
#    noise_levels = [0.005,0.0003]
    
#    all_rps_params = ['',[0.1,1.0,60.0],[0.2,1.0,60.0],[0.3,1.0,60.0],[0.4,1.0,60.0]] #array of stripfrac,stripping_dist,stripping_delV (beyond the first option
    stripping_dist = 1.0 #kpc
    stripping_delV = 10.0 #km/s
    
    axis = 0 #0, 1, 2 (for x, y and z respectively), the axis along which ram pressure stripping takes place. 
    
    stripping_fractions = [0.0,0.1,0.2,0.3,0.4]
    
    all_rps_params = []
    for i in range(len(stripping_fractions)):
        all_rps_params.append([stripping_fractions[i],stripping_dist,stripping_delV]) #array of stripfrac,stripping_dist,stripping_delV (beyond the first option
    
    totalcount = len(incs)*len(res_infos)*len(all_rps_params)
    
    info_dict = {'niterations': niterations,
                 'PA': PA,
                 'RA_deg': RA_deg,
                 'DEC_deg': DEC_deg,
                 'vsys_initial': vsys_initial,
                 'gas_mass': gas_mass,
                 'incs': incs,
                 'res_infos': res_infos,
                 'SB_infos': SB_infos,
                 'all_rps_params': all_rps_params,
                 'axis': axis,
                 'noise_levels': noise_levels,
                 'totalcount': totalcount}
    
    return info_dict

def rms_noise_computer(cubepath,linefree_min,linefree_max):
    cube_output = hdulister(cubepath,True)
    cubeheader = cube_output['head']
    cube = cube_output['cube']
    cdelt1 = cube_output['cdelt1']
    cdelt2 = cube_output['cdelt2']
    x1=((np.arange(0,cubeheader['NAXIS1'])-(cubeheader['NAXIS1']//2))*cdelt1)# + hdr['CRVAL1']
    y1=((np.arange(0,cubeheader['NAXIS2'])-(cubeheader['NAXIS1']//2))*cdelt2)# + hdr['CRVAL2']
    quarterx=np.array(x1.size/4.).astype(int)
    quartery=np.array(y1.size/4.).astype(int)
    cropped_noise_cube = cube[linefree_min:linefree_max,1*quartery:3*quartery,quarterx*1:3*quarterx]
    rmsnoise = np.nanstd(cropped_noise_cube)
    
    return rmsnoise