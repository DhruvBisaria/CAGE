#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 00:42:32 2023

@author: DhruvB
"""

#pixel_masser.py

#This code generates a mass for each pixel 

plotting = 'yes'

import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u
from astropy import constants as const

import sys
sys.path.insert(0,'../functions/')
#from cube_path_generator import maskpath_VIVA
from hdulister import hdulister
from cube_vel_finder import cubevels

virgo_distance = 16.5 * u.Mpc  # Mei+05
#virgo_distance_chung09 = 16.0 * u.Mpc
#co21_rest = 230.538 * u.GHz  # CO (2-1)
#HI_rest = 1.42041 * u.GHz
R21_vertico = 0.8  # from VERTICO-HRS comparison
alpha_co_10 = 4.35 * ((u.Msun * u.pc ** -2) * (u.K * u.km * u.s ** -1) ** -1)  # Bolatto+13
alpha_co_21_vertico = alpha_co_10 / R21_vertico

def K_to_mpa(Kval,cube_k,co21_obs,numpixinbeam,chan_w,distance,z,alpha_co_21_vertico,pc_per_pix):
    jypb_val = (Kval*u.K).to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(co21_obs)) 
    
    temp_Sco = (jypb_val/numpixinbeam)*chan_w
    temp_Lco = (3.25e7 * temp_Sco * (co21_obs) ** (-2) * distance ** 2 * (1 + z) ** (-3)).value * (u.K * u.km / u.s * u.pc ** 2)  # K km/s pc^2
    temp_Mmol = alpha_co_21_vertico * temp_Lco
    mpa_val = temp_Mmol.value*(pc_per_pix**(-2))
    
    mpa_out = [mpa_val,jypb_val.value]
    
    return mpa_out

def mpa_to_K(mpa_val,cube_k,co21_obs,numpixinbeam,chan_w,distance,z,alpha_co_21_vertico,pc_per_pix):
    
    temp_Mmol = mpa_val*(pc_per_pix**2)
    temp_Lco = temp_Mmol/alpha_co_21_vertico
    temp_Sco = temp_Lco / ((3.25e7 * (co21_obs) ** (-2) * distance ** 2 * (1 + z) ** (-3)).value * (u.K * u.km / u.s * u.pc ** 2))  # K km/s pc^2
    jypb_val = (temp_Sco*numpixinbeam/chan_w).value
    
    K_val = (jypb_val * u.K).to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(co21_obs))
    
    return K_val

def jypb_to_mpa_VIVA(jypb_val,numpixinbeam,chan_w,distance,pc_per_pix):
#    jypb_val = (Kval*u.K).to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(HI_obs)) 
    
    pixel = jypb_val/(numpixinbeam) #in units of Jy/beam * pix/beam = Jy/pix
    temp_SHI = pixel*chan_w # Now in units of Jy/pix * km/s
    temp_MHI = 2.356e5 * temp_SHI * distance**2
    mpa_val = temp_MHI.value*(pc_per_pix**(-2))
    
    return mpa_val

#Go through each frame of the cube, and compute the mass of HI at each pixel
def pixel_masser_VERTICO(cubepath,ssdir,sigmarootval,speclimits=[]):
    distance = virgo_distance
    #This function computes the mass of each pixel in a frame of a data cube.
    cube_output = hdulister(cubepath,True)
    head = cube_output['head']
    cdelt1 = cube_output['cdelt1']
    co21_rest = (head['RESTFRQ']*u.Hz).to('GHz')
    spec_axis_raw = np.array(cubevels(head))/1000.0 * u.km / u.s    
    cube_k = SpectralCube.read(cubepath).with_spectral_unit(u.km / u.s, rest_value=co21_rest, velocity_convention="optical")

    mcube_k = cube_k#.with_mask(mask_3d_t)

    if speclimits != []:
        msubcube_k = mcube_k[speclimits[0]:speclimits[1]]
        spec_axis = spec_axis_raw[speclimits[0]:speclimits[1]]
    if speclimits == []:
        msubcube_k = mcube_k
        spec_axis = spec_axis_raw

    vlo = min(spec_axis)
    vhi = max(spec_axis)
        
    z = np.mean(spec_axis) / const.c.to(u.km / u.s)
    dv = abs(vhi - vlo)  # line width [km/s]
    chan_w = np.median(abs(spec_axis[0:-1] - spec_axis[1:]))  # median channel width ~10 km/s

#    co21_obs = co21_rest*(1.0 + (np.median(spec_axis).value/ (const.c.value/1000.0) )) #in units of GHz
    co21_obs = np.median(msubcube_k.with_spectral_unit(u.GHz, rest_value=co21_rest).spectral_axis)  # GHz

    msubcube_jypb = msubcube_k.to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(co21_obs))    
    numpixinbeam = msubcube_jypb.pixels_per_beam
    
    arcsperpix = cdelt1*3600.0
    theta = np.pi/648000 #one arcsecond converted to radians (in rad/arcseconds)
    galdistance = distance.to(u.pc).value #convert from Mpc to pc
    galscale = galdistance*theta
    pc_per_pix = arcsperpix*galscale
    
    masscube = []
    
    seed = 1.0
    #if np.isnan(pixel) == False:
    pixel = seed/(numpixinbeam) #in units of Jy/beam * pix/beam = Jy/pix
    Sco = pixel*chan_w #Now in units of Jy/pix * km/s
    Lco = (3.25e7 * Sco * (co21_obs) ** (-2) * distance ** 2 * (1 + z) ** (-3)).value * (u.K * u.km / u.s * u.pc ** 2)  # K km/s pc^2
    Mmol = alpha_co_21_vertico * Lco
    factor = Mmol.value*(pc_per_pix**(-2))
    
    masscube = np.array(msubcube_jypb)*factor  
    mass_sum = np.nansum(masscube)*pc_per_pix**2
    
    #Now compute the mass per area delta value (mpa_delta)
    
    mpa_delta = K_to_mpa(sigmarootval,cube_k,co21_obs,numpixinbeam,chan_w,distance,z,alpha_co_21_vertico,pc_per_pix)
    
    outputdict = {'masscube':masscube,
                  'mass_sum':mass_sum,
                  'line_width':dv.value,
                  'spec_axis':spec_axis,
                  'chan_w':chan_w.value,
                  'mpa_delta':mpa_delta,
                  'factor':factor}
    
    return outputdict

def pixel_masser_VIVA(cubepath,ssdir,sigmarootval,speclimits=[]):
    distance = virgo_distance
    cube_output = hdulister(cubepath,True)
    head = cube_output['head']
    cdelt1 = cube_output['cdelt1']
    HI_rest = (head['RESTFRQ']*u.Hz).to('GHz') #was just Hz before, hopefully that doesn't mess anything up.
    
    raw_spec_axis = np.array(cubevels(head))/1000.0 * u.km / u.s
    cube_k = SpectralCube.read(cubepath).with_spectral_unit(u.km / u.s, rest_value = HI_rest, velocity_convention='optical')
    
    mcube_k = cube_k
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if speclimits == []:
        msubcube_k = mcube_k
        spec_axis = raw_spec_axis
    if speclimits != []:
        msubcube_k = mcube_k[speclimits[0]:speclimits[1]]
        spec_axis = raw_spec_axis[speclimits[0]:speclimits[1]]
    
    vlo = min(spec_axis)
    vhi = max(spec_axis)
        
#    z = np.mean(spec_axis) / const.c.to(u.km / u.s)
    dv = abs(vhi - vlo)  # line width [km/s]
    
    chan_w = np.median(abs(spec_axis[0:-1] - spec_axis[1:]))  # median channel width ~10 km/s
    
#    HI_obs = HI_rest*(1.0 + (np.median(spec_axis).value/(const.c.value/1000.0)))
    HI_obs = np.median(msubcube_k.with_spectral_unit(u.GHz, rest_value=HI_rest).spectral_axis)  # GHz

#    raw_spec_sum = cube_jypb.sum(axis=(1,2))
#    raw_jypb_spec_sum = raw_spec_sum / cube_jypb.pixels_per_beam
    
    msubcube_jypb = msubcube_k.to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(HI_obs))
    numpixinbeam = msubcube_jypb.pixels_per_beam
    
    arcsperpix = cdelt1*3600.0
    theta = np.pi/648000 #one arcsecond converted to radians (in rad/arcseconds)
    galdistance = virgo_distance.to(u.pc).value #convert from Mpc to pc
    galscale = galdistance*theta
    pc_per_pix = arcsperpix*galscale
    
    masscube = []
    
    seed = 1.0
    #if np.isnan(pixel) == False:
    pixel = seed/(numpixinbeam) #in units of Jy/beam * pix/beam = Jy/pix
    SHI = pixel*chan_w #now in units of Jy/beam * km/s # Now in units of Jy/pix * km/s
    MHI = 2.356e5 * SHI * virgo_distance**2
    factor = MHI.value*(pc_per_pix**(-2))
    
    masscube = np.array(msubcube_jypb)*factor
    
    mass_sum = np.nansum(masscube)*pc_per_pix**2
    
    mpa_delta = jypb_to_mpa_VIVA(sigmarootval,numpixinbeam,chan_w,distance,pc_per_pix)
    
    outputdict = {'masscube':masscube,
                  'mass_sum':mass_sum,
                  'line_width':dv.value,
                  'spec_axis':spec_axis,
                  'chan_w':chan_w.value,
                  'mpa_delta':mpa_delta,
                  'factor':factor}
    
    return outputdict