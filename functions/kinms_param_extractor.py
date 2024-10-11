#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:38:51 2022

@author: DhruvB
"""

#kinms_param_extractor.py
import numpy as np
from hdulister import hdulister
from deg_to_pix import deg2pix
import os

from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

def dictvalue(key,dictionary):
    if key in dictionary:
        return dictionary[key]
    else:
        return None

def kpe(npz_path,cube_path,SB_form):
    galdistance = 16.5 #Mpc, distance to Virgo cluster
    if os.path.exists(npz_path+'.npz') == True:
        f = np.load(npz_path+'.npz',allow_pickle=True)
        labels = f['labels']
        bestvals = f['bestvals']
        besterrs = f['besterrs']
        bincentroids = f['bc']
        f.close()
        
        bestvalsdict = dict(zip(labels,zip(bestvals,besterrs)))
        
        #Grab the best fitting values
        PA = dictvalue('PA',bestvalsdict)
        inc = dictvalue('inc',bestvalsdict)
        xc_deg = dictvalue('Xc',bestvalsdict)
        yc_deg = dictvalue('Yc',bestvalsdict)
        Vsys = dictvalue('Vsys',bestvalsdict)
        
        PeakFlux_exp = 0.0
        Rscale_exp = 0.0
        
        PeakFlux_gauss = 0.0
        Mean_gauss = 0.0
        sigma_gauss = 0.0
        
        peak_modsersic = 0.0
        mean_modsersic = 0.0
        sigma_modsersic = 0.0
        index_modsersic = 0.0
        
        if SB_form == 'exp':
            PeakFlux_exp = dictvalue('PeakFlux_exp',bestvalsdict)
            Rscale_exp = dictvalue('Rscale_exp',bestvalsdict)
        if SB_form == 'gaussian':
            PeakFlux_gauss = dictvalue('PeakFlux_gauss',bestvalsdict)
            Mean_gauss = dictvalue('Mean_gauss',bestvalsdict)
            sigma_gauss = dictvalue('sigma_gauss',bestvalsdict)
        if SB_form == 'sersic':
            peak_modsersic = dictvalue('peak_modsersic',bestvalsdict)
            mean_modsersic = dictvalue('mean_modsersic',bestvalsdict)
            sigma_modsersic = dictvalue('sigma_modsersic',bestvalsdict)
            index_modsersic = dictvalue('index_modsersic',bestvalsdict)
        if SB_form == 'exp_gaussian':
            PeakFlux_exp = dictvalue('PeakFlux_exp',bestvalsdict)
            Rscale_exp = dictvalue('Rscale_exp',bestvalsdict)
            PeakFlux_gauss = dictvalue('PeakFlux_gauss',bestvalsdict)
            Mean_gauss = dictvalue('Mean_gauss',bestvalsdict)
            sigma_gauss = dictvalue('sigma_gauss',bestvalsdict)
        if SB_form == 'exp_sersic':
            PeakFlux_exp = dictvalue('PeakFlux_exp',bestvalsdict)
            Rscale_exp = dictvalue('Rscale_exp',bestvalsdict)
            peak_modsersic = dictvalue('peak_modsersic',bestvalsdict)
            mean_modsersic = dictvalue('mean_modsersic',bestvalsdict)
            sigma_modsersic = dictvalue('sigma_modsersic',bestvalsdict)
            index_modsersic = dictvalue('index_modsersic',bestvalsdict)
        if SB_form == 'gaussian_sersic':
            PeakFlux_gauss = dictvalue('PeakFlux_gauss',bestvalsdict)
            Mean_gauss = dictvalue('Mean_gauss',bestvalsdict)
            sigma_gauss = dictvalue('sigma_gauss',bestvalsdict)
            peak_modsersic = dictvalue('peak_modsersic',bestvalsdict)
            mean_modsersic = dictvalue('mean_modsersic',bestvalsdict)
            sigma_modsersic = dictvalue('sigma_modsersic',bestvalsdict)
            index_modsersic = dictvalue('index_modsersic',bestvalsdict)
            
        #Grab the best fitting values for the rotation curve
        angradius = bincentroids #in arcseconds
        
        Vt = np.empty(len(angradius))
        eVt = np.empty(len(angradius))
        for key in bestvalsdict:
            keysplit = key.split('V')
            if len(keysplit) == 2:
                if keysplit[0] == '' and keysplit[1].isnumeric() == True:
                    index = int(keysplit[1])
                    Vt[index] = bestvalsdict[key][0]
                    eVt[index] = bestvalsdict[key][1]
        
        angradius = np.array(angradius)
        Vt = np.array(Vt)
        eVt = np.array(eVt)
        
        #Convert the angradius to a pixradius and a kpcradius
        cube_output = hdulister(cube_path,True)
        cubeheader = cube_output['head']
        cdelt1 = cube_output['cdelt1']
        
        xc_pix,yc_pix = deg2pix([xc_deg[0],yc_deg[0]],cubeheader)
        
        arcsperpix = cdelt1*3600.0
        pixradius = angradius/arcsperpix
        theta = np.pi/648000 #one arcsecond converted txo radians (in rad/arcseconds)
        galdistance = galdistance*1000.0 #convert from Mpc to kpc
        galscale = galdistance*theta
        kpcradius = angradius*galscale
        
        outputdict = {'PA': PA,
                      'inc': inc,
                      'xc_pix': xc_pix,
                      'yc_pix': yc_pix,
                      'xc_deg': xc_deg,
                      'yc_deg': yc_deg,
                      'Vsys': Vsys,
                      'arcsperpix': arcsperpix,
                      'angradius': angradius,
                      'pixradius': pixradius,
                      'kpcradius': kpcradius,
                      'galscale': galscale,
                      'Vt': Vt,
                      'eVt': eVt,
                      'PeakFlux_exp': PeakFlux_exp,
                      'Rscale_exp': Rscale_exp,###
                      'PeakFlux_gauss': PeakFlux_gauss,
                      'Mean_gauss': Mean_gauss,
                      'sigma_gauss': sigma_gauss,#####
                      'peak_modsersic': peak_modsersic,
                      'mean_modsersic': mean_modsersic,
                      'sigma_modsersic': sigma_modsersic,
                      'index_modsersic': index_modsersic}
    else:
        outputdict = None
    
    return outputdict