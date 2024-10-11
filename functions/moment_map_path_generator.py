#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:20 2022

@author: DhruvB
"""

#moment_map_path_generator.py

def mmpg_VIVA(galaxy,forplotting=False):
    versionstring = 'v1.2.3'
#    if galaxy == 'NGC4298':
#        galaxy = 'NGC4302'

    folderpath = '../../../VIVA_DATA/2D_data/'+versionstring+'/'+galaxy+'/'
    if forplotting == True:
        folderpath = '../'+folderpath
    
    fitsname = galaxy+'_viva_hi_15as_np_round_reproj_mom1.fits'
    
    homemademom1s = ['NGC4216','NGC4222','NGC4294','NGC4298','NGC4299','NGC4302','NGC4532','NGC4536','NGC4580']
    
    if galaxy in homemademom1s:
        folderpath = '../../../VIVA_DATA/2D_data/'+versionstring+'/homemade_moment_maps/'
        fitsname = galaxy+'_homemade_mom1.fits'
        if forplotting == True:
            folderpath = '../'+folderpath
        
    mom1path = folderpath+fitsname

    return mom1path

def mmpg_VERTICO(survey,galaxy,resolution,forplotting=False):
    fitsname = galaxy+'_7m+tp_co21_pbcorr_'
    
    fitsname = fitsname+'{res}'.format(res = '15as_' if resolution == '15arcsec' else '')
    fitsname = fitsname+'{res}'.format(res = '9as_' if resolution == '9arcsec' else '')
    fitsname = fitsname+'{res}'.format(res = '' if resolution == 'native' else '')
    fitsname = fitsname+'round_mom1'
    
    mom1path = '../../../VERTICO_DATA/v1_3_2/'+resolution+'/'+galaxy+'/'+fitsname+'.fits'
    if forplotting == True:
        mom1path = '../'+mom1path
    
    return mom1path
