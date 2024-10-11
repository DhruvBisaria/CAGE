#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:55:28 2022

@author: DhruvB
"""

#hdulister.py

import numpy as np
from astropy.io import fits

def hdulister(path,extra=False):
    hdulist = fits.open(path)
    head = hdulist[0].header
    cube = hdulist[0].data
    if extra == True:
        xlen = head['NAXIS1']
        ylen = head['NAXIS2']
        zlen = head['NAXIS3']
        cdelt1 = np.abs(head['CDELT1'])
        cdelt2 = np.abs(head['CDELT2'])
        cdelt3 = np.abs(head['CDELT3'])
        bmajor = head['BMAJ']*3600.0 #convert from degrees to arcseconds
        bminor = head['BMIN']*3600.0 #convert from degrees to arcseconds
        bpa = head['BPA']
        xref = head['CRPIX1'] 
        yref = head['CRPIX2']
    hdulist.close()    
    if extra == True:
        outputdict = {'head':head,
                      'cube':cube,
                      'xlen':xlen,
                      'ylen':ylen,
                      'zlen':zlen,
                      'cdelt1':cdelt1,
                      'cdelt2':cdelt2,
                      'cdelt3':cdelt3,
                      'bmajor':bmajor,
                      'bminor':bminor,
                      'bpa':bpa,
                      'xref':xref,
                      'yref':yref}
    if extra == False:
        outputdict = {'head':head,
                      'cube':cube}
    
    return outputdict
    
def hdulister2d(path,extra=False):
    hdulist = fits.open(path)
    head = hdulist[0].header
    datamap = hdulist[0].data
    if extra == True:
        xlen = head['NAXIS1']
        ylen = head['NAXIS2']
        cdelt1 = np.abs(head['CDELT1'])
        cdelt2 = np.abs(head['CDELT2'])
        bmajor = head['BMAJ']*3600.0 #convert from degrees to arcseconds
        bminor = head['BMIN']*3600.0 #convert from degrees to arcseconds
        bpa = head['BPA']
        xref = head['CRPIX1'] 
        yref = head['CRPIX2']
    hdulist.close()    
    if extra == True:
        return [head,datamap,xlen,ylen,cdelt1,cdelt2,bmajor,bminor,bpa,xref,yref]
    else:
        return [head,datamap]
