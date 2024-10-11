#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:18:57 2022

@author: DhruvB
"""

#cube_vel_finder.py

import numpy as np
import astropy

def cubevels(CubeHeader):
    Vels=[]
    for i in range(CubeHeader['NAXIS3']):
        VTemp=float(i-CubeHeader['CRPIX3']+1)*CubeHeader['CDELT3']+CubeHeader['CRVAL3']
        Vels.append(VTemp)
        
    vels = [round(j,2) for j in Vels]
    return vels

def cubefreqs(CubeHeader):
    freqs = []
    restfreq = CubeHeader['RESTFRQ']
    for i in range(CubeHeader['NAXIS3']):
        VTemp=float(i-CubeHeader['CRPIX3']+1)*CubeHeader['CDELT3']+CubeHeader['CRVAL3']
        freq = restfreq*(1.0 -  VTemp/astropy.constants.c.value)
        freqs.append(freq)
    return freqs

#Need a function that will determine if an array is ordered lowest-highest or highest-lowest.
def iao(arr): #is array ordered?
    # Check if the array is in ascending order
    if all(arr[i] <= arr[i+1] for i in range(len(arr)-1)):
        return True
    # Check if the array is in descending order
    elif all(arr[i] >= arr[i+1] for i in range(len(arr)-1)):
        return False
    # If neither ascending nor descending, it's unordered
    else:
        return None

def intersect(a, b):
    return np.sort(list(set(a) & set(b)))

def cubesorter(cube,vels,sortedvels):
    sortedcube = []
    for i in range(len(sortedvels)):
        for j in range(len(vels)):
            if sortedvels[i] == vels[j]:
                sortedcube.append(cube[j])
    return np.array(sortedcube)
