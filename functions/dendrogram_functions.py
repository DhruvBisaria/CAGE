#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:08:09 2022

@author: DhruvB
"""

#dendrogram_functions.py

import numpy as np
import astrodendro
from astropy.io import fits
from astrodendro import Dendrogram
import scipy.ndimage as ndimage
import sys

def centroidfinder(leaf,resframe): #a leaf is the most refined subsection of a dendrogram, basically a collection of pixels.
    totalflux = 0.0
    xweighted = 0.0
    yweighted = 0.0
    for i in range(len(leaf)):
        fluxval = resframe[leaf[i][0]][leaf[i][1]] #0 is y, 1 is x - very important
        if np.isnan(fluxval) == False:
            xweighted = leaf[i][1]*fluxval + xweighted
            yweighted = leaf[i][0]*fluxval + yweighted
            totalflux = fluxval + totalflux
    xc = xweighted/totalflux
    yc = yweighted/totalflux    
    return [xc,yc]

def centroidchecker(centroid,trunkpixels): #checks if the centroid is within the trunk, the model contour
    xcint = int(round(centroid[0]))
    ycint = int(round(centroid[1]))
    isin = False
    for i in range(len(trunkpixels)):
        if xcint == trunkpixels[i][1] and ycint == trunkpixels[i][0]:
            isin = True
            break
    return isin

def numpixfinder(major,minor,c1,c2):
#    beamarea = np.pi*major*minor/4.0 #units are square arcseconds. Need to divide by 2 because we want the axes (radii) not diameters.
    beamarea = major*minor*2*np.pi/(8*np.log(2))
    #print(major,minor,c1,c2)
    pixelarea = (c1*3600.0)*(c2*3600.0) #units are square arcseconds
#    print(beamarea/pixelarea)
    return int(np.ceil(beamarea/pixelarea))

def dir_creator(dirs,basedir):
    alldirs = []
    dirstring = basedir+dirs[0]
    for i in range(len(dirs)-1):
        alldirs.append(dirstring)
        dirstring+=dirs[i+1]
    #for i in range(len(alldirs)):
    #    if os.path.exists(alldirs[i]) == False:
    #        os.system('mkdir '+alldirs[i])
    return alldirs[-1]

def nac(arraylike): #nan array creator
    shape = np.shape(arraylike)
    blankcube = np.empty(shape)
    blankcube[:] = 0.0
    
    return blankcube

def onlyonce(list_cor):
    arr, uniq_cnt = np.unique(list_cor, axis=0, return_counts=True)
    uniq_arr = arr[uniq_cnt==1]
    dupes = arr[uniq_cnt > 1]
    return np.concatenate([uniq_arr,dupes],axis=0)
    
def trunkpixelfinder(dendro):
    alltpixs = []
    if len(dendro) == 0:
        return []
    if len(dendro) > 0:
        for i in range(len(dendro)):
            alltpixs.append(np.transpose(astrodendro.structure.Structure.indices(dendro[i]))) 
        return onlyonce(np.concatenate(alltpixs,axis=0))

def cube_nan_to_zero(cube,xlen,ylen):
    zerocube = np.zeros(np.shape(cube))
    for k in range(len(zerocube)):
        for i in range(ylen):
            for j in range(xlen):
                val = cube[k][i][j]
                if np.isnan(val) == False:
                    zerocube[k][i][j] = val
    return zerocube

def cubewriter(array,header,keyword,filename):
    array = np.array(array)
    header.set('TYPE',keyword,'Rotating or anomalous gas')
    fits.writeto(filename,array,header=header,overwrite=True)       
    return

def rotanom_dendro(sortedpbcorr,sorted_mpa_pbcorr,mpa_modelcube,mpa_rescube,xlen,ylen,mpa_modelval,mpa_resval,mpa_delta,pix_in_beam): #rotating and anomalous cube generator via dendrograms
    
    #Initialize the rotating and anomalous cube arrays
    rotatingcube = nac(sortedpbcorr)
    anomalouscube = nac(sortedpbcorr)
    
    #Initialize other useful/necessary arrays
    trunkflags = [] #append a true to this if a frame has no trunk in it (zero significant data)
    allds = [] #the dendrogram from each model frame gets appended to this
    allresds = [] #the dendrogram from each residual frame gets appended to this
    alltrunks = [] #all model trunks are appended to this
    alltrunkpixels = [] #all pixels from each model trunk is appended to this, per frame
    allrestrunkpixels = [] #same as alltrunkpixels, but for the residual trunks
    alldleaves = [] #all leaves from the model dendrogram per frame
    allresleaves = [] #all leaves from the residual dendrogram per frame
    allrestrunks = [] #all trunks from the residual dendrogram per frame
    allleafindices = [] #all indices from the residual leaves
    allcentroids = [] #all centroids from each leave
    allclocs = [] #all centroid locations vis a vis their location within the model trunk (true for IN, false for OUT)

    for k in range(len(mpa_modelcube)):
        model = mpa_modelcube[k]
        res = mpa_rescube[k]
        spbc = sorted_mpa_pbcorr[k]

        for i in range(ylen):
            for j in range(xlen):
                pixval = spbc[i][j]
                if np.isnan(pixval) == False:
                    anomalouscube[k][i][j] = sortedpbcorr[k][i][j]
        
        modeldelta = mpa_delta
        resdelta = mpa_delta
        d = Dendrogram.compute(model, min_value=mpa_modelval, min_delta=modeldelta, min_npix=pix_in_beam, verbose=True) #Compute the model dendrogram
        allds.append(d)
        if len(d) == 0:
            trunkflags.append(True)
            alltrunks.append([])
            alldleaves.append([])
            alltrunkpixels.append([])
            allrestrunkpixels.append([])
            allresleaves.append([])
            allrestrunks.append([])
            allleafindices.append([])
            allcentroids.append([])
            allclocs.append([])
            allresds.append([])
            continue

        trunkflags.append(False)
        resd = Dendrogram.compute(res, min_value=mpa_resval, min_delta=resdelta, min_npix=pix_in_beam, verbose=True) #Compute the residual dendrogram
        allresds.append(resd)

        resleaves = resd.leaves
        restrunk = resd.trunk
        trunk = d.trunk
        dleaves = d.leaves
        alldleaves.append(dleaves)

        trunkpixels = trunkpixelfinder(d)
        restrunkpixels = trunkpixelfinder(resd)

        for i in range(len(restrunkpixels)):
            x = restrunkpixels[i][0]
            y = restrunkpixels[i][1]
#            rotatingcube[k][x][y] = np.nan
            anomalouscube[k][x][y] = sortedpbcorr[k][x][y]

        centroids = []
        clocs = []
        leafindices = []

        for i in range(len(trunkpixels)):
            x = trunkpixels[i][0]
            y = trunkpixels[i][1]
            rotatingcube[k][x][y] = sortedpbcorr[k][x][y]
            anomalouscube[k][x][y] = 0.0 #used to be np.nan
        
        for i in range(len(resleaves)):
            leaf = np.transpose(astrodendro.structure.Structure.indices(resleaves[i]))

            leafindices.append(resleaves[i].idx)
            centroid = centroidfinder(leaf,res)
            cloc = centroidchecker(centroid,trunkpixels)

            for j in range(len(leaf)):
                x = leaf[j][0]
                y = leaf[j][1]
                if cloc == True:
                    rotatingcube[k][x][y] = sortedpbcorr[k][x][y]
                    anomalouscube[k][x][y] = 0.0 #used to be np.nan
                if cloc == False:
                    rotatingcube[k][x][y] = 0.0 #used to be np.nan
                    anomalouscube[k][x][y] = sortedpbcorr[k][x][y]

            centroids.append(centroid)
            clocs.append(cloc)

        alltrunks.append(trunk)
        allresleaves.append(resleaves)
        allrestrunks.append(restrunk)
        alltrunkpixels.append(trunkpixels)
        allrestrunkpixels.append(restrunkpixels)
        allleafindices.append(leafindices)
        allcentroids.append(centroids)
        allclocs.append(clocs)

    toreturn = {'trunkflags':trunkflags,
                'allds':allds,
                'allresds':allresds,
                'alltrunks':alltrunks,
                'alltrunkpixels':alltrunkpixels,
                'allrestrunkpixels':allrestrunkpixels,
                'alldleaves':alldleaves,
                'allresleaves':allresleaves,
                'allrestrunks':allrestrunks,
                'allleafindices':allleafindices,
                'allcentroids':allcentroids,
                'allclocs':allclocs,
                'rotatingcube':rotatingcube,
                'anomalouscube':anomalouscube}

    return toreturn

def rotanom_dendro_old(sortedpbcorr,sorted_mpa_pbcorr,mpa_modelcube,mpa_rescube,vels,xlen,ylen,mpa_rootval,mpa_delta,pix_in_beam): #rotating and anomalous cube generator via dendrograms
    trunkflags = []
    allds = []
    allresds = []
    alltrunks = []
    alltrunkpixels = []
    allrestrunkpixels = []
    alldleaves = []
    allresleaves = []
    allrestrunks = []
    allleafindices = []
    allcentroids = []
    allclocs = []

    rotatingcube = nac(sortedpbcorr)
    anomalouscube = nac(sortedpbcorr)

    for k in range(len(vels)):
        model = mpa_modelcube[k]
        res = mpa_rescube[k]
        spbc = sorted_mpa_pbcorr[k]
        
        for i in range(ylen):
            for j in range(xlen):
                pixval = spbc[i][j]
                if np.isnan(pixval) == False:
                    anomalouscube[k][i][j] = sortedpbcorr[k][i][j]

        modeldelta = mpa_delta
        resdelta = mpa_delta
        d = Dendrogram.compute(model, min_value=mpa_rootval, min_delta=modeldelta, min_npix=pix_in_beam, verbose=True)
        allds.append(d)
        if len(d) == 0:
            trunkflags.append(True)
            alltrunks.append([])
            alldleaves.append([])
            alltrunkpixels.append([])
            allrestrunkpixels.append([])
            allresleaves.append([])
            allrestrunks.append([])
            allleafindices.append([])
            allcentroids.append([])
            allclocs.append([])
            allresds.append([])
            continue

        trunkflags.append(False)
        resd = Dendrogram.compute(res, min_value=mpa_rootval, min_delta=resdelta, min_npix=pix_in_beam, verbose=True)
        allresds.append(resd)

        resleaves = resd.leaves
        restrunk = resd.trunk
        trunk = d.trunk
        dleaves = d.leaves
        alldleaves.append(dleaves)

        trunkpixels = trunkpixelfinder(d)
        restrunkpixels = trunkpixelfinder(resd)

        for i in range(len(restrunkpixels)): 
            x = restrunkpixels[i][0]
            y = restrunkpixels[i][1]
#            rotatingcube[k][x][y] = np.nan
            anomalouscube[k][x][y] = sortedpbcorr[k][x][y]

        centroids = []
        clocs = []
        leafindices = []

#        for i in range(len(trunkpixels)):
#            x = trunkpixels[i][0]
#            y = trunkpixels[i][1]
#            rotatingcube[k][x][y] = sortedpbcorr[k][x][y]
#            anomalouscube[k][x][y] = 0.0 #used to be np.nan

        for i in range(len(resleaves)):
            leaf = np.transpose(astrodendro.structure.Structure.indices(resleaves[i]))

            leafindices.append(resleaves[i].idx)
            centroid = centroidfinder(leaf,res)
            cloc = centroidchecker(centroid,trunkpixels)

            for j in range(len(leaf)):
                x = leaf[j][0]
                y = leaf[j][1]
                if cloc == True:
                    rotatingcube[k][x][y] = sortedpbcorr[k][x][y]
                    anomalouscube[k][x][y] = 0.0 #used to be np.nan
                if cloc == False:
                    rotatingcube[k][x][y] = 0.0 #used to be np.nan
                    anomalouscube[k][x][y] = sortedpbcorr[k][x][y]

            centroids.append(centroid)
            clocs.append(cloc)

        alltrunks.append(trunk)
        allresleaves.append(resleaves)
        allrestrunks.append(restrunk)
        alltrunkpixels.append(trunkpixels)
        allrestrunkpixels.append(restrunkpixels)
        allleafindices.append(leafindices)
        allcentroids.append(centroids)
        allclocs.append(clocs)

    toreturn = {'trunkflags':trunkflags,
                'allds':allds,
                'allresds':allresds,
                'alltrunks':alltrunks,
                'alltrunkpixels':alltrunkpixels,
                'allrestrunkpixels':allrestrunkpixels,
                'alldleaves':alldleaves,
                'allresleaves':allresleaves,
                'allrestrunks':allrestrunks,
                'allleafindices':allleafindices,
                'allcentroids':allcentroids,
                'allclocs':allclocs,
                'rotatingcube':rotatingcube,
                'anomalouscube':anomalouscube}

    return toreturn

def mask_manipulator(mask_cube,cubeheader,cubeshape,forplotting,dilation,speclimits,nbeam_dilation=1): #function which dilates and stacks the masks
    
    # a boolean 2d mask using the moment map of the mask cube the cube
    use_mask_cube = mask_cube[speclimits[0]:speclimits[1]]    
    mask_2d = use_mask_cube.moment(order=0) > 0
    
    # Number of pixels to dilate by
#    npix_dilation = np.ceil(abs(cube_k.header["BMAJ"] / cube_k.header["CDELT1"]) * nbeam_dilation).astype(int)
    npix_dilation = np.ceil(abs(cubeheader['BMAJ'] / cubeheader['CDELT1']) * nbeam_dilation).astype(int)
    
    # the starting point is the 2d mask
    dilated_mask_2d = mask_2d
    
    # binary_dilation of mask by n pix
    for j in range(npix_dilation):
        struct = ndimage.generate_binary_structure(2, j)
        dilated_mask_2d = ndimage.binary_dilation(dilated_mask_2d, structure=struct)
    
    if dilation == True:
        use_mask_2d = dilated_mask_2d    
    elif dilation == False:
        use_mask_2d = mask_2d
    
    #Now repeat the boolean mask for as many frames as the original cube to create a compressed mask cube.
    mask_3d = np.repeat(use_mask_2d[:, :, np.newaxis], cubeheader['NAXIS3'], axis=2)
    
    # transpose the mask to match the cube because the shape will be off.
    mask_3d_t = np.transpose(mask_3d, axes=[2, 0, 1])
    
    #Now convert the 2d mask back into values if the function calls for it.    
    mask_2d_unitless= np.zeros(np.shape(mask_2d))
    if forplotting == True:
        for i in range(len(use_mask_2d)):
            for j in range(len(use_mask_2d[i])):
                if use_mask_2d[i][j] == True:
                    mask_2d_unitless[i][j] = 1
                if use_mask_2d[i][j] == False:
                    mask_2d_unitless[i][j] = 0
    # assert that the mask is the same shape as the cube
    try:
        assert mask_3d_t.shape == cubeshape
    except AssertionError:
        print("AssertionError: mask not the same shape as cube.")
        sys.exit()
    
    return [mask_3d_t,mask_2d_unitless]