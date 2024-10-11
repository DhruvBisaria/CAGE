#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:38:42 2023

@author: DhruvB
"""

#DMR_DRA_simcube.py
from pprint import pprint as pp
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
from astrodendro import Dendrogram
from spectral_cube import SpectralCube
import astropy.units as u
import scipy.ndimage as ndimage
from helper_functions import ror

import sys
sys.path.insert(0,'functions/')
from cube_vel_finder import cubevels
from cube_vel_finder import intersect
from cube_vel_finder import cubesorter
from dendrogram_functions import rotanom_dendro
from dendrogram_functions import numpixfinder
from dendrogram_functions import mask_manipulator
from dendrogram_functions import cubewriter
from hdulister import hdulister
from pixel_masser import pixel_masser_VIVA
from pixel_masser import jypb_to_mpa_VIVA
from simcube_noise_adder import noisy_cube_generator

virgo_distance = 16.5 * u.Mpc
spec_colours = ['k','b','r']  


def DMR_DRA_plotter(run,plotting,SB_form,basedir,cubepath,kinmscubepath,maskcubepath,specmin,specmax,threshold,run_label,frames_name):
    co = hdulister(cubepath,True)
    ch = co['head']
    input_cube = np.array(co['cube'])
    xlen = co['xlen']
    ylen = co['ylen']
    cdelt1 = co['cdelt1']
    cdelt2 = co['cdelt2']
    bmajor = co['bmajor']
    bminor = co['bminor']
    bpa = co['bpa']
    
    kco = hdulister(kinmscubepath,True)
    kh = kco['head']
    kinms_cube = np.array(kco['cube'])
    
    input_cubevels = cubevels(ch)
    
    simvels = cubevels(kh)
    commonvels = intersect(input_cubevels,simvels)
    
#    if specmin != [] and specmax != []:
#        input_cube = input_cube[specmin:specmax]
    
    input_cube = cubesorter(input_cube,input_cubevels,commonvels)
    kinms_cube = cubesorter(kinms_cube,simvels,commonvels)
    
    mo = hdulister(maskcubepath,True)
    mask_cube = mo['cube']
    mask_sum = np.any(mask_cube, axis=0).astype(int) #binary_map = mask_sum.astype(int)
    maskvels = cubevels(mo['head'])
    
    mask_cube = cubesorter(mask_cube,maskvels,commonvels)
    
    rescube = input_cube - kinms_cube
    
    pix_in_beam = numpixfinder(bmajor,bminor,cdelt1,cdelt2)
    
    sigmarootval = rms_noise
    
    a = pixel_masser_VIVA(cubepath,'',sigmarootval)        
    chan_w = a['chan_w']
    mpa_delta = a['mpa_delta']
    factor = float(a['factor'])
    
    mpa_input_cube = input_cube * factor
    mpa_modelcube = kinms_cube * factor
    mpa_rescube = rescube * factor
    mpa_modelval = threshold * factor
    mpa_resval = threshold * factor
    
    dd = rotanom_dendro(input_cube, mpa_input_cube, mpa_modelcube, mpa_rescube, xlen, ylen, mpa_modelval, mpa_resval, mpa_delta, pix_in_beam)
    
    trunkflags = dd['trunkflags']
    allds = dd['allds']
    allresds = dd['allresds']
    alltrunks = dd['alltrunks']
    allresleaves = dd['allresleaves']
    allrestrunks = dd['allrestrunks']
    allleafindices = dd['allleafindices']
    allcentroids = dd['allcentroids']
    allclocs = dd['allclocs']
    rotatingcube = dd['rotatingcube']
    anomalouscube = dd['anomalouscube']
    
    rotcubepath = run_label+'_'+run+'_rot_cube_test.fits'
    anomcubepath = run_label+'_'+run+'_anom_cube_test.fits'
    
    cubewriter(rotatingcube,kh,'rotating_',rotcubepath)
    cubewriter(anomalouscube,kh,'anomalous_',anomcubepath)
    
    mpa_to_K_ratio = mpa_delta/sigmarootval
    
    sigmastartval = sigmarootval*(mpa_modelval/mpa_delta)
    
    vminval = 0.0
    mpa_vminval = vminval*mpa_to_K_ratio
    nsteps = 3
    
    vmaxval = sigmastartval + sigmarootval*nsteps
    
    arcsperpix = cdelt1*3600.0
    theta = np.pi/648000 #one arcsecond converted to radians (in rad/arcseconds)
    galdistance = virgo_distance.to(u.pc).value #convert from Mpc to pc
    galscale = galdistance*theta
    pc_per_pix = arcsperpix*galscale
    
    mpa_vmaxval = jypb_to_mpa_VIVA(vmaxval,pix_in_beam,chan_w,virgo_distance,pc_per_pix)
    
    bounds = np.array(np.arange(vminval,vmaxval,sigmarootval))
    boundlabels = np.round(bounds*1000.0,1)
    
    mpa_bounds = bounds*mpa_to_K_ratio
    mpa_boundlabels = np.round(mpa_bounds*1000.0,1)
    
    colorlist = []
    mpa_colorlist = []
    
    cMap = plt.cm.plasma
    mpa_cMap = plt.cm.gnuplot
    
    cmapTest = mpl.cm.get_cmap(cMap)
    mpa_cmapTest = mpl.cm.get_cmap(mpa_cMap)
    
    for i in range(len(bounds)):
    #    sigmaval = sigmarootval*i
        normz = bounds[i]/((vmaxval+sigmarootval)-vminval)
        colorlist.append(cmapTest(normz))
        
        mpa_normz = mpa_bounds[i]/((mpa_vmaxval+mpa_delta)-mpa_vminval)
        mpa_colorlist.append(mpa_cmapTest(mpa_normz))
    
    basenorm = sigmarootval/(vmaxval-vminval) #the proportional spacing between bounds/colors corresponding to vmaxval, vminval, and nsteps
    mpa_basenorm = mpa_delta/(mpa_vmaxval-mpa_vminval) #the proportional spacing between bounds/colors corresponding to vmaxval, vminval, and nsteps
    
    normz += basenorm
    mpa_normz += mpa_basenorm
    
    overcolor = cmapTest(normz)
    mpa_overcolor = mpa_cmapTest(mpa_normz)
    
    undercolor = (0,0,0,1)
    mpa_undercolor = (0,0,0,1)
    
    cmap = mpl.colors.ListedColormap(colorlist)
    mpa_cmap = mpl.colors.ListedColormap(mpa_colorlist)
    
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    mpa_norm = mpl.colors.BoundaryNorm(mpa_bounds,mpa_cmap.N)
    
    DMR_paneltitles = ['Data','Model','Residuals']
    DRA_paneltitles = ['Data','Rotating Gas','Anomalous Gas']
    paneltitles = np.concatenate([DMR_paneltitles,DRA_paneltitles])
    cblabel = 'Intensity [mK]'
    mpa_cblabel = r'Mass/Area [$M_{\odot} / pc^2 $]'
    
    centroidsize = 1.75
    centroidlinewidth = 0.4
    trunklinewidths = [0.75,0.6,0.3]
    trunkcolors = ['w','k','orange'] #background, main, highlight
    restrunkcolors = ['orange','darkgreen','r']
    
    BPA = np.deg2rad(bpa+90.0)
    pixperarc = 1.0/arcsperpix #pixels per arcseconds
    bminorpix = bminor*pixperarc
    bmajorpix = bmajor*pixperarc
    maxdist = 1.1*np.max([bminorpix,bmajorpix])
    ellipse_xc = maxdist #+ spatial_bounds[0][0]
    ellipse_yc = maxdist #+ spatial_bounds[1][0]
    
    #mm_returns = mask_manipulator('simulated',mask_spec_cube,realcubeheader,np.shape(realcube),True,True,speclimits) #function which dilates and stacks the masks
    #mask_3d_t = mm_returns[0]
    #mask_2d_unitless = mm_returns[1]    
    
    mo = hdulister(maskcubepath,True)
    mask_cube = mo['cube']
    mask_sum = np.any(mask_cube, axis=0).astype(int) #binary_map = mask_sum.astype(int)
    
#    mask_2d = use_mask_cube.moment(order=0) > 0
    
    # Number of pixels to dilate by
#    npix_dilation = np.ceil(abs(cube_k.header["BMAJ"] / cube_k.header["CDELT1"]) * nbeam_dilation).astype(int); nbeam_dilation = 1
    npix_dilation = np.ceil(abs(ch['BMAJ'] / ch['CDELT1']) * 1).astype(int)
    
    # the starting point is the 2d mask
    dilated_mask_2d = mask_sum
    
    # binary_dilation of mask by n pix
    for j in range(npix_dilation):
        struct = ndimage.generate_binary_structure(2, j)
        dilated_mask_2d = ndimage.binary_dilation(dilated_mask_2d, structure=struct)
    
    use_mask_2d = dilated_mask_2d.astype(int)
    
#    mask_dend = Dendrogram.compute(mask_sum, min_value=0.0, min_delta=0.0, min_npix=pix_in_beam, verbose=True)
    mask_dend = Dendrogram.compute(use_mask_2d, min_value=0.0, min_delta=0.0, min_npix=pix_in_beam, verbose=True)
#    mask_trunk = mask_dend.trunk
    
    ellipse_factor = 1.0/np.sqrt(np.log(2.0))
    
    if plotting == True:
        with PdfPages(frames_name) as pdf:
            vert = 6
            horiz = 8
            for k in range(len(commonvels)):
                if k < 200:
                    if np.mean(mpa_modelcube[k]) == 0.0:
                        continue
                    vel = commonvels[k]
                    labelvel = str(np.round(vel/1000.0,2))
                    fig = plt.figure(figsize=(horiz,vert))
        
        #            arrays = [sortedflatcube[k],sortedsimcube[k],residualcube[k],sortedpbcorrcube[k],rotatingcube[k],anomalouscube[k]]
        #            arrays = [sorted_mpa_flat[k],mpa_modelcube[k],mpa_rescube[k],sortedpbcorrcube[k],rotatingcube[k],anomalouscube[k]]
        #            arrays = [sorted_mpa_flat[k],mpa_modelcube[k],mpa_rescube[k],sortedpbcorrcube[k],rotatingcube[k],anomalouscube[k]]
                    arrays = [mpa_input_cube[k],mpa_modelcube[k],mpa_rescube[k],input_cube[k],rotatingcube[k],anomalouscube[k]]
                    
                    for j in range(len(arrays)):
        
                        ax = fig.add_subplot(2,3,j+1)
                        
                        if j < 3:
                            mpa_im = ax.imshow(arrays[j],origin='lower',cmap=mpa_cmap,norm=mpa_norm)
                        if j >= 3:
                            im = ax.imshow(arrays[j],origin='lower',cmap=cmap,norm=norm)
                        outcount = 0
                        incount = 0 #for legend purposes
                    
                        p_md = mask_dend.plotter() #p for mask dendrogram
                        p_md.plot_contour(ax,structure=0,colors='w',linewidths=0.5) #plot the mask contour
        
                        if j != 1:
                            ax.set_title(paneltitles[j],ha='center',fontsize=16)
                        if j == 1:
                            ax.set_title(paneltitles[j],ha='center',fontsize=16)
                        if trunkflags[k] == False:
                            p = allds[k].plotter()
                            for h in alltrunks[k]:
                                p.plot_contour(ax,structure=h,colors=trunkcolors[0],linewidths=trunklinewidths[0],alpha=0.25)
                                p.plot_contour(ax,structure=h,colors=trunkcolors[1],linewidths=trunklinewidths[1])
                                p.plot_contour(ax,structure=h,colors=trunkcolors[2],linewidths=trunklinewidths[2],linestyles='dotted')
        
                        if len(allresds[k]) > 0:
                            pres = allresds[k].plotter()
                            for h in allrestrunks[k]:
                                p.plot_contour(ax,structure=h,colors=restrunkcolors[0],linewidths=trunklinewidths[0],alpha=0.25)
                                p.plot_contour(ax,structure=h,colors=restrunkcolors[1],linewidths=trunklinewidths[1])
                                p.plot_contour(ax,structure=h,colors=restrunkcolors[2],linewidths=trunklinewidths[2],linestyles='dotted')
                        
                            if len(allresleaves[k]) > 0:
                                for g in range(len(allleafindices[k])):
                                    pres.plot_contour(ax,structure=allleafindices[k][g],colors='darkgreen',linewidths=0.5,zorder=2)
                                    pres.plot_contour(ax,structure=allleafindices[k][g],colors='w',linewidths=0.25,zorder=2,linestyles='dotted')
                                for g in range(len(allcentroids[k])):
                                    centroid = allcentroids[k][g]
                                    if allclocs[k][g] == True:
                                        if incount == 0:
                                            ax.scatter(centroid[0],centroid[1],s=centroidsize,linewidths=centroidlinewidth,color='w',marker='o',edgecolor='k',zorder=3,label='IN')
                                        if incount > 0:
                                            ax.scatter(centroid[0],centroid[1],s=centroidsize,linewidths=centroidlinewidth,color='w',marker='o',edgecolor='k',zorder=3)
                                        incount+=1
                                    if allclocs[k][g] == False:
                                        if outcount == 0:
                                            ax.scatter(centroid[0],centroid[1],s=centroidsize,linewidths=centroidlinewidth,color='r',marker='o',edgecolor='k',zorder=3,label='OUT')
                                        if outcount > 0:
                                            ax.scatter(centroid[0],centroid[1],s=centroidsize,linewidths=centroidlinewidth,color='r',marker='o',edgecolor='k',zorder=3)
                                        outcount+=1
                            if j == 2:
                                if incount > 0 or outcount > 0:
                                    ax.legend(loc=4, fontsize=6)
        
                        ax.xaxis.set_major_locator(plt.NullLocator())
                        ax.yaxis.set_major_locator(plt.NullLocator())
                        ax.add_patch(Ellipse((ellipse_xc,ellipse_yc),ellipse_factor*bmajorpix,ellipse_factor*bminorpix,BPA,edgecolor='k',facecolor='none',lw=1.0))
                    cax = fig.add_axes([0.91, 0.175, 0.02, 0.65])
                    cb = fig.colorbar(im, cax = cax, orientation='vertical',extend='both')
                    cb.cmap.set_over(overcolor)
                    cb.cmap.set_under(undercolor)
                    cb.set_ticks(bounds)
                    cb.set_ticklabels(boundlabels)
                    cb.ax.tick_params(labelsize=16)
                    cb.update_ticks()
                    cb.set_label(cblabel, rotation=90,fontsize = 18)
                    
                    mpa_cax = fig.add_axes([-0.1, 0.175, 0.02, 0.65])
                    mpa_cb = fig.colorbar(mpa_im, cax = mpa_cax, orientation='vertical',extend='both')
                    mpa_cb.cmap.set_over(mpa_overcolor)
                    mpa_cb.cmap.set_under(mpa_undercolor)
                    mpa_cb.set_ticks(mpa_bounds)
                    mpa_cb.set_ticklabels(mpa_boundlabels)
                    mpa_cb.ax.tick_params(labelsize=16)
                    mpa_cb.update_ticks()
                    mpa_cb.set_label(mpa_cblabel, rotation=90,fontsize = 18)
                        
                    fig.suptitle(r' @ $\mathcal{V}_{\mathrm{los}} = $'+labelvel+' km s$^{-1}$, Mass/area = '+str(np.round(mpa_modelval,2)),fontsize=25)
                    pdf.savefig(dpi=300,bbox_inches='tight')  # saves the current figure into a pdf page
                    print(str(k)+'/'+str(len(commonvels)))
                    plt.close()
    return rotcubepath,anomcubepath

def spectrum_grabber(cubepath,maskcubepath,mask_for_plotting,beam_dilation,speclimits,cubetype='allgas',runtype='regular'):
    
    cube_output = hdulister(cubepath,True)
    head = cube_output['head']
    HI_rest = head['RESTFRQ'] * u.Hz
    spec_axis_raw = np.array(cubevels(head))/1000.0 * u.km / u.s
    cube_k = SpectralCube.read(cubepath).with_spectral_unit(u.km / u.s, rest_value = HI_rest, velocity_convention='optical')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask_cube = SpectralCube.read(maskcubepath, unitless=True).with_spectral_unit(u.km / u.s, rest_value=HI_rest, velocity_convention="optical")
    
#    mask_cube = SpectralCube.read(maskpath, unitless=True).with_spectral_unit(u.km / u.s, rest_value=co21_rest, velocity_convention="radio")
    
    # a boolean 2d mask using the moment map of the mask cube the cube
#    mm_returns = mask_manipulator(mask_cube,cubeheader,cubeshape,forplotting,dilation,speclimits,nbeam_dilation=1): #function which dilates and stacks the masks
    mm_returns = mask_manipulator(mask_cube,cube_k.header,np.shape(cube_k),mask_for_plotting,beam_dilation,[0,len(mask_cube)]) #function which dilates and stacks the masks
    mask_3d_t = mm_returns[0]
    
    # and mask the cube
    mcube_k = cube_k.with_mask(mask_3d_t)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if speclimits != []:
        specmin,specmax = speclimits
    
    if speclimits == []:
        specmin,specmax = 0,len(cube_output['cube'])
    
    msubcube_k = mcube_k[specmin:specmax]
    spec_axis = spec_axis_raw[specmin:specmax]    
    
    vlo = min(spec_axis)
    vhi = max(spec_axis)

    vlims = [vlo,vhi]
    
    HI_obs = np.median(msubcube_k.with_spectral_unit(u.GHz, rest_value=HI_rest).spectral_axis)  # GHz
#    HI_obs = HI_rest*(1.0 + (np.median(spec_axis).value/(const.c.value/1000.0)))
    
    mcube_jypb = mcube_k.to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(HI_obs))
    
    if cubetype == 'allgas' and runtype == 'regular' and speclimits != []:
        pre_spec_frames = mcube_jypb[0:specmin]
        post_spec_frames = mcube_jypb[specmax:]
        if len(pre_spec_frames) > 0 and len(post_spec_frames) > 0:
            all_noise_frames = np.concatenate([pre_spec_frames,post_spec_frames])
        if len(pre_spec_frames) > 0 and len(post_spec_frames) == 0:
            all_noise_frames = pre_spec_frames
        if len(pre_spec_frames) == 0 and len(post_spec_frames) > 0:
            all_noise_frames = post_spec_frames
#            
    if cubetype != 'allgas' and speclimits == []:
        all_noise_frames = []
        eSHI = 0.0
        eMHI = 0.0

    raw_spec_sum = mcube_jypb.sum(axis=(1,2))
#    raw_jypb_spec_sum = raw_spec_sum / cube_jypb.pixels_per_beam
    
    msubcube_jypb = msubcube_k.to(u.Jy / u.beam, equivalencies=cube_k.beam.jtok_equiv(HI_obs))    
    
    spec_sum = msubcube_jypb.sum(axis=(1, 2))
    
#    jypb_spec_sum = spec_sum / subcube_jypb.pixels_per_beam
    
    integrated_line_flux = abs(np.trapz(y=spec_sum, x=spec_axis))  # Sum Flux [Jy/beam km/s pix]
#    Sco = abs(np.trapz(y=jypb_spec_sum, x=spec_axis))  # Sum Flux [Jy/beam km/s pix] #THIS IS THE SAME THING AS THE FOLLOWING LINE:
    
    SHI = integrated_line_flux / msubcube_jypb.pixels_per_beam  # HI integrated flux [Jy km/s] = Sum Flux [Jy/beam km/s pix] / pix_per_beam [pix/beam]
    
    MHI = 2.356e5 * SHI * virgo_distance**2
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if cubetype == 'allgas' and runtype == 'regular' and speclimits != []:
        chan_w = np.median(abs(spec_axis_raw[0:-1] - spec_axis_raw[1:])) # median channel width ~10 km/s
        nchan_line = len(spec_axis)
        
        noise_frames_jypb = all_noise_frames.std(axis=0)
        
        noise_map_jy_kms = noise_frames_jypb * chan_w * np.sqrt(nchan_line)
        
        eSHI = np.sqrt(np.nansum(noise_map_jy_kms ** 2) / mcube_jypb.pixels_per_beam)
        
        eMHI = 2.356e5 * eSHI * virgo_distance**2
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if cubetype != 'allgas' or runtype == 'noisy_sim' or speclimits == []:
        eSHI = 0.0 * SHI.unit
        eMHI = 0.0 * MHI.unit
        chan_w = 0.0 * u.km / u.s
    
    output = {'spec_axis_raw':spec_axis_raw,
              'spec_axis':spec_axis,
              'vlims':vlims,
              'chan_w':chan_w,
              'HI_obs':HI_obs,
              'raw_spec_sum':raw_spec_sum,
              'spec_sum':spec_sum,
              'integrated_line_flux':integrated_line_flux,
              'SHI':[SHI,eSHI],
              'MHI':[MHI.value,eMHI.value]}
    
    return output

def spectrum_plotter(runtype,threshold_factor,cubepath,rotcubepath,anomcubepath,mpa_modelval,mpa_resval,results_dir,savestring,stencil,runlabel,speclimits,plotting='no'):
    if runtype == 'regular':
        full_spectrum = spectrum_grabber(cubepath,maskcubepath,True,True,speclimits,cubetype='allgas',runtype=runtype)
    if runtype == 'noisy_sim':
        full_spectrum = spectrum_grabber(cubepath,maskcubepath,True,True,speclimits=[],cubetype='allgas',runtype=runtype)
    rot_spectrum = spectrum_grabber(rotcubepath,maskcubepath,True,True,speclimits=[],cubetype='rotating',runtype=runtype)
    anom_spectrum = spectrum_grabber(anomcubepath,maskcubepath,True,True,speclimits=[],cubetype='anomalous',runtype=runtype)
    raw_spec_axes = [full_spectrum['spec_axis_raw'],rot_spectrum['spec_axis_raw'],anom_spectrum['spec_axis_raw']]
    spec_axes = [full_spectrum['spec_axis'],rot_spectrum['spec_axis'],anom_spectrum['spec_axis']]

    
    raw_spec_sums = [full_spectrum['raw_spec_sum'],rot_spectrum['raw_spec_sum'],anom_spectrum['raw_spec_sum']]
    spec_sums = [full_spectrum['spec_sum'],rot_spectrum['spec_sum'],anom_spectrum['spec_sum']]
    vlimss = [full_spectrum['vlims'],rot_spectrum['vlims'],anom_spectrum['vlims']]
    
    M_all = full_spectrum['MHI'][0]/10.0**6
    M_rot = rot_spectrum['MHI'][0]/10.0**6
    M_anom = anom_spectrum['MHI'][0]/10.0**6
    
    M_all_err = full_spectrum['MHI'][1]/10.0**6
    
    error_ratio = M_all_err/M_all
                                     
    percent_error = np.round((error_ratio)*100.0,1)
    
    f_anom = M_anom/(M_all)
    
    logM_all = np.round(np.log10(M_all*10.0**6),2)
    logM_rot = np.round(np.log10(M_rot*10.0**6),2)
    logM_anom = np.round(np.log10(M_anom*10.0**6),2)
    
    labels = ['All: log(M) = '+str(logM_all)+r' ($\pm$'+str(percent_error)+'%)','Rotating: log(M) = '+str(logM_rot),'Anomalous: log(M) = '+str(logM_anom)]
    
    if plotting == 'yes':
        np.savez(results_dir+'/f_anom_output/'+run_label+'_threshold_index_'+str(threshold_factor)+'_masses',masses_dict = {'M_all': M_all, 'M_rot': M_rot, 
             'M_anom': M_anom, 'M_all_err': M_all_err, 'error_ratio': error_ratio})
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(spec_axes)):
            ax.axvline(vlimss[i][0].value+i*2,color=spec_colours[i],linestyle='dotted')
            ax.axvline(vlimss[i][1].value+i*2,color=spec_colours[i],linestyle='dotted')
            ax.scatter(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i],label=labels[i])
        sumxarray = spec_axes[1]
        if raw_spec_axes[0][0] > raw_spec_axes[0][-1]:
            sumxarray = spec_axes[1][::-1]
        ax.scatter(sumxarray,(spec_sums[1]+spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
    #    ax.scatter(sumxarray,(spec_sums[1]+spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
        for i in range(len(spec_axes)):
            ax.plot(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i])
        ax.plot(0,0,color='w',label='f_anom = '+str(np.round(f_anom,3)))
    
        ax.set_xlabel('Radio Velocity [km s$^{-1}$]')
        
        specdiff = np.abs(np.mean(np.diff(spec_axes[0]))).value
        
        xends = [raw_spec_axes[0][0].value,raw_spec_axes[0][-1].value]    
        xlow = min(xends) - specdiff
        xhigh = max(xends) + specdiff
        
        ax.set_xlim([xlow,xhigh])
        ax.set_ylabel('Flux Density [Jy]')
    #    ax.set_title(galstring+' in HI, threshold = ('+str(mpa_modelval)+', '+str(mpa_resval)+') M$_{\odot}$ pc$^{-2}$')
        ax.set_title('Threshold = ('+str(mpa_modelval)+', '+str(mpa_resval)+') M$_{\odot}$ pc$^{-2}$')
        ax.axhline(0.0,color='gray',linestyle='dotted')
        plt.legend(fontsize=7)
    #    savename = savestring_VIVA+galaxy+'_spec_test'
        fig.savefig(savestring,dpi=300,bbox_inches='tight')
        plt.close()
    
    os.system('rm '+rotcubepath)
    os.system('rm '+anomcubepath)
    
    outputdict = {'full_spectrum':full_spectrum,
                  'rot_spectrum':rot_spectrum,
                  'anom_spectrum':anom_spectrum,
                  'M_all':M_all, 'M_rot':M_rot, 'M_anom':M_anom, 'M_err':M_all_err,
                  'error_ratio': error_ratio,
                  'f_anom':f_anom}
    
    return outputdict

if __name__ == "__main__":    
    import argparse
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('csv_path', type=str, help='the path to the csv input file.')
    
    args = my_parser.parse_args()
    
    csv_path = str(args.csv_path)
    
    rod = ror(csv_path)
        
    SB_form = str(rod['SB_form'])
    SB_params = rod['SB_params']
    run_label = str(rod['run_label'])
    cubepath = rod['cube_path']
    results_dir = str(rod['results_dir'])
    kinmscubepath = results_dir+'modelling_results/'+run_label+'_model_cube.fits'
    maskcubepath = str(rod['mask_cube_path'])
    rms_noise = float(rod['rms_noise'])
    thresholds = np.array(rod['thresholds'])
    plottings = rod['plottings']
    specmin = rod['specmin']
    specmax = rod['specmax']
    
    speclimits = [specmin,specmax]
    if np.isnan(specmin) == True and np.isnan(specmin) == True:
        speclimits = []
    
    kco = hdulister(kinmscubepath)['cube']

    x_array = np.linspace(1,len(thresholds),len(thresholds))
    a = pixel_masser_VIVA(kinmscubepath,'',rms_noise)        
    factor = float(a['factor'])
    
    mask_co = hdulister(maskcubepath,True)
    mask_cube = np.array(mask_co['cube'])
    mask_spec = mask_cube.sum(axis=(1,2))
    
    runs = ['regular','noisy_sim']
    both_f_anoms = []
    
    os.makedirs(results_dir+'modelling_results/f_anom_output/',exist_ok=True)
    os.makedirs(results_dir+'modelling_results/frames/',exist_ok=True)
    os.makedirs(results_dir+'modelling_results/spectra/',exist_ok=True)
    
    for run in runs:
        if run == 'regular':
            cubepath_used = cubepath
        if run == 'noisy_sim':
            tolerance=0.001
            if os.path.exists(kinmscubepath+'_noise_added.fits') == False:
                cubepath_used = noisy_cube_generator(cubepath,kinmscubepath.split('.fits')[0],tolerance)
            if os.path.exists(kinmscubepath+'_noise_added.fits') == True:
                cubepath_used = kinmscubepath+'_noise_added.fits'

        f_anoms = []
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            
            frames_name = results_dir+'frames/DMR_DRA_at_'+str(i+1).replace('.','_')+'_'+str(run)+'.pdf'
            
            rotcubepath, anomcubepath = DMR_DRA_plotter(run,plottings[i],SB_form,results_dir,cubepath_used,kinmscubepath,maskcubepath,specmin,specmax,threshold,run_label,frames_name)
            savestring = results_dir+'spectra/spectra_'+str(int(i+1))+'_'+run+'.png'
        
            threshold_for_plot = threshold*factor    
            outputdict = spectrum_plotter(run,i+1,cubepath_used,rotcubepath,anomcubepath,threshold*factor,threshold*factor,results_dir,savestring,'no',run_label,speclimits,'yes')
            f_anoms.append(outputdict['f_anom'])
            
            if i == 0 and run == 'regular':
                error_ratio = outputdict['error_ratio']
        both_f_anoms.append(f_anoms)

    delta_f_anoms = []
    for j in range(len(x_array)):
        delta_f_anoms.append(both_f_anoms[0][j]-both_f_anoms[1][j])

    all_output = {'both_f_anoms':both_f_anoms,
                  'delta_f_anoms':delta_f_anoms,
                  'x_array':x_array,
                  'error_ratio':error_ratio}
    np.savez(results_dir+'f_anom_output/f_anom_results',**all_output)
        
    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(211)
    ax1.scatter(x_array,both_f_anoms[0],color='b',alpha=0.7,label='real')
    ax1.scatter(x_array,both_f_anoms[1],color='g',alpha=0.7,label='noisy_sim')
    ax1.axhline(error_ratio, linestyle='dotted', color='red')
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.scatter(x_array,delta_f_anoms,color='k',label='delta')
    ax2.axhline(error_ratio, linestyle='dotted', color='red')
    ax2.sharex(ax1)
    ax2.legend()
    ax1.set_title(str(np.round(np.mean(delta_f_anoms),6))+' +/- '+str(np.round(error_ratio,6)))
    plt.savefig(results_dir+'f_anom_output/f_anom_plot.png',dpi=300)
    plt.close()