#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:10:49 2023

@author: DhruvB
"""

#DMR_DRA_functions.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectral_cube import SpectralCube
import astropy.units as u

import sys
sys.path.insert(0,'functions/')
from galnamegen import galnamegen
from spectrum_code import spectrum_grabber_VERTICO
#from spectrum_code import spectrum_grabber_VERTICO_noisy
from spectrum_code import spectrum_grabber_VIVA
#from spectrum_code import spectrum_grabber_VIVA_noisy

dotsperpix = 300
spec_colours = ['k','b','r']

def mom_map_grid(galnum,cubepath,maskcube,simcubepath,rotcubepath,anomcubepath,moment,savedir,stencil,spatial_bounds = []):
    
    galaxy,galstring = galnamegen(galnum)
    
    cMaps = [plt.cm.coolwarm,plt.cm.coolwarm,plt.cm.PuOr_r,plt.cm.coolwarm,plt.cm.coolwarm,plt.cm.PuOr_r,plt.cm.coolwarm,plt.cm.coolwarm,plt.cm.PuOr_r]
    percentile = 90
    unitstrings = ['[K km s$^{-1}$]','[km s$^{-1}$]']

    datacube = SpectralCube.read(cubepath).with_mask(maskcube).with_spectral_unit(u.km/u.s)

    data_mom = datacube.moment(order=moment)
    
    simcube = SpectralCube.read(simcubepath).with_spectral_unit(u.km/u.s)
    
    sim_mom = simcube.moment(order=moment)
    
    res_mom = data_mom - sim_mom
    
    rotcube = SpectralCube.read(rotcubepath).with_spectral_unit(u.km/u.s)
    anomcube = SpectralCube.read(anomcubepath).with_spectral_unit(u.km/u.s)
    
    res_mask = np.empty(np.shape(res_mom))
    grey_mask = np.empty(np.shape(res_mom))
    for i in range(len(res_mom)):
        for j in range(len(res_mom[i])):
            if np.isnan(res_mom[i][j]) == False:
                res_mask[i][j] = 1.0
                grey_mask[i][j] = np.nan
            if np.isnan(res_mom[i][j]) == True:
                res_mask[i][j] = np.nan
                grey_mask[i][j] = 1.0
    
    rot_mom = rotcube.moment(order=moment)
    anom_mom = anomcube.moment(order=moment)
    
    anom_mom_masked = anom_mom*res_mask
    
    sim_mom_masked = sim_mom*res_mask
    
    data_rot_res = data_mom - rot_mom
    
    data_anom_res = data_mom - anom_mom
    
    arrays = [data_mom,sim_mom_masked,res_mom,
              data_mom,rot_mom,data_rot_res,
              data_mom,anom_mom_masked,data_anom_res]
    
    mapvmin = np.nanpercentile(data_mom.value,100-percentile)
    mapvmax = np.nanpercentile(data_mom.value,percentile)
    
    resvmin = np.nanpercentile(res_mom.value,100-percentile)
    resvmax = np.nanpercentile(res_mom.value,percentile)
    
    anomresvmin = np.nanpercentile(data_anom_res.value,100-percentile)
    anomresvmax = np.nanpercentile(data_anom_res.value,percentile)
    
    vminvals = [mapvmin,mapvmin,resvmin,mapvmin,mapvmin,resvmin,mapvmin,mapvmin,anomresvmin]
    vmaxvals = [mapvmax,mapvmax,resvmax,mapvmax,mapvmax,resvmax,mapvmax,mapvmax,anomresvmax]
    
    ticklabelsize = 6        
    
    titles = ['Data',galstring+' Moment '+str(moment)+' Maps'+' '+unitstrings[moment]+'\n'+'Model from KinMS','Residuals','Data','Rot-only (from data)','','Data','Anomalous (from data)','']
    
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.7,wspace=-0.4)
    
    for i in range(len(arrays)):
        ax = fig.add_subplot(3,3,i+1)#,projection=wcs[0])
        
        if (i+1)%3 != 0:
            im = ax.imshow(arrays[i].value,origin='lower',cmap=cMaps[i],vmin=vminvals[i],vmax=vmaxvals[i])
        
        if (i+1)%3 == 0:
            if moment == 1:
                norm = colors.TwoSlopeNorm(vmin = vminvals[i],vmax = vmaxvals[i], vcenter = 0.0)        
            if moment == 0:
                norm = colors.Normalize(vmin = vminvals[i],vmax = vmaxvals[i])
            im = ax.imshow(arrays[i].value,origin='lower',cmap=cMaps[i],norm=norm)
        
        ax.set_title(titles[i],size=8)
        
        if spatial_bounds != []:
            ax.set_xlim(spatial_bounds[0])
            ax.set_ylim(spatial_bounds[1])
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size="10%", pad=0.05)
        fig.add_axes(cax)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=ticklabelsize)
    if stencil == 'no':
        fig.savefig(savedir+galaxy+'_mom'+str(moment)+'_maps.png',bbox_inches='tight',dpi=dotsperpix)
    if stencil == 'yes':
        fig.savefig(savedir+galaxy+'_mom'+str(moment)+'_maps_stencil.png',bbox_inches='tight',dpi=dotsperpix)
    plt.close()
    return

def spectrum_plotter_VERTICO(galnum,cubetype,cubepath,flatcubepath,maskpath,rotcubepath,anomcubepath,ssdir,speclimits,mpa_modelval,mpa_resval,savestring_VERTICO,stencil):
    
    galaxy, galstring = galnamegen(galnum)

#    if cubetype == 'real':
    full_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',cubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype+'_allgas')
    rot_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',rotcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype+'_rotating')                
    anom_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',anomcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype+'_anomalous')
#    if cubetype == 'noisy':
#        full_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',cubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='noisy_allgas')
#        rot_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',rotcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='noisy_rotating')                
#        anom_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',anomcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='noisy_anomalous')
        
    raw_spec_axes = [full_spectrum['spec_axis_raw'],rot_spectrum['spec_axis_raw'],anom_spectrum['spec_axis_raw']]
    raw_spec_sums = [full_spectrum['raw_jypb_spectrum'],rot_spectrum['raw_jypb_spectrum'],anom_spectrum['raw_jypb_spectrum']]
    vlimss = [full_spectrum['vlims'],rot_spectrum['vlims'],anom_spectrum['vlims']]
    
    M_all = full_spectrum['Mmol'][0]/10.0**6
    M_err = full_spectrum['Mmol'][1]/10.0**6
    M_rot = rot_spectrum['Mmol'][0]/10.0**6
    M_anom = anom_spectrum['Mmol'][0]/10.0**6
    
    f_anom = M_anom/(M_all)
    
    logM_all = np.round(np.log10(M_all*10.0**6),2)
    logM_rot = np.round(np.log10(M_rot*10.0**6),2)
    logM_anom = np.round(np.log10(M_anom*10.0**6),2)
    
    labels = ['All: log(M) = '+str(logM_all),'Rotating: log(M) = '+str(logM_rot),'Anomalous: log(M) = '+str(logM_anom)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(raw_spec_axes)):
        ax.axvline(vlimss[i][0].value+i*2,color=spec_colours[i],linestyle='dotted')
        ax.axvline(vlimss[i][1].value+i*2,color=spec_colours[i],linestyle='dotted')
        ax.scatter(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i],label=labels[i])
    ax.scatter(raw_spec_axes[1],(raw_spec_sums[1]+raw_spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
    for i in range(len(raw_spec_axes)):
        ax.plot(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i])
    ax.plot(0,0,color='w',label='f_anom = '+str(np.round(f_anom,3)))

    ax.set_xlabel('Radio Velocity [km s$^{-1}$]')
    
    specdiff = np.abs(np.mean(np.diff(raw_spec_axes[0]))).value
    
    xends_mod = [raw_spec_axes[1][0].value,raw_spec_axes[1][-1].value]
    
    xlow_mod = min(xends_mod) - specdiff
    xhigh_mod = max(xends_mod) + specdiff
    
    ax.set_xlim([xlow_mod,xhigh_mod])
    ax.set_ylabel('Flux Density [Jy]')
    ax.set_title(galstring+' in CO(2-1), threshold = ('+str(mpa_modelval)+', '+str(mpa_resval)+') M$_{\odot}$ pc$^{-2}$')
    ax.axhline(0.0,color='gray',linestyle='dotted')
    plt.legend(fontsize=7)
    savename = savestring_VERTICO+galaxy+'_spec_test'
    savename += '_'+cubetype
    if stencil == 'yes':
        savename+= '_stencil'
    fig.savefig(savename+'_CO.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    outputdict = {'full_spectrum':full_spectrum,
                  'rot_spectrum':rot_spectrum,
                  'anom_spectrum':anom_spectrum,
                  'M_all':M_all, 'M_rot':M_rot, 'M_anom':M_anom, 'M_err':M_err,
                  'f_anom':f_anom}
    
    return outputdict

def spectrum_plotter_VERTICO_highlighter(galnum,cubetype,cubepath,flatcubepath,maskpath,rotcubepath,anomcubepath,ssdir,speclimits,mpa_modelval,mpa_resval,savestring_VERTICO,stencil,highlightvels):
    
    galaxy, galstring = galnamegen(galnum)

#    if cubetype == 'real':
    full_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',cubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype+'_allgas')
    rot_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',rotcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype+'_rotating')                
    anom_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',anomcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype+'_anomalous')
#    if cubetype == 'noisy':
#        full_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',cubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='noisy_allgas')
#        rot_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',rotcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='noisy_rotating')                
#        anom_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',anomcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='noisy_anomalous')
        
    raw_spec_axes = [full_spectrum['spec_axis_raw'],rot_spectrum['spec_axis_raw'],anom_spectrum['spec_axis_raw']]
    raw_spec_sums = [full_spectrum['raw_jypb_spectrum'],rot_spectrum['raw_jypb_spectrum'],anom_spectrum['raw_jypb_spectrum']]
    vlimss = [full_spectrum['vlims'],rot_spectrum['vlims'],anom_spectrum['vlims']]
    
    M_all = full_spectrum['Mmol'][0]/10.0**6
#    M_err = full_spectrum['Mmol'][1]/10.0**6
    M_rot = rot_spectrum['Mmol'][0]/10.0**6
    M_anom = anom_spectrum['Mmol'][0]/10.0**6
    
    f_anom = M_anom/(M_all)
    
    logM_all = np.round(np.log10(M_all*10.0**6),2)
    logM_rot = np.round(np.log10(M_rot*10.0**6),2)
    logM_anom = np.round(np.log10(M_anom*10.0**6),2)
    
    labels = [r'log(M$_{\mathrm{all}}$) = '+str(logM_all),r'log(M$_{\mathrm{rot}}$) = '+str(logM_rot),r'log(M$_{\mathrm{anom}}$) = '+str(logM_anom)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    specdiff = np.abs(np.mean(np.diff(raw_spec_axes[0]))).value
    for i in range(len(highlightvels)):
        ax.axvspan(highlightvels[i]-specdiff/2.0, highlightvels[i]+specdiff/2.0, color='yellow', alpha=0.5)
    
    for i in range(len(raw_spec_axes)):
#        ax.axvline(vlimss[i][0].value+i*2,color=spec_colours[i],linestyle='dotted')
#        ax.axvline(vlimss[i][1].value+i*2,color=spec_colours[i],linestyle='dotted')
        ax.axvline(vlimss[i][0].value,color='grey',linestyle='dotted')
        ax.axvline(vlimss[i][1].value,color='grey',linestyle='dotted')
        ax.scatter(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i],label=labels[i])
#    ax.scatter(raw_spec_axes[1],(raw_spec_sums[1]+raw_spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
    for i in range(len(raw_spec_axes)):
        ax.plot(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i])
    ax.plot(0,0,color='w',label=r'f$_{\mathrm{anom,data}}$ = '+str(np.round(f_anom,3)))

    ax.set_xlabel('Radio Velocity [km s$^{-1}$]')
    
    specdiff = np.abs(np.mean(np.diff(raw_spec_axes[0]))).value
    
    xends_mod = [raw_spec_axes[1][0].value,raw_spec_axes[1][-1].value]
    
    xlow_mod = min(xends_mod) - specdiff
    xhigh_mod = max(xends_mod) + specdiff
    
    ax.set_xlim([xlow_mod,xhigh_mod])
    ax.set_ylabel(r'Flux Density [Jy beam$^{-1}$]')
    ax.set_title(galstring+r' in CO(2-1), $\tau$ = '+str(mpa_modelval)+' M$_{\odot}$ pc$^{-2}$')
    ax.axhline(0.0,color='gray',linestyle='dotted')
    plt.legend(fontsize=7)
    savename = savestring_VERTICO+galaxy+'_spec_test'
    savename += '_'+cubetype
    if stencil == 'yes':
        savename+= '_stencil'
#    fig.savefig(savename+'_CO_highlighted.png',dpi=300,bbox_inches='tight')
    fig.savefig(savename+'_CO_highlighted.pdf',dpi=300,bbox_inches='tight')
    plt.close()
    
    return
    
#def spectrum_plotter_VERTICO_highlighter(galnum,noisy,cubepath,flatcubepath,maskpath,rotcubepath,anomcubepath,ssdir,speclimits,mpa_modelval,mpa_resval,savestring_VERTICO,stencil,highlightvels):
#    
#    galaxy, galstring = galnamegen(galnum)
#
#    if noisy == 'no':
#        full_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',cubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='allgas')
#        rot_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',rotcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='rotating')
#        anom_spectrum = spectrum_grabber_VERTICO(galaxy,'v1.2',anomcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='anomalous')
#    if noisy == 'yes':
#        full_spectrum = spectrum_grabber_VERTICO_noisy(galaxy,'v1.2',cubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='allgas')
#        rot_spectrum = spectrum_grabber_VERTICO_noisy(galaxy,'v1.2',rotcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='rotating')
#        anom_spectrum = spectrum_grabber_VERTICO_noisy(galaxy,'v1.2',anomcubepath,flatcubepath,maskpath,'pbcorr',ssdir,speclimits,cubetype='anomalous')
#        
#    raw_spec_axes = [full_spectrum['spec_axis_raw'],rot_spectrum['spec_axis_raw'],anom_spectrum['spec_axis_raw']]
#    raw_spec_sums = [full_spectrum['raw_jypb_spectrum'],rot_spectrum['raw_jypb_spectrum'],anom_spectrum['raw_jypb_spectrum']]
#    vlimss = [full_spectrum['vlims'],rot_spectrum['vlims'],anom_spectrum['vlims']]
#    
#    M_all = full_spectrum['Mmol'][0]/10.0**6
#    M_rot = rot_spectrum['Mmol'][0]/10.0**6
#    M_anom = anom_spectrum['Mmol'][0]/10.0**6
#    
#    f_anom = M_anom/(M_all)
#    
#    logM_all = np.round(np.log10(M_all*10.0**6),2)
#    logM_rot = np.round(np.log10(M_rot*10.0**6),2)
#    logM_anom = np.round(np.log10(M_anom*10.0**6),2)
#    
#    labels = ['All: log(M) = '+str(logM_all),'Rotating: log(M) = '+str(logM_rot),'Anomalous: log(M) = '+str(logM_anom)]
#    
#    specdiff = np.abs(np.mean(np.diff(raw_spec_axes[0]))).value
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    
#    for i in range(len(highlightvels)):
#        ax.axvspan(highlightvels[i]-specdiff, highlightvels[i]+specdiff, color='yellow', alpha=0.5)
#    
#    for i in range(len(raw_spec_axes)):
#        ax.axvline(vlimss[i][0].value+i*2,color=spec_colours[i],linestyle='dotted')
#        ax.axvline(vlimss[i][1].value+i*2,color=spec_colours[i],linestyle='dotted')
#        ax.scatter(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i],label=labels[i])
#    ax.scatter(raw_spec_axes[1],(raw_spec_sums[1]+raw_spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
#    for i in range(len(raw_spec_axes)):
#        ax.plot(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i])
#    ax.plot(0,0,color='w',label='f_anom = '+str(np.round(f_anom,3)))
#
#    ax.set_xlabel('Radio Velocity [km s$^{-1}$]')
#    
#    xends = [raw_spec_axes[0][0].value,raw_spec_axes[0][-1].value]
#    
#    xlow = min(xends) - specdiff
#    xhigh = max(xends) + specdiff
#    
#    ax.set_xlim([xlow,xhigh])
#    ax.set_ylabel('Flux Density [Jy]')
#    ax.set_title(galstring+r' in CO(2-1), I$_{\mathrm{thres}}$ = ('+str(mpa_modelval)+') M$_{\odot}$ pc$^{-2}$')
#    ax.axhline(0.0,color='gray',linestyle='dotted')
#    plt.legend(fontsize=7)
#    savename = savestring_VERTICO+galaxy+'_spec_test'
#    if noisy == 'yes':
#        savename += '_noisy'
#    if stencil == 'yes':
#        savename+= '_stencil'
#    fig.savefig(savename+'_CO_highlighted.png',dpi=300,bbox_inches='tight')
#    fig.savefig(savename+'_CO_highlighted.pdf',dpi=300,bbox_inches='tight')
#    plt.close()
#    return

def spectrum_plotter_VIVA(galnum,cubetype,cubepath,rotcubepath,anomcubepath,ssdir,speclimits,linefree_channels,mpa_modelval,mpa_resval,savestring_VIVA,stencil):
    
    galaxy, galstring = galnamegen(galnum)

    full_spectrum = spectrum_grabber_VIVA(galaxy,cubepath,ssdir,True,True,speclimits,linefree_channels,cubetype+'_allgas')
    rot_spectrum = spectrum_grabber_VIVA(galaxy,rotcubepath,ssdir,True,True,speclimits,linefree_channels,cubetype+'_rotating')
    anom_spectrum = spectrum_grabber_VIVA(galaxy,anomcubepath,ssdir,True,True,speclimits,linefree_channels,cubetype+'_anomalous')
    raw_spec_axes = [full_spectrum['spec_axis_raw'],rot_spectrum['spec_axis_raw'],anom_spectrum['spec_axis_raw']]
    spec_axes = [full_spectrum['spec_axis'],rot_spectrum['spec_axis'],anom_spectrum['spec_axis']]

    raw_spec_sums = [full_spectrum['raw_spec_sum'],rot_spectrum['raw_spec_sum'],anom_spectrum['raw_spec_sum']]
    spec_sums = [full_spectrum['spec_sum'],rot_spectrum['spec_sum'],anom_spectrum['spec_sum']]
    vlimss = [full_spectrum['vlims'],rot_spectrum['vlims'],anom_spectrum['vlims']]
    
    M_all = full_spectrum['MHI'][0]/10.0**6
    M_err = full_spectrum['MHI'][1]/10.0**6
    M_rot = rot_spectrum['MHI'][0]/10.0**6
    M_anom = anom_spectrum['MHI'][0]/10.0**6
    
    f_anom = M_anom/(M_all)
    
    logM_all = np.round(np.log10(M_all*10.0**6),3)
    logM_rot = np.round(np.log10(M_rot*10.0**6),3)
    logM_anom = np.round(np.log10(M_anom*10.0**6),3)
    
    labels = [r'log(M$_{\mathrm{all}}$) = '+str(logM_all),r'log(M$_{\mathrm{rot}}$) = '+str(logM_rot),r'log(M$_{\mathrm{anom}}$) = '+str(logM_anom)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(spec_axes)):
#        ax.axvline(vlimss[i][0].value+i*2,color=spec_colours[i],linestyle='dotted')
#        ax.axvline(vlimss[i][1].value+i*2,color=spec_colours[i],linestyle='dotted')
        ax.axvline(vlimss[i][0].value,color='grey',linestyle='dotted')
        ax.axvline(vlimss[i][1].value,color='grey',linestyle='dotted')
        ax.scatter(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i],label=labels[i])
    sumxarray = spec_axes[1]
#    if raw_spec_axes[0][0] > raw_spec_axes[0][-1]:
#        sumxarray = spec_axes[1][::-1]
    ax.scatter(sumxarray,(spec_sums[1]+spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
    for i in range(len(spec_axes)):
        ax.plot(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i])
    ax.plot(0,0,color='w',label='f_anom = '+str(np.round(f_anom,3)))

    ax.set_xlabel('Radio Velocity [km s$^{-1}$]')
    
    specdiff = np.abs(np.mean(np.diff(spec_axes[0]))).value
    
    xends = [spec_axes[0][0].value,spec_axes[0][-1].value]    
    xlow = min(xends) - specdiff
    xhigh = max(xends) + specdiff
    
    ax.set_xlim([xlow,xhigh])
    ax.set_ylabel('Flux Density [Jy]')
    ax.set_title(galstring+' in HI, threshold = ('+str(mpa_modelval)+', '+str(mpa_resval)+') M$_{\odot}$ pc$^{-2}$')
    ax.axhline(0.0,color='gray',linestyle='dotted')
    plt.legend(fontsize=7)
    savename = savestring_VIVA+galaxy+'_spec_test'
    savename += '_'+cubetype
    if stencil == 'yes':
        savename+= '_stencil'
    fig.savefig(savename+'_HI.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    outputdict = {'full_spectrum':full_spectrum,
                  'rot_spectrum':rot_spectrum,
                  'anom_spectrum':anom_spectrum,
                  'M_all':M_all, 'M_rot':M_rot, 'M_anom':M_anom, 'M_err':M_err,
                  'f_anom':f_anom}
    
    return outputdict

def spectrum_plotter_VIVA_highlighter(galnum,cubetype,cubepath,rotcubepath,anomcubepath,ssdir,speclimits,linefree_channels,mpa_modelval,mpa_resval,savestring_VIVA,stencil,highlightvels):
    
    galaxy, galstring = galnamegen(galnum)

    full_spectrum = spectrum_grabber_VIVA(galaxy,cubepath,ssdir,True,True,speclimits,linefree_channels,cubetype+'_allgas')
    rot_spectrum = spectrum_grabber_VIVA(galaxy,rotcubepath,ssdir,True,True,speclimits,linefree_channels,cubetype+'_rotating')
    anom_spectrum = spectrum_grabber_VIVA(galaxy,anomcubepath,ssdir,True,True,speclimits,linefree_channels,cubetype+'_anomalous')
    raw_spec_axes = [full_spectrum['spec_axis_raw'],rot_spectrum['spec_axis_raw'],anom_spectrum['spec_axis_raw']]
    spec_axes = [full_spectrum['spec_axis'],rot_spectrum['spec_axis'],anom_spectrum['spec_axis']]

    raw_spec_sums = [full_spectrum['raw_spec_sum'],rot_spectrum['raw_spec_sum'],anom_spectrum['raw_spec_sum']]
    spec_sums = [full_spectrum['spec_sum'],rot_spectrum['spec_sum'],anom_spectrum['spec_sum']]
    vlimss = [full_spectrum['vlims'],rot_spectrum['vlims'],anom_spectrum['vlims']]
    
    M_all = full_spectrum['MHI'][0]/10.0**6
    M_err = full_spectrum['MHI'][1]/10.0**6
    M_rot = rot_spectrum['MHI'][0]/10.0**6
    M_anom = anom_spectrum['MHI'][0]/10.0**6
    
    f_anom = M_anom/(M_all)
    
    logM_all = np.round(np.log10(M_all*10.0**6),2)
    logM_rot = np.round(np.log10(M_rot*10.0**6),2)
    logM_anom = np.round(np.log10(M_anom*10.0**6),2)
    
    labels = [r'log(M$_{\mathrm{tot}}$) = '+str(logM_all),r'log(M$_{\mathrm{rot}}$) = '+str(logM_rot),r'log(M$_{\mathrm{anom}}$) = '+str(logM_anom)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    specdiff = np.abs(np.mean(np.diff(spec_axes[0]))).value
    
    for i in range(len(highlightvels)):
        ax.axvspan(highlightvels[i]-specdiff/2.0, highlightvels[i]+specdiff/2.0, color='yellow', alpha=0.5)
    for i in range(len(spec_axes)):
#        ax.axvline(vlimss[i][0].value+i*2,color=spec_colours[i],linestyle='dotted')
#        ax.axvline(vlimss[i][1].value+i*2,color=spec_colours[i],linestyle='dotted')
        ax.axvline(vlimss[i][0].value,color='grey',linestyle='dotted')
        ax.axvline(vlimss[i][1].value,color='grey',linestyle='dotted')
        ax.scatter(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i],label=labels[i])
#    sumxarray = spec_axes[1]
#    if raw_spec_axes[0][0] > raw_spec_axes[0][-1]:
#        sumxarray = spec_axes[1][::-1]
#    ax.scatter(sumxarray,(spec_sums[1]+spec_sums[2]),color='g',alpha=0.5,marker='x',label = 'Rotating + Anomalous')
    for i in range(len(spec_axes)):
        ax.plot(raw_spec_axes[i],raw_spec_sums[i],color=spec_colours[i])
    ax.plot(0,0,color='w',label=r'f$_{\mathrm{anom}}$ = '+str(np.round(f_anom,3)))

    ax.set_xlabel('Radio Velocity [km s$^{-1}$]')
    
    xends = [spec_axes[0][0].value,spec_axes[0][-1].value]    
    xlow = min(xends) - specdiff
    xhigh = max(xends) + specdiff
    
    ax.set_xlim([xlow,xhigh])
    ax.set_ylabel('Flux Density [Jy]')
    ax.set_title(galstring+r' in HI, $\tau$ = '+str(mpa_modelval)+' M$_{\odot}$ pc$^{-2}$')
    ax.axhline(0.0,color='gray',linestyle='dotted')
    plt.legend(fontsize=7)
    savename = savestring_VIVA+galaxy+'_spec_test'
    savename += '_'+cubetype
    if stencil == 'yes':
        savename+= '_stencil'
    fig.savefig(savename+'_HI_highlighted.pdf',dpi=300,bbox_inches='tight')
    plt.close()
    
    outputdict = {'full_spectrum':full_spectrum,
                  'rot_spectrum':rot_spectrum,
                  'anom_spectrum':anom_spectrum,
                  'M_all':M_all, 'M_rot':M_rot, 'M_anom':M_anom, 'M_err':M_err,
                  'f_anom':f_anom}
    
    return outputdict
