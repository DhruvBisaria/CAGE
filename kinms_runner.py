#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DhruvB
"""

#kinms_runner.py

import os
import numpy as np
import argparse
from kinms_fitter import kinms_fitter
from kinms_fitter.sb_profs import sb_profs
from helper_functions import ror

import sys
from pathlib import Path
sys.path.insert(0,'functions/')
from hdulister import hdulister

def kinms_runner(csv_path):
    rod = ror(csv_path)
    
    cube_path = str(rod['cube_path'])
    model_cube_name = str(rod['model_cube_name'])
    specmin = rod['specmin']
    specmax = rod['specmax']
    rms_noise = float(rod['rms_noise'])
    n_iterations = int(rod['n_iterations'])
    fixed_params = rod['fixed_params']
    PA_guess = float(rod['PA_guess'])
    i_guess = float(rod['i_guess'])
    x_center_guess = float(rod['x_center_guess'])
    y_center_guess = float(rod['y_center_guess'])
    vsys_guess = float(rod['vsys_guess'])
    num_rings = int(rod['num_rings']) #very important that num_rings be int
    SB_form = str(rod['SB_form'])
    #SB_params = np.array(rod['SB_params'])
#    SB_params = [float(item) for item in rod['SB_params']]
    SB_params = rod['SB_params']
    run_label = str(rod['run_label'])
    results_dir = str(rod['results_dir'])    
    modelling_results_dir = results_dir+'/modelling_results/'
    
    if not os.path.exists(modelling_results_dir):
        os.makedirs(modelling_results_dir)
#    
    #First, grab all the important parameters from the csv input file:
    if np.isnan(specmin) == False and np.isnan(specmax) == False:
        fit = kinms_fitter(cube_path,spectral_trim=[int(specmin),int(specmax)])
    else:
        fit = kinms_fitter(cube_path)
    fit.rms = rms_noise
    ##The number of iterations the MCMC fitting process will undergo:
    fit.niters = n_iterations
    #
    if fixed_params == 'none':
        fit.pa_guess = PA_guess
        if i_guess == 90.0:
            fit.inc_guess = 85.0
        if i_guess <= 89.0 and i_guess >= 1.0:
            fit.inc_guess = i_guess
        fit.xc_guess = x_center_guess #THE CENTER GUESS NEEDS TO BE IN RA AND DEC, NOT PIXELS!
        fit.yc_guess = y_center_guess
        if vsys_guess >= fit.vsys_range[0] and vsys_guess <= fit.vsys_range[1]:
            fit.vsys_guess = vsys_guess
    #
    ##The number of rings at which the tilted ring model will be fit at.
    fit.nrings = num_rings
    
    bmajor = hdulister(cube_path,True)['bmajor']
    r_ub = np.floor(bmajor*(num_rings-1))
    
    if SB_form == 'exp':
        I_guess = SB_params[0]
        r_scale_guess = SB_params[1]
        fit.sb_profile=[sb_profs.expdisk(guesses=[I_guess,r_scale_guess],minimums=[0.0,0.0],maximums=[10.0*I_guess,r_ub])]
    if SB_form == 'gaussian':
        I_guess = SB_params[0]
        r_o_guess = SB_params[1]
        r_scale_guess = SB_params[2]
        fit.sb_profile=[sb_profs.gaussian(guesses=[I_guess,r_o_guess,r_scale_guess],minimums=[0.0,0.0,0.0],maximums=[10.0*I_guess,r_ub,r_ub])]
    if SB_form == 'sersic':
        I_guess = SB_params[0]
        r_o_guess = SB_params[1]
        r_scale_guess = SB_params[2]
        n_init_sersic_guess = SB_params[3]
        fit.sb_profile=[sb_profs.mod_sersic(guesses=[I_guess,r_o_guess,r_scale_guess,n_init_sersic_guess],minimums=[0.0,0.0,0.0,0.5],maximums=[10.0*I_guess,r_ub,r_ub,4.0])]
    if SB_form == 'exp_gaussian':
        I_exp_guess = SB_params[0]
        r_scale_exp_guess = SB_params[1]        
        I_gaussian_guess = SB_params[2]
        r_o_gaussian_guess = SB_params[3]
        r_scale_gaussian_guess = SB_params[4]
        
        I_max = 10.0*np.max([I_exp_guess,I_gaussian_guess])
        
        fit.sb_profile=[sb_profs.expdisk(guesses=[I_exp_guess,r_scale_exp_guess],minimums=[0.0,0.0],maximums=[I_max,r_ub],fixed=[True,False]),
                 sb_profs.gaussian(guesses=[I_gaussian_guess,r_o_gaussian_guess,r_scale_gaussian_guess],minimums=[0.0,0.0,0.0],maximums=[I_max,r_ub,r_ub],fixed=[False,False,False])]
    if SB_form == 'exp_sersic':
        I_exp_guess = SB_params[0]
        r_scale_exp_guess = SB_params[1]  
        
        I_sersic_guess = SB_params[2]
        r_o_sersic_guess = SB_params[3]
        r_scale_sersic_guess = SB_params[4]
        n_init_sersic_guess = SB_params[5]
        
        I_max = 10.0*np.max([I_exp_guess,I_sersic_guess])
        
        fit.sb_profile=[sb_profs.expdisk(guesses=[I_exp_guess,r_scale_exp_guess],minimums=[0,0],maximums=[I_max,r_ub],fixed=[True,False]),
                 sb_profs.mod_sersic(guesses=[I_sersic_guess,r_o_sersic_guess,r_scale_sersic_guess,n_init_sersic_guess],minimums=[0.0,0.0,0.0,0.0],maximums=[I_max,r_ub,r_ub,4.0],fixed=[False,False,False,False])]
    if SB_form == 'gaussian_sersic':
        I_gaussian_guess = SB_params[0]
        r_o_gaussian_guess = SB_params[1]
        r_scale_gaussian_guess = SB_params[2]  
        
        I_sersic_guess = SB_params[3]
        r_o_sersic_guess = SB_params[4]
        r_scale_sersic_guess = SB_params[5]
        n_init_sersic_guess = SB_params[6]
        
        I_max = 10.0*np.max([I_gaussian_guess,I_sersic_guess])
        
        fit.sb_profile=[sb_profs.gaussian(guesses=[I_gaussian_guess,r_o_gaussian_guess,r_scale_gaussian_guess],minimums=[0.0,0.0,0.0],maximums=[I_max,r_ub,r_ub],fixed=[True,False,False]),
                 sb_profs.mod_sersic(guesses=[I_sersic_guess,r_o_sersic_guess,r_scale_sersic_guess,n_init_sersic_guess],minimums=[0.0,0.0,0.0,0.0],maximums=[I_max,r_ub,r_ub,4.0],fixed=[False,False,False,False])]
    
    #What we are fixing
    if fixed_params != 'none':
        wfPA, wfinc, wfxc_deg, wfyc_deg, wfVsys = fixed_params
        if len(fixed_params) > 0:
            if 'PA' in fixed_params:
                PAguess = float(wfPA[0])
                fit.pa_guess = PAguess
                fit.pa_range = [PAguess,PAguess]
            if 'inc' in fixed_params:
                iguess = float(wfinc[0])
                fit.inc_guess = iguess
                fit.inc_range = [iguess,iguess]
            if 'Xc' in fixed_params:
                xcenterguess = wfxc_deg[0] #THE CENTER GUESS NEEDS TO BE IN RA AND DEC, NOT PIXELS! - Tim Davis (he didn't say it like this)
                fit.xc_guess = xcenterguess
                fit.xcent_range = [xcenterguess,xcenterguess]
            if 'Yc' in fixed_params:
                ycenterguess = wfyc_deg[0] #THE CENTER GUESS NEEDS TO BE IN RA AND DEC, NOT PIXELS! - Tim Davis (he didn't say it like this)
                fit.yc_guess = ycenterguess
                fit.ycent_range = [ycenterguess,ycenterguess]
            if 'Vsys' in fixed_params:
                vsysguess = float(wfVsys[0])
                fit.vsys_guess = vsysguess
                fit.vsys_range = [vsysguess,vsysguess]

    #Output configurations:
    fit.pdf_rootname = run_label
    fit.show_corner = True
    fit.pdf = True
    fit.interactive = False
    fit.show_plots = False
    fit.text_output = True
    fit.output_cube_fileroot = run_label
    bestvals, besterrs, outputvalue, outputll,_ = fit.run(method='mcmc')
    ##print(Table([fit.labels,bestvals,besterrs],names=('Quantity', 'Bestfit','1-sigma error')))
    #
#    ringbins = []
    ringbins = np.array(fit.bincentroids)
    #
    if ringbins != []:
        np.savez(modelling_results_dir+run_label+'_results.npz',bestvals=bestvals,
                                          besterrs=besterrs,
                                          outputvalue=outputvalue,
                                          outputll=outputll,
                                          labels=fit.labels,
                                          bincentroids=fit.bincentroids,
                                          bc=ringbins) #left fixed=fixed out since no parameters were fixed.
    if ringbins == []:
        np.savez(modelling_results_dir+run_label+'_results.npz',bestvals=bestvals,
                                          besterrs=besterrs,
                                          outputvalue=outputvalue,
                                          outputll=outputll,
                                          labels=fit.labels)
        
    #Now shuttle the results to the proper directory:
    simcubepath = run_label+'_simcube.fits'
    
    os.system('mv '+fit.pdf_rootname+'.pdf '+modelling_results_dir+fit.pdf_rootname+'_summary_plot.pdf')
    os.system('mv '+fit.pdf_rootname+'.npz '+modelling_results_dir+fit.pdf_rootname+'.npz')
    if os.path.exists(fit.pdf_rootname+'_MCMCcornerplot.pdf') == False:
        f = open(modelling_results_dir+'/'+run_label+'_error_message','w')
        f.write('Some parameters had no accepted guesses. Skipping corner plot. Try increasing niters.'+'\n')
        f.close()
    if os.path.exists(fit.pdf_rootname+'_MCMCcornerplot.pdf') == True:
        os.system('mv '+fit.pdf_rootname+'_MCMCcornerplot.pdf '+modelling_results_dir+run_label+'_MCMCcornerplot.pdf')
    os.system('mv '+simcubepath+' '+modelling_results_dir+model_cube_name)

    if os.path.exists(run_label+'_KinMS_fitter_output.txt') == True:
        os.system('mv '+run_label+'_KinMS_fitter_output.txt '+modelling_results_dir+run_label+'_output_FROM_KinMS.txt')

    return

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('csv_path', type=str, help='the path to the csv input file.')
    args = my_parser.parse_args()
    
    kinms_runner(args.csv_path)
