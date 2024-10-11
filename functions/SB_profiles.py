#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 23:52:59 2023

@author: DhruvB
"""

#SB_profiles.py

import numpy as np

def exp(r, I_exp, r_scale):
    return I_exp*np.exp(-r / r_scale)

def gaussian(r, I_gaussian, r_o, r_scale):
    z = (r - r_o) / r_scale
    return I_gaussian*np.exp(-z*z/2.0)

def sersic(r, I_sersic, r_o, r_scale, n): #r_o is mean, r_scale is scale length
    return I_sersic*np.exp(-(((np.abs(r - r_o)) / r_scale) ** n))

def exp_gaussian(r, I_exp      , r_scale_exp, I_gaussian      , r_o_gaussian, r_scale_gaussian):
    return exp(r,I_exp,r_scale_exp) + gaussian(r,I_gaussian,r_o_gaussian,r_scale_gaussian)

def exp_sersic(r, I_exp      , r_scale_exp, I_sersic      , r_o_sersic, r_scale_sersic, n):
    return exp(r,I_exp,r_scale_exp) + sersic(r,I_sersic,r_o_sersic,r_scale_sersic,n)

def gaussian_sersic(r, I_gaussian      , r_o_gaussian    , r_scale_gaussian, I_sersic, r_o_sersic, r_scale_sersic, n):
    return gaussian(r,I_gaussian,r_o_gaussian,r_scale_gaussian) + sersic(r,I_sersic,r_o_sersic,r_scale_sersic,n)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#def exp(r, r_scale, I_exp = 1.0):
#    return I_exp*np.exp(-r / r_scale)
#
#def gaussian(r, r_o, r_scale, I_gaussian = 1.0):
#    z = (r - r_o) / r_scale
#    return I_gaussian*np.exp(-z*z/2.0)
#
#def sersic(r, r_o, r_scale, n, I_sersic = 1.0):
#    return I_sersic*np.exp(-(((np.abs(r - r_o)) / r_scale) ** n))
#
#def exp_gaussian(r, I_exp      , r_scale_exp, I_gaussian      , r_o_gaussian, r_scale_gaussian):
#    return exp(r,r_scale_exp,I_exp) + gaussian(r,r_o_gaussian,r_scale_gaussian,I_gaussian)
#
#def exp_sersic(r, I_exp      , r_scale_exp, I_sersic      , r_o_sersic, r_scale_sersic, n):
#    return exp(r,r_scale_exp,I_exp) + sersic(r,r_o_sersic,r_scale_sersic,n,I_sersic)
#
#def gaussian_sersic(r, I_gaussian      , r_o_gaussian    , r_scale_gaussian, I_sersic, r_o_sersic, r_scale_sersic, n):
#    return gaussian(r,r_o_gaussian,r_scale_gaussian,I_gaussian) + sersic(r,r_o_sersic,r_scale_sersic,n,I_sersic)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#def exp(r, sigma_o, r_scale):
#    return sigma_o * np.exp(-r / r_scale)
#
#def gaussian(r, sigma_o, r_o, r_scale):
#    return sigma_o * np.exp(-(((np.abs(r - r_o)) / r_scale) ** 2))
#
#def sersic(r, sigma_o, r_o, r_scale, n):
#    return sigma_o * np.exp(-(((np.abs(r - r_o)) / r_scale) ** n))
#
#def exp_gaussian(r, sigma_o_exp, r_scale_exp, sigma_o_gaussian, r_o_gaussian, r_scale_gaussian):
#    exp_component = exp(r, sigma_o_exp, r_scale_exp)
#    gaussian_component = gaussian(r, sigma_o_gaussian, r_o_gaussian, r_scale_gaussian)
#    return exp_component + gaussian_component
#
#def exp_sersic(r, sigma_o_exp, r_scale_exp, sigma_o_sersic, r_o_sersic, r_scale_sersic, n_sersic):
#    exp_component = exp(r, sigma_o_exp, r_scale_exp)
#    sersic_component = sersic(r, sigma_o_sersic, r_o_sersic, r_scale_sersic, n_sersic)
#    return exp_component + sersic_component
#
#def gaussian_sersic(r, sigma_o_gaussian, r_scale_gaussian, r_o_gaussian, sigma_o_sersic, r_o_sersic, r_scale_sersic, n_sersic):
#    gaussian_component = gaussian(r, sigma_o_gaussian, r_o_gaussian, r_scale_gaussian)
#    sersic_component = sersic(r, sigma_o_sersic, r_o_sersic, r_scale_sersic, n_sersic)
#    return gaussian_component + sersic_component
