#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:07:48 2019

Phase diversity metric function.
This function takes in a vector of coefficients, and a tuple of args.
The args in order are as follows:
    - basis: the polynomial basis with respect to the coefficients
    - pupil_grid: the grid on which the pupil plane wavefront is calculated
    - coronagraph: the coronagraphic pupil plane apodizer
    - diversity: the diversity phase apodizer element
    - prop: the pupil plane to focal plane propagator function
    - img: the measured focal plane image
    - div_img: the measured diversity image
    - l_D: the unit of measurement in the focal plane i.e. wavelength by diameter

@author: vikram
"""
import numpy as np
from hcipy import *

def modal_decomposition(wf, basis):
    coeffs = np.dot(inverse_truncated(basis.transformation_matrix), wf.phase)
    return coeffs

def wf_reconstruction(coeffs, basis, pupil_grid, wavelength, aperture=1):
    wf_phase = np.dot(basis.transformation_matrix, coeffs)
    e_field = Field(np.exp(1j * wf_phase), pupil_grid)
    wfrecon = Wavefront(e_field * aperture(pupil_grid), wavelength)
    wfrecon.total_power=1
    return wfrecon

def diversity_metric(x, basis, pupil_grid, wavelength, aperture, coronagraph, diversity, prop, img, div_img, l_D):
    # First calculate the estimated wavefront from the coefficients
    est_wf = wf_reconstruction(x, basis, pupil_grid, wavelength, aperture)
    
    # Propagate this estimated wf to the focal plane
    est_coron_wf = coronagraph.forward(est_wf)
    est_img = prop(est_coron_wf).power
    
    # Propagate this estimated wf to the focal plane with phase diversity
    est_div_wf = diversity.forward(est_coron_wf)
    est_div_img = prop(est_div_wf).power
    
    # Crop the PSF of these images
#    cent_ind = np.where((img.grid.x >= (-8 * l_D)) &\
#                        (img.grid.x <= (8 * l_D)) &\
#                        (img.grid.y >= (-8 * l_D)) &\
#                        (img.grid.y <= (8 * l_D)))
#    est_img_psf = est_img[cent_ind]
#    est_div_img_psf = est_div_img[cent_ind]
#    img_psf = img[cent_ind]
#    div_img_psf = div_img[cent_ind]
#    
#    # Normalize the images
#    norm_est_img = est_img_psf #/ est_img_psf.max()
#    norm_est_div_img = est_div_img_psf #/ est_div_img_psf.max()
#    norm_img = img_psf #/ img_psf.max()
#    norm_div_img = div_img_psf #/ div_img_psf.max()
#    
    # Calculate the metric
    metric = 0.5 * np.sum((img - est_img) ** 2) + \
    0.5 * np.sum((div_img - est_div_img) ** 2)
    
    return metric

def test_function(x, coeff1, coeff2, coeff3, coeff4, coeff5):
    result = coeff1 * (x[0] ** 2) + coeff2 * (x[1] ** 2) + \
    coeff3 * x[0] + coeff4 * x[1] + coeff5
    return result