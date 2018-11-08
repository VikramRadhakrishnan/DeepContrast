#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:55:24 2017

Project a wavefront or surface onto an orthonormal basis and recover the coefficients
Or use coefficients of a modal basis to recover the phase of a wavefront

@author: vikram
"""

import numpy as np

def modal_decomposition(wf, basis):
    coeffs = np.dot(inverse_truncated(basis.transformation_matrix), wf.phase)
    return coeffs

def wf_reconstruction(coeffs, basis, pupil_grid):
    wf_phase = np.dot(basis.transformation_matrix, coeffs)
    aperture = np.exp(1j * wf_phase)
    aperture = Field(aperture, pupil_grid)
    wfrecon = Wavefront(aperture)
    return wfrecon