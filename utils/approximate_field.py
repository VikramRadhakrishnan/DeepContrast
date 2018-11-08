#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:50:05 2018

Generate the first four components of a PSF according to equation [3]
of the paper on speckle decorrelation ( http://iopscience.iop.org/article/10.1086/345826/pdf )

@author: vikram
"""

import numpy as np
from math import factorial
from hcipy import Apodizer

def approximate_field(wf, phi, terms):
    ''' A function to approximate the phase aberration by a series expansion.
    Returns approximated wavefronts - such as the linear, quadratic, cubic and higher order terms.
    wf - the aperture illumination wavefront
    phi - the phase of the aberration
    terms - the number of terms to approximate up to'''
    
    field_approx = []
    
    for term in np.arange(terms+1):
        phi_approx = 0
        for index in np.arange(term):
            phi_approx += np.power((1j * phi), index) / factorial(index)
        t = Apodizer(phi_approx).forward(wf)
        field_approx.append(t)
#    
#    phi1 = 1 + 1j * phi
#    phi2 = 1 + 1j * phi - np.square(phi)/factorial(2)
#    phi3 = 1 + 1j * phi - np.square(phi)/factorial(2) - (1j * np.power(phi, 3)/factorial(3))
#    
#    t1 = Apodizer(phi1).forward(wf)
#    t2 = Apodizer(phi2).forward(wf)
#    t3 = Apodizer(phi3).forward(wf)
    
    return field_approx