#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:01:00 2018

@author: vikram
"""
import numpy as np
from hcipy import *

def gs_phase_retrieval(wavelen, pupil, pupil_grid, prop, image, focal_grid, iterations):
    ''' Iteratively go between pupil plane and image plane,
    known values being the pupil shape and image plane intensity.'''
    
    # Start with a random phase
    #pupil_phase = np.random.uniform(low=-np.pi, high=np.pi, size=len(pupil))
    pupil_phase = np.zeros(len(pupil))
    
    # Start the GS loop here
    for iteration in range(iterations):
        # Create a wavefront with the known pupil shape and estimated pupil phase
        wf_est = Wavefront(Field(pupil * np.exp(1j * pupil_phase), pupil_grid), wavelen)
        
        # Propagate this to the focal plane
        focal_plane_est = prop.forward(wf_est)
        
        # Replace the magnitude with the image intensity
        focal_plane_wf = Wavefront(Field(image * np.exp(1j * focal_plane_est.phase), focal_grid), wavelen)
        
        # Propagate this back to the pupil plane
        pupil_plane_wf = prop.backward(focal_plane_wf)
        
        # Assign the estimated pupil phase to this phase
        pupil_phase = pupil_plane_wf.phase - np.mean(pupil_plane_wf.phase)
    
    return pupil_phase