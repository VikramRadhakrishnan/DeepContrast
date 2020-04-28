# -*- coding: utf-8 -*-

## This code calculates the influence matrix and response matrix for a
## Shack-Hartmann wavefront sensor + deformable mirror system

import numpy as np
import matplotlib.pyplot as plt
from hcipy import *

def shack_hartmann_calibrator(wf, shwfs, shwfse, dm, amp, prop, filt=None):
    
    num_modes = len(dm.influence_functions)
    
    dm.actuators = np.zeros(num_modes)
    
    # Get the refernce lenslet measurements
    img = shwfs(wf).intensity
    ref = shwfse.estimate([img]).ravel()
    num_measurements = ref.shape[0]
    
    Infmat = []
    
    plt.ion()
    for dm_mode in np.arange(num_modes):
        
        print("Now calibrating actuator {}/{}".format(dm_mode+1, num_modes))
        
        total_slopes = np.zeros((num_measurements,))
        
        for push in np.array([-amp, amp]):
        
            act_levels = np.zeros(num_modes)
            act_levels[dm_mode] = push
            
            dm.actuators = act_levels
            dm_wf = dm.forward(wf)
            if filt is None:
                print("No spatial filter")
                sh_wf = shwfs.forward(dm_wf)
            else:
                filt_wf = prop.backward(filt.forward(prop(dm_wf)))
                sh_wf = shwfs.forward(filt_wf)
            sh_img = sh_wf.power
            imshow_field(sh_img)
            plt.pause(0.1)
            lenslet_centers = shwfse.estimate([sh_img])
            total_slopes += (lenslet_centers.ravel()- ref)/(2*push)
        Infmat.append(total_slopes)
        
    dm.actuators = np.zeros(num_modes)
        
    Infmat = ModeBasis(Infmat)
    
    plt.ioff()
    return Infmat
