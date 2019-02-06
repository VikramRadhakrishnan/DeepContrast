#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:08:36 2019

Code to simulate a coronagraphic high contrast imaging instrument, with NCPA.
The AO system runs at 1kHz and the focal plane cameras are read out at 1 Hz.
There are two focal plane images, that differ by a known phase aberration.
The goal is to use phase diversity techniques to estimate the NCPA.

@author: vikram
"""

## Necessary imports
import numpy as np
from hcipy import *
from hcipy.atmosphere import *
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
import time

#from utils.shack_hartmann_calibrator import shack_hartmann_calibrator

## Create aperture and pupil/focal grids
wavelength = 532e-9
N = 512
D = 10.5e-3
pupil_grid = make_pupil_grid(N, D)
science_focal_grid = make_focal_grid(pupil_grid, 8, 20, wavelength)
wfs_focal_grid = make_focal_grid(pupil_grid, 8, 20, wavelength)
aperture = circular_aperture(D)

# Telescope parameters
Dtel=4
tel_pupil_grid = make_pupil_grid(N, Dtel)
tel_aperture = circular_aperture(Dtel)

## Create the deformable mirror
actuator_grid = make_pupil_grid(25, D*1.1)
sigma = D/25
gaussian_basis = make_gaussian_pokes(pupil_grid, actuator_grid, sigma)
dm = DeformableMirror(gaussian_basis)
num_modes = len(dm.influence_functions)
dm.actuators = np.zeros(num_modes)

# Atmosphere parameters
pixels_per_frame = 1
velocity = np.array([pixels_per_frame,0])
L0 = 40
r0 = 0.2
height = 0

# Make atmosphere
np.random.seed(42)
layers = []
layer = InfiniteAtmosphericLayer(tel_pupil_grid, Cn_squared_from_fried_parameter(r0, 500e-9), L0, velocity * tel_pupil_grid.delta[0], height, 2)
layer2 = ModalAdaptiveOpticsLayer(layer, dm.influence_functions, 1)
layers.append(layer2)
atmosphere = MultiLayerAtmosphere(layers, False)

# Make initial phasescreen
wf_tel = Wavefront(tel_aperture(tel_pupil_grid), wavelength)
wf_tel.total_power = 1
atms_time = 1
atmosphere.evolve_until(atms_time)
atms_time += 1
atmosphere.evolve_until(atms_time)
wfatms_tel = atmosphere.forward(wf_tel)

## Demagnify wavefront and phase-screen for the optics
mag = Magnifier(10.5e-3/4)
wf = mag.forward(wf_tel)
wfatms = mag.forward(wfatms_tel)

## Create propagator from pupil to focal plane
prop = FraunhoferPropagator(pupil_grid, science_focal_grid, wavelength)

## Get the app coronagraph
app_amp = fits.getdata('/home/vikram/Work/DeepContrast/coronagraphs/Square_20_80_20_25_0_2_amp_resampled_512.fits').ravel()
app_phase = fits.getdata('/home/vikram/Work/DeepContrast/coronagraphs/Square_20_80_20_25_0_2_phase_resampled_512.fits').ravel()
app = Apodizer(app_amp * np.exp(1j * app_phase))

## Create a known phase diversity aberration
num_div_coeffs = 5
div_coeffs = np.zeros(num_div_coeffs)
div_coeffs[2] = 1
div_zernike_basis = make_zernike_basis(num_div_coeffs, D, pupil_grid, 2)
diversity_phase = np.dot(div_zernike_basis.transformation_matrix, div_coeffs)
diversity = Apodizer(np.exp(1j * diversity_phase))

## Create detector
flatfield = 0.05 # = 5% flat field error 
darkcurrentrate = 2 # = dark current counts per second
readnoise = 100 # = rms counts per read out
photonnoise = True

#Creating our detector.
#science_camera = NoisyDetector(input_grid=science_focal_grid, include_photon_noise=photonnoise, flat_field=flatfield, dark_current_rate=darkcurrentrate, read_noise=readnoise)
science_camera = NoiselessDetector()
diversity_camera = NoiselessDetector()

## Create a spatial filter
filt_aperture = circular_aperture(25)
spatial_filter = Apodizer(filt_aperture(science_focal_grid))

## Generate a diffraction limited image for metrics
diff_lim_img = prop(wf).power

## Get the unit lambda/D
l_D = wavelength / D
plot_grid = make_focal_grid(make_pupil_grid(512), 8, 20)

## Create a noiseless camera image from the perfectly flat wavefront with coronograph
wfdm = dm.forward(wf)
wfapp = app.forward(wfdm)
imapp = prop(wfapp).power
dz_ind = np.where((imapp.grid.x >= (2 * l_D)) &\
                  (imapp.grid.x <= (8 * l_D)) &\
                  (imapp.grid.y >= (-3 * l_D)) &\
                  (imapp.grid.y <= (3 * l_D)))

# Create an NCP aberration
num_coeffs = 40
plaw_index = -1.
np.random.seed(42)
coeffs = ((np.random.rand(num_coeffs) - 0.5) * 2 ) * (np.arange(num_coeffs) + 1) ** plaw_index
zernike_basis = make_zernike_basis(num_coeffs, D, pupil_grid, 2)
ncp_phase = np.dot(zernike_basis.transformation_matrix, coeffs)
ncp = Apodizer(np.exp(1j * ncp_phase))

########### The test starts here #############################################
duration = 1000 # Duration of the simulation in milliseconds
phasescreens = []

# Start the simulation here
for phasescreen in np.arange(duration):
    dm_wf = dm.forward(wfatms)
    
    # Science optical path here
    ncp_wf = ncp.forward(dm_wf)
    app_wf = app.forward(ncp_wf)
    science_camera.integrate(prop(app_wf), dt=1e-3)
    
    # Phase diversity image path here
    div_wf = diversity.forward(app_wf)
    diversity_camera.integrate(prop(div_wf), dt=1e-3)
    
    atms_time += 1
    atmosphere.evolve_until(atms_time)
    wfatms_tel = atmosphere.forward(wf_tel)
    wfatms = mag.forward(wfatms_tel)
    phasescreens.append(wfatms)

sci_img = science_camera.read_out()
div_img = diversity_camera.read_out()

# Images to test phase diversity phase retrieval
test_img = prop(ncp.forward(wf)).power
test_div_img = prop(diversity.forward(ncp.forward(wf))).power

# Parameters for the model
zero_phase = np.zeros(app_phase.shape)
transmitter = Apodizer(np.exp(1j * zero_phase))