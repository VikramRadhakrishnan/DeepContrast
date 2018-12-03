#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:58:34 2018

Stochastic Parallel Gradient Descent optimizer code for optimizing an AO + coronagraph
optical imaging system. In this test, a nested optimizer is run within another optimizer
loop. The inner optimizer makes use of the Shack-Hartmann wavefront sensor data to optimize
contrast at the virtual, calculated conjugate image plane. This accounts for residual
turbulence and common path aberrations. The outer loop uses a similar optimizer, but optimizes
contrast at the actual science image plane. This accounts for non common path errors
and wavefront reconstruction errors.

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
Dtel=1
tel_pupil_grid = make_pupil_grid(N, Dtel)
tel_aperture = circular_aperture(Dtel)

# Atmosphere parameters
pixels_per_frame = 1
velocity = np.array([pixels_per_frame,0])
L0 = 40
r0 = 0.2
height = 0

# Make atmosphere
np.random.seed(42)
layers = [InfiniteAtmosphericLayer(tel_pupil_grid, Cn_squared_from_fried_parameter(r0, 500e-9), L0, velocity * tel_pupil_grid.delta[0], height, 2)]
atmosphere = MultiLayerAtmosphere(layers, False)

wf_tel = Wavefront(tel_aperture(tel_pupil_grid), wavelength)
wfatms_tel = atmosphere.forward(wf_tel)

## Demagnify wavefront and phase-screen for the optics
mag = Magnifier(10.5e-3/1)
wf = mag.forward(wf_tel)
wfatms = mag.forward(wfatms_tel)

## Create propagator from pupil to focal plane
prop = FraunhoferPropagator(pupil_grid, science_focal_grid, wavelength)

## Create the deformable mirror
actuator_grid = make_pupil_grid(22, D*1.1)
sigma = D/22.
gaussian_basis = make_gaussian_pokes(pupil_grid, actuator_grid, sigma)
dm = DeformableMirror(gaussian_basis)
num_modes = len(dm.influence_functions)
dm.actuators = np.zeros(num_modes)

## Get the app coronagraph
app_amp = fits.getdata('/home/vikram/Work/DeepContrast/coronagraphs/Square_20_80_20_25_0_2_amp_resampled_512.fits').ravel()
app_phase = fits.getdata('/home/vikram/Work/DeepContrast/coronagraphs/Square_20_80_20_25_0_2_phase_resampled_512.fits').ravel()
app = Apodizer(app_amp * np.exp(1j * app_phase))

## Create detector
flatfield = 0.05 # = 5% flat field error 
darkcurrentrate = 2 # = dark current counts per second
readnoise = 100 # = rms counts per read out
photonnoise = True

#Creating our detector.
#science_camera = NoisyDetector(input_grid=science_focal_grid, include_photon_noise=photonnoise, flat_field=flatfield, dark_current_rate=darkcurrentrate, read_noise=readnoise)
science_camera = NoiselessDetector()
shack_hartmann_camera = NoiselessDetector()

## Create a spatial filter
filt_aperture = circular_aperture(22)
spatial_filter = Apodizer(filt_aperture(science_focal_grid))

## Create the Shack-Hartmann Wavefront sensor
F_mla = 40. / 0.3
N_mla = 22
D_mla = 10.5e-3
shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid, F_mla, N_mla, D_mla)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

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

## Make intial phasescreen
atmosphere.evolve_until(1)
wfatms = atmosphere.forward(wf)

########### The test starts here #############################################

# Data to keep track of
cost_function = []
strehl_evol = []
contrast_evol = []
modes_evolution = []

# Reset the learning rate, gradients, and perturbatory sequence
lrate = 0.001
amp = 1e-9
iterations = 100000
merit = 999.99e9
grads = np.zeros(dm.actuators.size,)
dm_sequence = np.random.choice([-1, 1], size=(num_modes, iterations)) * amp

# Start inner loop closed-loop control here
for loop in np.arange(iterations):
    dm_wf = dm.forward(wfatms)
    
    # Science optical path here
    ncp_wf = ncp.forward(dm_wf)
    app_wf = app.forward(ncp_wf)
    science_camera.integrate(prop(app_wf), 1, 1)
    sci_img = science_camera.read_out()
    
    # Wavefront sensor optical path here
    # Implement spatial filter to simulate reconstructed wavefront
    reconstruct_wf = prop.backward(spatial_filter.forward(prop(dm_wf)))
    
    # Control path here
    wfout = app.forward(reconstruct_wf)
    test_img = prop(wfout).power
    
    actual_strehl = sci_img[np.argmax(diff_lim_img)] / diff_lim_img.max()
    actual_contrast = sci_img[dz_ind].mean() / diff_lim_img.max()
    
    test_strehl = test_img[np.argmax(diff_lim_img)] / diff_lim_img.max()
    test_contrast = test_img[dz_ind].mean() / diff_lim_img.max()
    new_merit = np.sqrt(test_contrast) / test_strehl
    
    # If the update in the cost function is small, or the merit has worsened
    if merit - new_merit < 1e-4:
        # If there has been an improvement (decrease) in the cost function
        # Then increase the learning rate
        if new_merit < merit:
            lrate *= 1.05
        # Otherwise decrease learning rate and go back to previous actuator values
        else:
            dm.actuators = modes_evolution[-1].copy()
            new_merit = merit
            lrate *= 0.7
    
    merit = new_merit
    cost_function.append(merit)
    
    strehl_evol.append(actual_strehl)
    contrast_evol.append(actual_contrast)
    modes_evolution.append(dm.actuators.copy())
    
    if loop % 1000 == 0:
        print("Loop {0}: Calculated strehl: {1:.4f} contrast: {2:.4E} cost: {3:.4E}\
              \n Actual strehl: {4:.4f} contrast: {5:.4E}\n"\
              .format(loop, test_strehl, test_contrast, merit,\
                      actual_strehl, actual_contrast))
    
    # Make a measurement of the metric with DM actuators with delta added
    dm.actuators += dm_sequence[:,loop]
    dm_wf = dm.forward(wfatms)
    reconstruct_wf = prop.backward(spatial_filter.forward(prop(dm_wf)))
    wfout = app.forward(reconstruct_wf)
    test_img = prop(wfout).power
    
    test_strehl = test_img[np.argmax(diff_lim_img)] / diff_lim_img.max()
    test_contrast = test_img[dz_ind].mean() / diff_lim_img.max()
    merit_delta = np.sqrt(test_contrast) / test_strehl
    
    # Take the difference between measurements
    # This is equivalent to having a low pass filter of the cost function
    LF = merit_delta - merit
    
    # Multiply by the delta amplitudes vector and divide by amp^2 to get gradient
    grads = (LF * dm_sequence[:,loop])# / (amp ** 2)
    
    # Calculate new DM actuator positions using learning rate and gradient
    dm.actuators -= lrate * grads