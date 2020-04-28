# -*- coding: utf-8 -*-

import numpy as np
from hcipy import *
from hcipy.atmosphere import *
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
import time


def shack_hartmann_calibrator(wf, shwfs, shwfse, dm, amp, prop, wavelength,filt=None):
    
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
            act_levels[dm_mode] = push*wavelength
            
            dm.actuators = act_levels
            dm_wf = dm.forward(wf)
            if filt is None:
                print("No spatial filter")
                sh_wf = shwfs.forward(dm_wf)
            else:
                filt_wf = prop.backward(filt.forward(prop(dm_wf)))
                sh_wf = shwfs.forward(filt_wf)
            sh_img = sh_wf.intensity
            imshow_field(sh_img)
            plt.pause(0.1)
            lenslet_centers = shwfse.estimate([sh_img])
            total_slopes += (lenslet_centers.ravel()- ref)/(2*push)
        Infmat.append(total_slopes)
        
    dm.actuators = np.zeros(num_modes)
        
    Infmat = ModeBasis(Infmat)
    
    plt.ioff()
    return Infmat


## Create aperture and pupil/focal grids
wavelength = 1e-6
N = 256*2
D = 1.0
pupil_grid = make_pupil_grid(N, 1.3*D)
science_focal_grid = make_focal_grid(pupil_grid, 8, 20).scaled(wavelength)
wfs_focal_grid = make_focal_grid(pupil_grid, 8, 20, wavelength)
aperture = circular_aperture(D)



## Create the wavefront at the entrance pupil
wf = Wavefront(aperture(pupil_grid),wavelength)

## Create propagator from pupil to focal plane
prop = FraunhoferPropagator(pupil_grid, science_focal_grid, wavelength)

## Create propagator from focal plane to pupil plane
prop2 = FraunhoferPropagator(science_focal_grid, pupil_grid, wavelength)

## Propagate the wavefront to create a perfect psf
diff_lim_img = prop(wf).intensity

## Create the deformable mirror
actuator_grid = make_pupil_grid( 9, D)
#sigma = D/9
#gaussian_basis = make_gaussian_pokes(pupil_grid, actuator_grid, sigma)
#dm = DeformableMirror(gaussian_basis)
num_modes = 20
modes = make_zernike_basis(num_modes, D, pupil_grid, 1, False)
dm = DeformableMirror(modes)


num_modes = len(dm.influence_functions)
dm.actuators = np.zeros(num_modes)

## Create the microlens array
F_mla = 1e4
N_mla = 20
D_mla = D
shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid, F_mla, N_mla, D_mla)
shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

## Simulate atmosperic turbulence on wavefront
r_0 = 0.15
spectral_noise_factory = SpectralNoiseFactoryFFT(von_karman_psd, pupil_grid, 8)
turbulence_layers = make_standard_multilayer_atmosphere(r_0, wavelength=wavelength)
atmospheric_model = AtmosphericModel(spectral_noise_factory, turbulence_layers)

## Get the unit lambda/D
l_D = wavelength / D
plot_grid = make_focal_grid(make_pupil_grid(512), 8, 20)

## Create a noiseless camera image from the perfectly flat wavefront with coronograph
wfdm = dm.forward(wf)


## Calibrate the wavefront sensor
wfs_infmat = shack_hartmann_calibrator(wf, shwfs, shwfse, dm, 0.05, prop,wavelength)

wfs_infmat = ModeBasis(wfs_infmat)
control_mat = inverse_truncated(wfs_infmat.transformation_matrix, rcond=1e-2)


# Lets get the proper lenslet measurements we want
dm_wf = dm.forward(wf)
#filt_wf = prop.backward(spatial_filter.forward(prop(dm_wf)))
ref_img = shwfs.forward(dm_wf).intensity
ref_slopes = shwfse.estimate([ref_img]).ravel()

# Set AO gain
gain = -0.3
# Set leaky integrator factor
leak =0

# Get a new phase-screen
wfatms = atmospheric_model(wf)

#atms_amp = np.load("/home/vikram/Work/Backups/SPIE_Paper/Amp_screen.npy")
#atms_phase = np.load("/home/vikram/Work/Backups/SPIE_Paper/Phase_screen.npy")
#wfatms = wf.copy()
#wfatms.electric_field *= atms_amp * np.exp(1j * atms_phase)

# Start closed-loop control here
for loop in range(30):
    dm_wf = dm.forward(wfatms)
    
    # Science optical path here
    #app_wf = app.forward(dm_wf)
    sci_img = prop(dm_wf).intensity
    
    # Wavefront sensor optical path here
    #filt_wf = prop.backward(spatial_filter.forward(prop(dm_wf)))
    sh_wf = shwfs.forward(dm_wf)
    sh_img = sh_wf.intensity
    
    # Control path here
    meas_vec = shwfse.estimate([sh_img])
    
#    if loop != 0:
#        # Remove piston and tip/tilt
#        meas_vec[0] -= meas_vec[0].mean()
#        meas_vec[1] -= meas_vec[1].mean()
        
    meas_vec = meas_vec.ravel()
    
    #actuator_vals = control_mat.dot(meas_vec-ref_slopes)
    dm.actuators +=wavelength* gain*control_mat.dot(meas_vec-ref_slopes)
    
    # Plot results
    plt.clf()
    plt.subplot(2,2,1)
    imshow_field(np.log10(sci_img / sci_img.max() ))
    titlestring = 'Actual image after ' + str(loop) + ' iterations'
    plt.title(titlestring)
    plt.colorbar()
    
    plt.subplot(2,2,2)
    imshow_field(dm_wf.phase, cmap='RdBu')
    titlestring = 'Residuals after ' + str(loop) + ' iterations'
    plt.title(titlestring)
    plt.colorbar()
    plt.subplot(2,2,3)
    imshow_field(sh_img)
    titlestring = 'SHWFS spots ' + str(loop) + ' iterations'
    plt.title(titlestring)
    plt.colorbar()
    plt.subplot(2,2,4)
    imshow_field(wfatms.phase, cmap='RdBu')
    titlestring = 'Full turbulence ' + str(loop) + ' iterations'
    plt.title(titlestring)
    plt.colorbar()
    plt.show()
    plt.pause(0.1)
#    title = '/home/vikram/Work/DeepContrast/tests/frames/closed_loop_AO_' + str(loop).zfill(2) + '.jpg'
#    plt.savefig(title)
    
    strehl = sci_img[np.argmax(diff_lim_img)] / diff_lim_img.max()
    #contrast = sci_img[dz_ind].mean() / diff_lim_img.max()
    
    #print("After {0:2d} iterations strehl is {1:.4f} and contrast is {2:1.2e}".format(loop, strehl, contrast))
    print("After {0:2d} iterations strehl is {1:.4f}".format(loop, strehl))