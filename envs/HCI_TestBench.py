# State --> the current wavefront sensor reading (projected in actuator space) concatenated with current actuator values
# Action --> additional pushes to apply on actuators to address the current wavefront, assuming no lag for now
# Reward --> Create an additional function which takes in the image, the dark hole indices, and calculates contrast
#            The negative of the log of the contrast is the starting point for reward shaping

# Necessary imports
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from hcipy import *
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
import os

from utils.shack_hartmann_calibrator import shack_hartmann_calibrator

class HCI_TestBench(gym.Env):
    
    def __init__(self, wavefront, timestep, turbulence_generator, demag, dm, wfs, wfse, ncp, coronagraph, prop, sci_cam, dh_ind):
        '''Variables used to characterize testbench -
        wavefront: the flat wavefront, characterized by the aperture function
        turbulence_generator: optical component to add evolving phase and/or amplitude aberrations
        demag: demagnifier to bring the wavefront down to the diameter of the instrument optics
        dm: deformable or/and movable mirror component
        wfs: optical elements in a separate optical path downstream of DM, for wavefront sensing
        wfse: estimator for the wavefront sensor
        ncp: non common path aberration taken here to be a separate apodizer
        coronagraph: pupil plane coronagraph optic
        prop: propagator from pupil plane to focal plane
        sci_cam: science camera detector
        dh_ind: indices of pixels within dark hole region that must be kept dark
        '''
        self.seed()

        self._wavefront = wavefront
        self._turbulence_generator = turbulence_generator
        self._demag = demag
        self._dm = dm
        self._wfs = wfs
        self._wfse = wfse
        self._ncp = ncp
        self._coronagraph = coronagraph
        self._prop = prop
        self._sci_cam = sci_cam
        self._dh_ind = dh_ind

        self._inverse_tm = inverse_tikhonov(dm.influence_functions.transformation_matrix.toarray(), rcond=1e-4)

        self._dm_actside = int(round(np.sqrt(len(self._dm.actuators))))
      
        self._timestep = timestep # A unit of time (let's say milliseconds)
      
        self._actstroke = 1 # maximum allowed stroke in microns
      
        # Calculate the diffraction limited science image for Strehl calculations
        self._diff_lim_img = self._prop(self._demag.forward(self._wavefront)).power

        self._dlmax = np.argmax(self._diff_lim_img) # Used in Strehl calculation or metric calculation

        # Calibrate the WFS or use a pre-calibrated control matrix
        if os.path.exists("control_mat.npy"):
            print("Loading pre-calibrated control matrix")
            self._control_mat = np.load("control_mat.npy")
        else:
            print("Calibrating control matrix\n")
            wfs_infmat = shack_hartmann_calibrator(wf, wfs, wfse, dm, 0.01e-6, prop, None)
            self._control_mat = inverse_tikhonov(wfs_infmat.transformation_matrix, rcond=2e-2)
            np.save("control_mat", self._control_mat)

        # Flat wavefront lenslet measurements for reference
        ref_img = self._wfs.forward(self._demag.forward(self._wavefront)).power
        self._ref_slopes = self._wfse.estimate([ref_img]).ravel()
      
        self._action_space = spaces.Box(-self._actstroke, self._actstroke, shape=(self._dm_actside, self._dm_actside), dtype='float64')
        self._observation_space = spaces.Box(-np.inf, np.inf, shape=(self._dm_actside, self._dm_actside, 2), dtype='float64')
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def max_action(self):
        return self._actstroke

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _turbulence(self):
        '''Propagate a wavefront through the turbulence generator (atmosphere).
        turb --> demagnify
        '''
        wf = self._turbulence_generator.forward(self._wavefront)
        wf = self._demag.forward(wf)
        return wf
    
    def _wfs_forward(self):
        '''Propagate the wavefront through the wavefront sensor and project the measurements on the DM mode basis
        '''
        # Propagate the wavefront through the turbulence and WFS
        wfatms = self._turbulence()
        wf = self._dm.forward(wfatms)
        sh_img = self._wfs.forward(wf).power

        # Get the measured slopes
        meas_vec = (self._wfse.estimate([sh_img])).ravel()

        # Calculate the DM amplitudes
        amplitudes = self._control_mat.dot(meas_vec-self._ref_slopes)

        # Remove piston by subtracting the mean amplitude
        amplitudes -= np.mean(amplitudes)

        # The amplitudes to put on the DM are the negative of these amplitudes
        return -amplitudes.copy()

    def _forward(self):
        '''Propagate a wavefront through the optics.
        DM --> NCPA --> Coronagraph --> focal plane
        '''
        wfatms = self._turbulence()
        wf = self._dm.forward(wfatms)
        wf = self._ncp.forward(wf)
        wf = self._coronagraph.forward(wf)
        wf = self._prop(wf)
        return wf

    def _calc_contrast(self):
        return self._img[self._dh_ind].mean() / self._img[self._dlmax]

    def _calc_strehl(self):
        return self._img[self._dlmax] / self._diff_lim_img.max()

    def step(self, action):
        '''Step through the environment. Performs the following:
        1. Sets the DM actuators.
        2. Propagate WF through the wavefront sensor optics.
        3. Calculate the measured slopes.
        4. Propagate through science camera optics.
        5. Integrate on science camera.
        6. Every N timesteps, read the wavefront sensor camera and update the obs.
        7. Calculate strehl or contrast etc and create the reward.
        8. Return observation space, reward, done, and info.
        9. Increment timestep.
        '''

        # The action is in units of microns, so convert this to standard units first
        action = action.clip(-1, 1)
        action *= 1e-6

        # Put this on the DM 
        action = action.reshape(-1,)
        assert action.shape == self._dm.actuators.shape, "Action shape does not match DM"

        self._dm.actuators = action.copy()
      
        # Science optical path
        self._sci_cam.integrate(self._forward(), dt=1)
        self._img = self._sci_cam.read_out()

        # Reward is the negative log of contrast
        #reward = -np.log10(self._calc_contrast())
        strehl = self._calc_strehl()

        # If the contrast is below a threshold we stop
        done = strehl <= 1e-6

        reward = strehl
      
        # Now that the previous phase was corrected by the DM, increment the turbulence generator for next state
        self._turbulence_generator.t += self._timestep

        # Wavefront sensor optical path
        self._wfs_phase_acts = self._wfs_forward()

        state_vector = np.concatenate((self._wfs_phase_acts.copy().reshape((self._dm_actside, self._dm_actside, -1)),
                                       self._dm.actuators.copy().reshape((self._dm_actside, self._dm_actside, -1))), axis=-1)

        # State is the incident wavefront in actuator space measured in units of microns
        self.state = state_vector / 1e-6


        return self.state, reward, done, {}

    def reset(self, atmseed=None):
        '''Does the following:
        1. Reset turbulence generator.
        2. Set DM actuators to values that would flatten DM
        3. Calculate the initial state from these values
        '''

        np.random.seed(atmseed)

        # Reset the turbulence generator
        self._turbulence_generator.evolve_until(None)
        # Make sure lag is accounted for
        for loop in np.arange(0.001, 0.1, 0.001):
            self._turbulence_generator.evolve_until(loop)

        self._dm.actuators = np.zeros(self._dm.actuators.shape) # Reset DM
        self._wfs_phase_acts = self._wfs_forward() # Calculate wavefront flattening actuators

        self._sci_cam.integrate(self._forward(), dt=1)
        self._img = self._sci_cam.read_out() # Focal plane image
        
        # From here on the controller keeps trying to improve contrast until it goes below a threshold.
        state_vector = np.concatenate((self._wfs_phase_acts.copy().reshape((self._dm_actside, self._dm_actside, -1)),
                                       self._dm.actuators.copy().reshape((self._dm_actside, self._dm_actside, -1))), axis=-1)


        # State is in units of microns
        self.state = state_vector / 1e-6
      
        return self.state
        
    def render(self, mode='human'):
        '''Display the state:
        Display reconstructed wavefront and science image.
        '''
        raise NotImplementedError

    def close(self):
        if self.viewer: self.viewer.close()

