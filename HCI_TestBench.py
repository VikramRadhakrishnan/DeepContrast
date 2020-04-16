# State --> the current incident wavefront (in actuator space)
# Action --> pushes to apply on actuators to address the current wavefront, assuming no lag for now
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

class HCI_TestBench(gym.Env):
    
    def __init__(self, wavefront, turbulence_generator, demag, dm, wfs, ncp, coronagraph, prop, sci_cam, dh_ind):
        '''Variables used to characterize testbench -
        wavefront: the wavefront, characterized by the aperture function
        turbulence_generator: optical component to add evolving phase and/or amplitude aberrations
        demag: demagnifier to bring the wavefront down to the diameter of the instrument optics
        dm: deformable or/and movable mirror component
        wfs: optical elements in a separate optical path downstream of DM, for wavefront sensing
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
        self._ncp = ncp
        self._coronagraph = coronagraph
        self._prop = prop
        self._sci_cam = sci_cam
        self._dh_ind = dh_ind

        self._inverse_tm = inverse_tikhonov(dm.influence_functions.transformation_matrix.toarray(), rcond=1e-7)

        self._dm_actside = int(round(np.sqrt(len(self._dm.actuators))))
      
        self._timestep = 1 # A unit of time (let's say milliseconds)
      
        self._actstroke = 1 # maximum allowed stroke in microns
      
        # Calculate the diffraction limited science image for Strehl calculations
        diff_lim_img = self._prop(self._demag.forward(self._wavefront)).power

        self._dlmax = np.argmax(diff_lim_img) # Used in Strehl calculation or metric calculation
      
        self._action_space = spaces.Box(-self._actstroke, self._actstroke, shape=(self._dm_actside, self._dm_actside), dtype='float64')
        self._observation_space = spaces.Box(-1, 1, shape=(self._dm_actside, self._dm_actside), dtype='float64')

        self.reset()
    
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
    
    def _turbulence(self, wf):
        '''Propagate a wavefront through the turbulence generator (atmosphere).
        turb --> demagnify
        '''
        wf = self._turbulence_generator.forward(wf)
        wf = self._demag.forward(wf)
        return wf
    
    def _wfs_forward(self):
        '''For now we will hack this to just take the provided wavefront and
        project it on the DM actuator basis
        '''
        if self._wfs == None:
            # First get the phase from the turbulence generator
            phase = self._turbulence_generator.phase_for(self._wavefront.wavelength)
        
            # Then apply the phase addition from the DM
            #measured_phase = phase + 2 * self._wavefront.wavenumber * self._dm.surface

            # To convert this to actuator space, first convert the phase into a corresponding DM surface
            dm_surface = phase / (-2 * self._wavefront.wavenumber)

            coeffs = np.dot(self._inverse_tm, dm_surface)

        return coeffs

    def _forward(self, wf):
        '''Propagate a wavefront through the optics.
        DM --> NCPA --> Coronagraph --> focal plane
        '''
        wf = self._dm.forward(wf)
        wf = self._ncp.forward(wf)
        wf = self._coronagraph.forward(wf)
        wf = self._prop(wf)
        return wf

    def _calc_contrast(self):
        return self._img[self._dh_ind].mean() / self._img[self._dlmax]

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
        action *= 1e-6

        # The action is in a num_acts x num_acts square matrix, which must be flattened out
        action = action.reshape(-1,)

        # Set this to be the DM actuator values
        
        self._dm.actuators = action
      
        # Science optical path
        self._sci_cam.integrate(self._forward(self._turbulence(self._wavefront)), dt=1)
        self._img = self._sci_cam.read_out()

        # Reward is the negative log of contrast
        reward = -np.log10(self._calc_contrast())

        # If the contrast is below 10^-3 we stop
        done = reward <= 0
      
        # Now that the previous phase was corrected by the DM, increment the turbulence generator for next state
        self._turbulence_generator.t += self._timestep

        # Wavefront sensor optical path
        # For now we are going to hack this because WFS is complicated to get to work
        self._wfs_phase_acts = self._wfs_forward()

        wfs_meas_reshaped = np.reshape(self._wfs_phase_acts, (self._dm_actside, self._dm_actside))
        #dm_acts_reshaped = np.reshape(self._dm.actuators.copy(), (self._dm_actside, self._dm_actside))

        # State is the incident wavefront in actuator space measured in units of microns
        self.state = wfs_meas_reshaped / 1e-6


        return self.state, reward, done, {}

    def reset(self, atmseed=None):
        '''Does the following:
        1. Reset dm actuators to defaults.
        2. Reset turbulence generator.
        '''

        np.random.seed(atmseed)

        # Reset the turbulence generator
        self._turbulence_generator.evolve_until(None)
        # Make sure lag is accounted for
        for loop in np.arange(0.001, 0.1, 0.001):
            self._turbulence_generator.evolve_until(loop)

        self._dm.actuators = np.zeros(len(self._dm.actuators)) # Flatten DM
        self._sci_cam.integrate(self._forward(self._turbulence(self._wavefront)), dt=1)
        self._img = self._sci_cam.read_out() # Focal plane image

        self._wfs_phase_acts = self._wfs_forward()

        wfs_meas_reshaped = np.reshape(self._wfs_phase_acts, (self._dm_actside, self._dm_actside))

        # State is the incident wavefront in actuator space measured in units of microns
        self.state = wfs_meas_reshaped / 1e-6
      
        return self.state
        
    def render(self, mode='human'):
        '''Display the state:
        Display reconstructed wavefront and science image.
        '''
        raise NotImplementedError

    def close(self):
        if self.viewer: self.viewer.close()

