#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 22:06:27 2018

@author: vikram
"""
import gym
from hcipy import *
from hcipy.atmosphere import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os, glob
import time
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class HCI_TestBench(gym.Env):
    
    def __init__(self, turbulence_generator, dm, wfs_optics, prop, coronagraph, wfs_cam, sci_cam):
        '''Variables used to characterize testbench
        turbulence_generator: optical component to add phase and amplitude aberrations
        dm: deformable mirror component
        wfs_optics: optical components placed in the wavefront sensor optical path
        prop: propagator from pupil plane to focal plane
        coronagraph: coronagraph optic
        wfs_cam: wavefront sensor camera detector
        sci_cam: science camera detector
        '''
        self._turbulence_generator = turbulence_generator
        self._dm = dm
        self._wfs_optics = wfs_optics
        self._prop = prop
        self._coronagraph = coronagraph
        self._wfs_cam = wfs_cam
        self._sci_cam = sci_cam
        
        
        self._timestep = 0
        
        self._act_stroke = 10e-6 #10 micron stroke is the maximum allowed
        
        self._action_space = spaces.Box(-1., 1., shape=(len(dm.influence_functions),), dtype='float32')
        self._observation_space = spaces.Dict(dict(
            measured_slopes=spaces.Box(-np.inf, np.inf, shape=obs['measured_slopes'].shape, dtype='float32'),
            image_pixels=spaces.Box(0, 255, shape=obs['image_pixels'].shape, dtype='float32')
        ))

        self.seed()
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return self._observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def atms_forward(self, wavefront):
        wf = wavefront.copy()
        self.turbulence_generator.evolve_until(self.timestep)
        wf = self.turbulence_generator.forward(wf)
        return wf
    
    def wfs_forward(self, wavefront):
        wf = wavefront.copy()
        wf = self.dm.forward(wf)
        wf = self.wfs_optics.forward(wf)
        return wf
    
    def wfs_backward(self, wavefront):
        wf = wavefront.copy()
        wf = self.wfs_optics.backward(wf)
        wf = self.dm.backward(wf)
        return wf
    
    def forward(self, wavefront):
        wf = wavefront.copy()
        wf = self.dm.forward(wf)
        wf = self.coronagraph.forward(wf)
        wf = self.prop(wf)
        return wf
    
    def backward(self, wavefront):
        wf = wavefront.copy()
        wf = self.coronagraph.backward(wf)
        wf = self.dm.backward(wf)
        return wf
    
    def wfs_readout(self, wavefront, dt=1, weight=1):
        wf = wavefront.copy()
        self.wfs_cam.integrate(wf, dt, weight)
        wfs_img = self.wfs_cam.read_out()
        wfs_img = wf.power
        return wfs_img
    
    def sci_cam_readout(self, wavefront, dt=1, weight=1):
        wf = wavefront.copy()
        self.sci_cam.integrate(wf, dt, weight)
        sci_img = self.sci_cam.read_out()
        sci_img = wf.power
        return sci_img

    def step(self, wavefront):
        wfatms = atms_forward(wavefront)
        
        # Wavefront sensor optical path
        wf_wfs = wfatms.copy()
        wf_wfs = wfs_forward(wf_wfs)
        wfs_slopes = 
        
        self.timestep += 1
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

