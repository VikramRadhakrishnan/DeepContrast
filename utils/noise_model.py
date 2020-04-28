# Implements Ornstein-Uhlenbeck noise

import numpy as np
import copy
import random

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class GaussianNoise:
    """Gaussian noise."""
    
    def __init__(self, size, seed, mu=0, sigma=1):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.seed = random.seed(seed)
        
    def reset(self):
        pass
    
    def sample(self):
        """Return Gaussian perturbations in the action space."""
        noise = np.random.normal(0, self.sigma, self.size)
        return noise
