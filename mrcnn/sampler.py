import numpy as np


class ParticleSampler():
    def __init__(self, particle_count=100, initial_particles=None):
        self.particle_count = particle_count
        self.initial_particles = initial_particles

    def sample(self):
        # np.random.normal(self.initial_particles, np.array([1,]), 1000)
        print('Not yet implemented')
