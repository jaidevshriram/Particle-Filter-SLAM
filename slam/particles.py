import numpy as np
from numba import jit
import numba
import matplotlib.pyplot as plt
import itertools

from utils.pose import make_pose

def make_pose_batched(x, y, theta):

    N = x.shape[0]
    # og_shape = x.shape
    x = x.flatten()
    y = y.flatten()
    theta = theta.flatten()

    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    
    t = np.array([[x], [y]])

    R = R.reshape((2, 2, N, -1))
    t = t.reshape((2, 1, N, -1))

    R = R.transpose((2, 3, 0, 1))
    t = t.transpose((2, 3, 0, 1))

    # R = R.reshape((N, -1, 2, 2))
    # t = t.reshape((N, -1, 2, 1))
    
    return R, t

# Given a matrix of (N, 3, 3), return x, y, theta
def pose_from_matrix(T):
    return T[..., 0, 2], T[..., 1, 2], np.arctan2(T[..., 1, 0], T[..., 0, 0])

def map_correlation_numba(binary_map, lidar_scan_world):

    # Get the value of the map at all the lidar points from each particle, and perturbation point
    corr = binary_map[lidar_scan_world[..., 0].flatten(), lidar_scan_world[..., 1].flatten()]
    corr = corr.sum()

    return corr

class ParticleFilter:

    def __init__(self, n, pos_variance, angle_variance):

        self.pos_variance = pos_variance
        self.angle_variance = angle_variance
        self.particles = np.zeros((n, 3))

        self.no_noise_particle = np.zeros((1, 3))

        self.weights = np.ones(n, dtype=np.float64) / n
        self.n = n

        self.grid_size = 5
        self.angle_grid = 3

        self.xy_perturb_limit = 0.1
        self.theta_perturb_limit = 2 * np.pi / 180

        self.perturbation = itertools.product(
            np.linspace(-self.xy_perturb_limit, +self.xy_perturb_limit, self.grid_size, endpoint=True),
            np.linspace(-self.xy_perturb_limit, +self.xy_perturb_limit, self.grid_size, endpoint=True),
            np.linspace(-self.theta_perturb_limit, +self.theta_perturb_limit, self.angle_grid, endpoint=True)) # 2, 9
        self.perturbation = np.array(list(self.perturbation)) # grid_size ^ 3, 3

        # self.xy_perturb_limit = 0
        # self.theta_perturb_limit = 0

    def predict(self, u, omega, time_diff, noise=False):

        scale = 1
        if not noise:
            scale = 100

        # Update no noise prediction
        self.no_noise_particle[:, 0] += u * np.cos(self.no_noise_particle[:, 2]) * time_diff
        self.no_noise_particle[:, 1] += u * np.sin(self.no_noise_particle[:, 2]) * time_diff
        self.no_noise_particle[:, 2] += omega * time_diff

        # Update particles with noise
        u = np.repeat(u, self.n)
        omega = np.repeat(omega, self.n)
        
        u += np.random.normal(0, self.pos_variance / scale, size=u.shape)
        omega += np.random.normal(0, self.angle_variance / scale, size=omega.shape)

        self.particles[:, 0] += u * np.cos(self.particles[:, 2]) * time_diff
        self.particles[:, 1] += u * np.sin(self.particles[:, 2]) * time_diff
        self.particles[:, 2] += omega * time_diff

    def map_correlation(self, binary_map, lidar_scan_world):

        # Create an empty matrix to store correlation values for each particle and perturbation point
        corr = np.zeros((lidar_scan_world.shape[0], self.grid_size ** 2 * self.angle_grid)) # N, grid_size ** 3 

        # temp_map = np.ones((binary_map.shape[0], binary_map.shape[1], 3)) * 0.5

        # Get the value of the map at all the lidar points from each particle, and perturbation point
        corr = binary_map[lidar_scan_world[..., 0].flatten(), lidar_scan_world[..., 1].flatten()]

        # temp_map[lidar_scan_world[..., 0].flatten(), lidar_scan_world[..., 1].flatten(), :] = 1
        # # plt.imshow(temp_map[200:400 , 200:400, :])
        # plt.imshow(binary_map)
        # plt.scatter(lidar_scan_world[0, 0, :, 1].flatten(), lidar_scan_world[0, 0,:, 0].flatten(), c='r', s=1)
        # plt.show()

        # Reshape the correlation matrix to N, grid_size ** 3, lidar_len and sum along the last axis
        corr = corr.reshape(lidar_scan_world.shape[0], self.grid_size ** 2 * self.angle_grid,  lidar_scan_world.shape[2]).sum(axis=-1) 

        # print("Corr values", corr[0])

        # Get the maximum correlation value and its index for each particle
        max_correspondence = corr.max(axis=-1) # N
        max_correspondence_idx = np.argmax(corr, axis=-1) # N

        return max_correspondence, max_correspondence_idx
    
    # @staticmethod
    # @jit
    def evaluate_mapp_corr_numba(particles, xy_sensor, binary_map, perturbation):

        updated_particles = np.zeros((100, 3), dtype=np.float64)
        corrs = np.zeros(100, dtype=np.float64)

        for particle_idx in range(100):

            corrs_max = -1
            best_perturbed = None

            for pertubation_idx in range(perturbation.shape[0]):
                
                perturbed_particle = particles[particle_idx] + perturbation[pertubation_idx]

                x = perturbed_particle[0]
                y = perturbed_particle[1]
                theta = perturbed_particle[2]

                R_perturb = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
                    
                t_perturb = np.array([[x], [y]])

                xy_world = R_perturb @ xy_sensor.T + t_perturb[:, np.newaxis]

                # xy_grid = map_obj.real2grid(xy_world[..., 0], xy_world[..., 1])
                xy_grid = np.zeros_like(xy_world)
                xy_grid[..., 0] = (xy_world[..., 0] - (-50)) / 0.1
                xy_grid[..., 1] = (xy_world[..., 1] - (-50)) / 0.1

                corr = map_correlation_numba(binary_map, xy_grid)

                if corr > corrs_max:
                    corrs_max = corr
                    best_perturbed = perturbed_particle

            updated_particles[particle_idx] = best_perturbed
            corrs[particle_idx] = corrs_max

        return 1 + corrs, updated_particles

        # # Repeat the lidar scan for each particle
        # xy_sensor = np.repeat(xy_sensor[np.newaxis, ...], self.n, axis=0) # N, lidar_len, 2

        # # Repeat the lidar points for each perturbation
        # xy_sensor = np.repeat(xy_sensor[:, np.newaxis, ...], self.grid_size ** 2 * self.angle_grid, axis=1) # N, grid_size ^ 3, lidar_len, 2

        # # Combine the perturbation with the particle
        # perturbed_particles = np.repeat(self.particles[:, np.newaxis, ...], self.grid_size ** 2 * self.angle_grid, axis=1) + perturbation[None, ...] # N, grid_size ^ 3, 3

        # # Compute transformation matrix for each perturbation
        # R_perturb, t_perturb = make_pose_batched(perturbed_particles[..., 0], perturbed_particles[..., 1], perturbed_particles[..., 2]) # N, grid_size ^ 3, 2, 2

        # # Repeat the transformation matrix for each lidar point
        # R_perturb = np.repeat(R_perturb[:, :, np.newaxis, ...], lidar_len, axis=2) # N, grid_size ^ 3, lidar_len, 2, 2
        # t_perturb = np.repeat(t_perturb[:, :, np.newaxis, ...], lidar_len, axis=2) # N, grid_size ^ 3, lidar_len, 2, 1

        # # Apply transformation to lidar points
        # xy_world = R_perturb @ xy_sensor[..., None] + t_perturb # N, grid_size ^ 3, lidar_len, 2, 1
        # xy_world = xy_world.squeeze(-1) # N, grid_size ^ 3, lidar_len, 2

        # # Get the map cells corresponding to the lidar points
        # xy_grid = map_obj.real2grid(xy_world[..., 0], xy_world[..., 1]).transpose(0, 2, 3, 1) # N, grid_size ^ 3, lidar_len, 2

        # max_correlation, max_correlation_idx = self.map_correlation(map_obj.binary_map, xy_grid)

        # # print(perturbed_particles[0, max_correlation_idx[0], :])

        # return 1 + max_correlation, perturbed_particles[np.arange(self.n), max_correlation_idx, :]

    # z = Lidar scan at time instance, m = map of world
    # @jit
    def evaluate_map_corr(self, xy_sensor, map_obj):

        lidar_len = xy_sensor.shape[0]

        # Repeat the lidar scan for each particle
        xy_sensor = np.repeat(xy_sensor[np.newaxis, ...], self.n, axis=0) # N, lidar_len, 2

        # Repeat the lidar points for each perturbation
        xy_sensor = np.repeat(xy_sensor[:, np.newaxis, ...], self.grid_size ** 2 * self.angle_grid, axis=1) # N, grid_size ^ 3, lidar_len, 2

        # Generate the perturbed grid around each particle
        # perturbation = itertools.product(
        #     np.linspace(-self.xy_perturb_limit, +self.xy_perturb_limit, self.grid_size, endpoint=True),
        #     np.linspace(-self.xy_perturb_limit, +self.xy_perturb_limit, self.grid_size, endpoint=True),
        #     np.linspace(-self.theta_perturb_limit, +self.theta_perturb_limit, self.angle_grid, endpoint=True)) # 2, 9
        # perturbation = np.array(list(perturbation)) # grid_size ^ 3, 3

        # Combine the perturbation with the particle
        perturbed_particles = np.repeat(self.particles[:, np.newaxis, ...], self.grid_size ** 2 * self.angle_grid, axis=1) + self.perturbation[None, ...] # N, grid_size ^ 3, 3

        # Compute transformation matrix for each perturbation
        R_perturb, t_perturb = make_pose_batched(perturbed_particles[..., 0], perturbed_particles[..., 1], perturbed_particles[..., 2]) # N, grid_size ^ 3, 2, 2

        # Repeat the transformation matrix for each lidar point
        R_perturb = np.repeat(R_perturb[:, :, np.newaxis, ...], lidar_len, axis=2) # N, grid_size ^ 3, lidar_len, 2, 2
        t_perturb = np.repeat(t_perturb[:, :, np.newaxis, ...], lidar_len, axis=2) # N, grid_size ^ 3, lidar_len, 2, 1

        # Apply transformation to lidar points
        xy_world = R_perturb @ xy_sensor[..., None] + t_perturb # N, grid_size ^ 3, lidar_len, 2, 1
        xy_world = xy_world.squeeze(-1) # N, grid_size ^ 3, lidar_len, 2

        # Get the map cells corresponding to the lidar points
        xy_grid = map_obj.real2grid(xy_world[..., 0], xy_world[..., 1]).transpose(0, 2, 3, 1) # N, grid_size ^ 3, lidar_len, 2

        max_correlation, max_correlation_idx = self.map_correlation(map_obj.binary_map, xy_grid)

        # print(perturbed_particles[0, max_correlation_idx[0], :])

        return 1 + max_correlation, perturbed_particles[np.arange(self.n), max_correlation_idx, :]

    # @jit
    def update(self, z, map_object):

        n_eff = 1. / np.sum(np.square(self.weights))

        if n_eff <  self.n / 3:
            self.resample()

        corr_values, updated_particles = self.evaluate_map_corr(z, map_object)
        # corr_values, updated_particles = self.evaluate_mapp_corr_numba(self.particles, z, map_object.binary_map, self.perturbation)

        self.weights *= corr_values
        self.weights /= np.sum(self.weights)

        self.particles = updated_particles

        return self.particles[np.argmax(self.weights)]

    def resample(self):

        self.particles = self.particles[np.random.choice(len(self.particles), len(self.particles), p=self.weights), :]
        self.weights = np.ones(len(self.particles)) / len(self.particles)

    def get_best_particle(self):

        return self.particles[np.argmax(self.weights)]
    
    def get_zero_particle(self):

        return self.no_noise_particle[0]
    
    def compare_particles(self):

        zero_particle = self.get_zero_particle()
        best_particle = self.get_best_particle()

        # print("Zero particle: ", zero_particle)
        # print("Best particle: ", best_particle, " with weight: ", np.max(self.weights))        
        # # print("Difference: ", zero_particle - best_particle)
        # # print("Weights: ", self.weights)

        # plt.plot(self.weights)
        # plt.show()

    def get_all_particles(self):

        return self.particles