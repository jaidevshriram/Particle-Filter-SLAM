import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line as line_maker

OCCUPIED = [0, 0, 0]
FREE = [1, 1, 1]

class Map:
    """
        This class represents the map of the environment. It is a 2D grid of cells.
        Each cell is either occupied or free or unknown.
    """

    def __init__(self, xmin, xmax, ymin, ymax, resolution=0.05, scene=20):
        self.width = int(np.ceil((xmax - xmin) / resolution + 1))
        self.height = int(np.ceil((ymax - ymin) / resolution + 1))
        self.resolution = resolution
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.cells = np.zeros((self.height, self.width), dtype=np.float64)
        self.texture_map = np.zeros((self.height, self.width, 3), dtype=np.float64)
        self.color_n  = np.ones((self.height, self.width, 3), dtype=np.float64)

        self.logs_odd_ratio = np.log(4)
        self.logs_odd_limit_max = 30.0
        self.logs_odd_limit_min = -30.0
        self.buffer = 10

        self.trajectory = []
        self.particle_trajectories = []

        self.scene_num = scene

    @property
    def binary_map(self):

        map = np.zeros_like(self.cells)

        # Convert log odds to probability
        probability_map = np.exp(self.cells) / (1 + np.exp(self.cells))

        # Occupancy map
        # map[self.cells == 0] = 0
        map[probability_map > 0.5] = 1
        map[probability_map < 0.4] = 0

        return map

    def add_trajectory(self, x, y, theta):
        self.trajectory.append([x, y, theta])

    def add_particles(self, particles):
        self.particle_trajectories.append(particles.copy())

    def add_occupied(self, x, y):

        x = np.int16((x - self.xmin) / self.resolution)
        y = np.int16((y - self.ymin) / self.resolution)

        self.cells[x, y] += self.logs_odd_ratio
        self.cells = np.clip(self.cells, self.logs_odd_limit_min, self.logs_odd_limit_max)

    def add_line(self, start, end, state='free'):

        start = np.int16((start - np.array([self.xmin, self.ymin])[None, :]) / self.resolution)

        end = np.int16((end - np.array([self.xmin, self.ymin])[None, :]) / self.resolution)
        
        if state == 'free':
            color = FREE
        elif state == 'occupied':
            color = OCCUPIED

        for i in range(end.shape[0]):

            line = line_maker(*start[0], *(end[i, :]))

            self.cells[line[0][:-1], line[1][:-1]] -= self.logs_odd_ratio * 0.5

        self.cells = np.clip(self.cells, self.logs_odd_limit_min, self.logs_odd_limit_max)

    def add_texture(self, points, rgb):

        x = np.int16((points[:, 0] - self.xmin) / self.resolution)
        y = np.int16((points[:, 1] - self.ymin) / self.resolution)

        self.texture_map[x, y, :] = (self.texture_map[x, y, :] * self.color_n[x, y, :] + rgb) / (self.color_n[x, y, :] + 1)
        self.color_n[x, y, :] += 1

    def show_textured(self, title="", save=False):

        first_row, final_row, first_col, final_col = self.find_horizontal_bounds()
        map = np.zeros_like(self.cells)

        # Convert log odds to probability
        probability_map = np.exp(self.cells) / (1 + np.exp(self.cells))
        map[probability_map > 0.5] = 1

        # Occupancy map
        self.texture_map[probability_map > 0.5] = 0

        plt.imshow(self.texture_map[first_row - self.buffer: final_row + self.buffer, first_col - self.buffer:final_col + self.buffer, :])
        plt.imshow(map[first_row - self.buffer: final_row + self.buffer, first_col - self.buffer:final_col + self.buffer], cmap='coolwarm', alpha=map
[first_row - self.buffer: final_row + self.buffer, first_col - self.buffer:final_col + self.buffer])
        
        # Trajectory
        trajectory = np.array(self.trajectory)
        trajectory = self.real2grid(trajectory[:, 0], trajectory[:, 1])
        
        plt.plot(trajectory[:, 1] - first_col + self.buffer, trajectory[:, 0] - first_row + self.buffer, 'r')

        plt.title(title)

        if save:
            plt.savefig(f'./outputs_{self.scene_num}/textured_{title}.png')
        else:
            plt.show()

    def show(self, title="", particle=False, save=False):

        map = np.zeros_like(self.cells)

        # Convert log odds to probability
        probability_map = np.exp(self.cells) / (1 + np.exp(self.cells))

        # Occupancy map
        map[self.cells == 0] = 0.5
        map[probability_map > 0.5] = 0
        map[probability_map < 0.5] = 1

        first_row, final_row, first_col, final_col = self.find_horizontal_bounds()

        # plt.imshow(map[first_row - self.buffer: final_row + self.buffer, first_col - self.buffer:final_col + self.buffer], cmap='gray')

        # Trajectory
        trajectory = np.array(self.trajectory)
        trajectory = self.real2grid(trajectory[:, 0], trajectory[:, 1])

        if particle:

            particle_trajectories = np.array(self.particle_trajectories)

            for i in range(len(particle_trajectories[0])):

                particles = particle_trajectories[:, i, :]
                particles = self.real2grid(particles[:, 0], particles[:, 1])

                plt.plot(particles[:, 1] - first_col + self.buffer, particles[:, 0] - first_row + self.buffer)

        plt.plot(trajectory[:, 1] - first_col + self.buffer, trajectory[:, 0] - first_row + self.buffer, color='r')
        plt.title(title)

        if save:
            plt.savefig(f'./outputs_{self.scene_num}/{title}.png')
        else:
            plt.show()

    def real2grid(self, x, y):

        x = np.int16((x - self.xmin) / self.resolution)
        y = np.int16((y - self.ymin) / self.resolution)

        return np.stack((x, y), axis=1)
    
    def reset(self):

        self.cells = np.zeros((self.height, self.width), dtype=np.float64)
        self.trajectory = []

    def find_horizontal_bounds(self):

        # Find the first column that is not empty
        binary_map = self.binary_map
        positive = np.argwhere(binary_map > 0)

        (ystart, xstart), (ystop, xstop) = positive.min(0), positive.max(0) + 1 

        return ystart, ystop, xstart, xstop
        # return 0, self.height, 0, self.width

    def save(self, idx):

        self.show(idx, save=True)
        self.show_textured(idx, save=True)

