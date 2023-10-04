import os
import cv2
from scipy.signal import butter, lfilter, filtfilt
import numpy as np

class Lidar:

    """
    Sets up a Lidar sensor with the following parameters:
    angle_min: minimum angle of the lidar sensor
    angle_max: maximum angle of the lidar sensor
    angle_increment: angular distance between measurements
    range_min: minimum range value
    range_max: maximum range value

    """
    def __init__(self, angle_min, angle_max, angle_increment, range_min, range_max, ranges, ts):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max

        self.angles = np.linspace(angle_min, angle_max, int((angle_max - angle_min) / angle_increment) + 1)

        self.ts = ts
        self.ranges = ranges

    def __len__(self):
        return len(self.ts)

    def _add_new_measurement(self, ranges, ts):
        self.ts.append(ts)
        self.ranges.append(ranges)

    def convert_to_cartesian(self, ranges):

        indValid = np.logical_and((ranges < self.range_max), (ranges > self.range_min))
        ranges = ranges[indValid]
        angles = self.angles[indValid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    def get_data_at(self, t):

        idx = np.argmin(np.abs(np.array(self.ts) - t))
        return self.ranges[:, idx]

class Encoder:

    """
    Represents an encoder, which counts the rotations of the four wheels at 40Hz. The encoder counter is reset after each reading. 
    
    """

    def __init__(self, data, ts):

        self.data = data
        self.ts = ts
        self.tic_dist = 0.0022

    def __len__(self):
        return len(self.ts)

    def get_data_at(self, t):
            
        idx = np.argmin(np.abs(np.array(self.ts) - t))
        return self.data[:, idx]
    
    def _get_wheel_dist(self, idx):

        data = self.data[:, idx]
        fr, fl, rr, rl = data * self.tic_dist

        return (fr + rr) / 2, (fl + rl) / 2
    
    def __getitem__(self, idx):

        dl, dr = self._get_wheel_dist(idx)

        return dl, dr, self.ts[idx]

class IMU:

    """
    Represents an IMU sensor, which measures the angular velocity and linear acceleration of the robot at 40Hz. 

    """
    
    def __init__(self, angular_velocity, linear_acceleration, ts):

        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration
        self.ts = ts

        fs = 100.0
        cutoff = 10.0
        order = 5

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Apply the filter to your numpy array using the lfilter function
        self.angular_velocity = filtfilt(b, a, self.angular_velocity)
        self.linear_acceleration = filtfilt(b, a, self.linear_acceleration)

    def __len__(self):
        return len(self.ts)

    def get_data_at(self, t):

        idx = np.argmin(np.abs(np.array(self.ts) - t))
        return self.angular_velocity[:, idx], self.linear_acceleration[:, idx]
    
    def get_angular_at(self, t):

        idx = np.argmin(np.abs(np.array(self.ts) - t))
        return self.angular_velocity[:, idx]

    def __getitem__(self, idx):

        return self.get_data_at(self.ts[idx])
    
class RGBDSensor:

    def __init__(self, path, scene, rgb_ts, disp_ts):

        self.scene = scene
        self.path = path
        self.rgb_ts = rgb_ts
        self.disp_ts = disp_ts

    def get_at_time(self, t):

        idx = np.argmin(np.abs(np.array(self.rgb_ts) - t)) + 1

        rgb = cv2.imread(os.path.join(f"{self.path}", f"RGB{self.scene}", f"rgb{self.scene}_{idx}.png"))[..., ::-1]
        disparity = cv2.imread(os.path.join(f"{self.path}", f"Disparity{self.scene}", f"disparity{self.scene}_{idx}.png"), cv2.IMREAD_UNCHANGED)

        disparity = disparity.astype(np.float32)

        dd = (-0.00304 * disparity + 3.31)
        z = 1.03 / dd

        # Get 3D points 
        # calculate u and v coordinates 
        v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
        #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))
        
        # get 3D coordinates 
        fx = 585.05108211
        fy = 585.05108211
        cx = 315.83800193
        cy = 242.94140713
        x = (u-cx) / fx * z
        y = (v-cy) / fy * z

        rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
        rgbv = np.round((v * 526.37 + 16662.0)/fy)
        # valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
        valid = (rgbu>= 0)&(rgbu < 640)&(rgbv>=0)&(rgbv<480)

        points = np.stack([x,y,z], axis=2)
        t_optical_to_world = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

        # print(points.shape)

        points.T[0, ...] -= 0.18 # x offset
        points.T[1, ...] -= 0.005 # y offset
        points.T[2, ...] -= 0.36 # z offset

        points = t_optical_to_world @ points.reshape(-1, 3).T
        rot_sensor_2_robot = get_pose(0.18, 0.005, 0.36)

        points = rot_sensor_2_robot.T @ points
        points = points.T.reshape((480, 640, 3))

        valid = valid & (points[..., 2] < 0.1)
        points = points[valid]
        rgb = rgb[rgbv[valid].astype(np.int), rgbu[valid].astype(np.int), :]/255.0

        return points, rgb

def get_pose(roll, pitch, yaw):

    def rot_x(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def rot_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    def rot_z(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    R = np.matmul(rot_z(yaw), np.matmul(rot_y(pitch), rot_x(roll)))

    return R

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)