#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm.auto import tqdm


# In[2]:


from utils.load_data import load_data
from utils.pose import make_pose
from slam.mapper import Map
from slam.sensor import Lidar, Encoder, IMU, RGBDSensor
from slam.particles import ParticleFilter

import argparse


args = argparse.ArgumentParser()
args.add_argument("--scene", type=int, default=20)
args.add_argument("--num_particles", type=int, default=100)
args = args.parse_args()

SCENE_NUM = args.scene


# In[5]:


dataset = load_data(SCENE_NUM, "./data/")


# In[6]:


map_obj = Map(-50, 50, -50, 50, 0.1)
lidar_obj = Lidar(dataset['lidar_angle_min'], dataset['lidar_angle_max'], dataset['lidar_angle_increment'], dataset['lidar_range_min'], dataset['lidar_range_max']-5, dataset['lidar_ranges'], dataset['lidar_stamps'])
encoder_obj = Encoder(dataset['encoder_counts'], dataset['encoder_stamps'])
imu_obj = IMU(dataset['imu_angular_velocity'], dataset['imu_linear_acceleration'], dataset['imu_stamps'])
rgbd_obj = RGBDSensor("./data/", SCENE_NUM, dataset['rgb_stamps'], dataset['disp_stamps'])
particle_filter = ParticleFilter(args.num_particles, 0.01, 0.01)


# # Run Through the Encoder Motions

# In[7]:


def save_state(i):

    np.save(f"./saves/{i}_{SCENE_NUM}.npy", {
        'map': map_obj,
        'lidar': lidar_obj,
        'encoder': encoder_obj,
        'imu': imu_obj,
        'rgbd': rgbd_obj,
        'particle_filter': particle_filter,
        'robot_state': robot_state,
        'prev_ts': prev_ts,
    }, allow_pickle=True)

robot_state = [[
    0,
    0,
    0
]]

calculated_omegas = []
omegas_imu = []

axle_length = 0.31 # m

prev_ts = datetime.datetime.fromtimestamp(encoder_obj.ts[0])
start = 0

for i in tqdm(range(start, len(encoder_obj))):

    # Get current state x_t
    curr_state = robot_state[-1]

    # Get encoder data u_t
    dr, dl, ts = encoder_obj[i]

    # Get lidar data z_t+1
    ranges = lidar_obj.get_data_at(ts)
    xy_sensor = lidar_obj.convert_to_cartesian(ranges)

    # Convert difference in time from unix to seconds (t+1 - t)
    time_t = datetime.datetime.fromtimestamp(ts)
    time_diff = (time_t - prev_ts).total_seconds()

    if np.isclose(time_diff, 0):
        prev_ts = time_t
        continue

    # Calculate new state
    vl = dl / time_diff
    vr = dr / time_diff
    vt = (vr + vl) / 2

    calculated_omegas.append((vr - vl) / axle_length)

    omegas = imu_obj.get_angular_at(ts)
    omega_t = omegas[2]
    omegas_imu.append(omega_t)

    if (i % 5 == 0 and i > 1) or i > 2000:
        # Predict step of particle filter \mu_t+1 = f(\mu_t, u_t)
        particle_filter.predict(vt, omega_t, time_diff, noise=True)
        # Update step of particle filter \mu_t+1 = g(\mu_t+1, z_t+1)
        new_state = particle_filter.update(xy_sensor, map_obj)
    else:  
        # Predict step of particle filter \mu_t+1 = f(\mu_t, u_t)
        particle_filter.predict(vt, omega_t, time_diff, noise=True)
        new_state = particle_filter.get_best_particle()

    # Lidar projected using updates pose
    R_body_wrt_world, t_body_wrt_world = make_pose(new_state[0], new_state[1], new_state[2])
    lidar_points_pose_wrt_world = np.eye(3)
    lidar_points_pose_wrt_world[:2, :2] = R_body_wrt_world
    lidar_points_pose_wrt_world[:2, 2] = t_body_wrt_world[..., 0]
    xy_world = (R_body_wrt_world @ xy_sensor.T + t_body_wrt_world).T

    # Get RGBD at this time
    points, rgb = rgbd_obj.get_at_time(ts)
    points_world = (R_body_wrt_world @ points[..., :2].T + t_body_wrt_world).T

    # Add to map
    map_obj.add_line(new_state[:2], xy_world, state='free')
    map_obj.add_occupied(xy_world[:, 0], xy_world[:, 1])

    # Add trajectories to map
    map_obj.add_trajectory(new_state[0], new_state[1], new_state[2])
    map_obj.add_particles(particle_filter.get_all_particles())
    
    # Add texture
    map_obj.add_texture(points_world, rgb)

    robot_state.append(new_state)

    prev_ts = time_t

    if i % 250 == 0:
        # map_obj.show(title=i, particle=False)
        # map_obj.show_textured(title=i)

        map_obj.save(i)
        # save_state(i)