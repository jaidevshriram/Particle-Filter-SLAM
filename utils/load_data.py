import numpy as np

def load_data(number, base_path = "../data/"):
  dataset = number
  
  with np.load(f"{base_path}/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load(f"{base_path}/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load(f"{base_path}/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load(f"{base_path}/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

  out = {}

  out["encoder_counts"] = encoder_counts
  out["encoder_stamps"] = encoder_stamps
  out["lidar_angle_min"] = lidar_angle_min
  out["lidar_angle_max"] = lidar_angle_max
  out["lidar_angle_increment"] = lidar_angle_increment
  out["lidar_range_min"] = lidar_range_min
  out["lidar_range_max"] = lidar_range_max
  out["lidar_ranges"] = lidar_ranges
  out["lidar_stamps"] = lidar_stamps
  out["imu_angular_velocity"] = imu_angular_velocity
  out["imu_linear_acceleration"] = imu_linear_acceleration
  out["imu_stamps"] = imu_stamps
  out["disp_stamps"] = disp_stamps
  out["rgb_stamps"] = rgb_stamps

  return out