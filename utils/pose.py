import numpy as np

def make_pose(x, y, theta):

    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    
    t = np.array([[x], [y]])
    
    return R, t