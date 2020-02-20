import numpy as np


def space_points_gen():
    nx = 400
    dr = 1. / nx
    x_space = np.linspace(dr, 16., nx)
    return x_space


def potential(x):
    return -1. / x