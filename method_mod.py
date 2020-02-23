import numpy as np
from functools import reduce
from autograd import grad

u = 1
nx = 400
lmb = 0.08
beta = 0.001


def space_points_gen():
    dr = 1. / nx
    x_space = np.linspace(dr, 16., nx)
    return x_space


def potential(x):
    return -1. / x


def divider():
    dr = 1. / nx
    xlow = dr
    xhigh = (nx - 1) * dr
    return dr, xlow, xhigh


def psi(x, beta_, net_out):
    return np.exp(-beta_ * x ** 2) * net_out


def hamiltonian_psi(x, beta_, net_out):  #
    h = 0.00003034
    lap_psi = (psi(x+h, beta_, net_out) - 2*psi(x, beta_, net_out) + psi(x-h, beta_, net_out))/h**2
    h_psi = (-0.5 * lap_psi / u) + potential(x) * psi(x, beta_, net_out)
    return h_psi
