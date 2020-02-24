import numpy as np
import torch

u = 1
nx = 200
lmb = 0.08
beta = 0.001


def space_points_gen():
    dr = 1. / nx
    x_space = np.linspace(dr, 1., nx)
    return x_space


def potential(x):
    return -1. / x


def divider():
    dr = 1. / nx
    xlow = dr
    xhigh = (nx - 1) * dr
    return dr, xlow, xhigh


def psi(x, beta_, net):
    return np.exp(-beta_ * x ** 2) * net(torch.as_tensor([[x]]))


def hamiltonian_psi(x, beta_, net):  #
    h = 0.002
    lap_psi = (psi(x+h, beta_, net) - 2*psi(x, beta_, net) + psi(x-h, beta_, net))/h**2
    h_psi = (-0.5 * lap_psi / u) + potential(x) * psi(x, beta_, net)
    return h_psi
