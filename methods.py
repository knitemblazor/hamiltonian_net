from method_mod import *
from neural_net import *
import torch


def num_den(x, beta_, net_out):
    num = psi(x, beta_, net_out) * hamiltonian_psi(x, beta_, net_out)
    den = (psi(x, beta_, net_out) * net_out) ** 2
    return num, den


def trapezoidal(beta_, net_out):
    x_space = space_points_gen()
    dr, xhigh, xlow = divider()
    num_integral = []
    den_integral = []
    for x in x_space[1:len(x_space)-1]:
        numerator, denominator = num_den(x, beta_, net_out)
        num_integral.append(dr*numerator)
        den_integral.append(dr*denominator)

    def trap_sub(n):
        numerator, denominator = num_den(x_space[n], beta_, net_out)
        num_integral.append(.5*dr*numerator)
        den_integral.append(.5*dr*denominator)
        return None
    trap_sub(0)
    trap_sub(len(x_space)-1)
    return sum(numerator)/sum(denominator)


def epsilon(beta_, net_out):
    return trapezoidal(beta_, net_out)


def error_function():
    return None


