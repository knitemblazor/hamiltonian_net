from method_mod import *
import torch


def num_den(x, beta_, net):
    num = psi(x, beta_, net) * hamiltonian_psi(x, beta_, net)
    den = (psi(x, beta_, net) * net(torch.as_tensor([[x]]))) ** 2
    return num, den


def trapezoidal(beta_, net):
    x_space = space_points_gen()
    dr, xhigh, xlow = divider()
    num_integral = []
    den_integral = []

    for x in x_space[1:len(x_space)-1]:
        numerator, denominator = num_den(x, beta_, net)
        num_integral.append(dr*numerator)
        den_integral.append(dr*denominator)

    def trap_sub(n):
        numerator, denominator = num_den(x_space[n], beta_, net)
        num_integral.append(.5*dr*numerator)
        den_integral.append(.5*dr*denominator)
        return None
    trap_sub(0)
    trap_sub(len(x_space)-1)
    return sum(numerator), sum(denominator)


def epsilon(beta_, net):
    num_int, den_int = trapezoidal(beta_, net)
    return sum(num_int)/sum(den_int)


def error_function(beta_, net):
    loss_sum = 0.
    _, den_int = trapezoidal(beta_, net)
    x_space = space_points_gen()
    print("epsilon", epsilon(beta_, net))
    for x in x_space:
        err_eq = hamiltonian_psi(x, beta_, net) - epsilon(beta_, net) * psi(x, beta_, net)
        loss_sum += err_eq * err_eq
    return loss_sum / (nx * den_int)
