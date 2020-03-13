import numpy as np
import torch


class HamiltonianNet:

    def __init__(self):
        self.u = 1
        self.nx = 800
        self.dr = 0.01
        self.beta = 0.08
        self.h = 0.01

    def space_points_gen(self):
       self.x_space = [self.dr*i for i in range(1, self.nx+1)]

    def potential(self, x):
       return -1 / x

    def divider(self):
        self.xlow = self.x_space[0]
        self.xhigh = self.x_space[-1]

    def psi(self, x,  net):
        return np.exp(-self.beta * x ** 2) * net(torch.as_tensor([[x]]))

    def hamiltonian_psi(self, x, net):  #
        lap_psi = (self.psi(x + self.h, net) - 2 * self.psi(x, net) + self.psi(x - self.h, net))/self.h**2
        h_psi = (-0.5 * lap_psi / self.u) + (self.potential(x) * self.psi(x, net))
        return h_psi

    def num_den(self, x, net):
        num = self.psi(x, net) * self.hamiltonian_psi (x, net)
        den = (self.psi(x, net)) ** 2
        return num, den

    def trapezoidal(self, net):
        self.space_points_gen()
        num_integral = []
        den_integral = []
        for x in self.x_space[1:len(self.x_space) - 1]:
            numerator, denominator = self.num_den(x, net)
            num_integral.append(self.dr * numerator)
            den_integral.append(self.dr * denominator)

        def trap_sub(n):
            numerator, denominator = self.num_den(self.x_space[n], net)
            num_integral.append (.5 * self.dr * numerator)
            den_integral.append (.5 * self.dr * denominator)
            return None

        trap_sub(0)
        trap_sub(len(self.x_space) - 1)
        return sum(numerator), sum(denominator)

    def epsilon(self, net):
        num_int, den_int = self.trapezoidal(net)
        return num_int / den_int

    def error_function(self, net):
        loss_sum = 0.
        num_int, den_int = self.trapezoidal(net)
        epsilon = num_int / den_int
        print("epsilon:", epsilon.detach().numpy()[0])
        for x in self.x_space:
            err_eq = (self.hamiltonian_psi (x, net)) - (epsilon * self.psi (x, net))
            loss_sum += err_eq * err_eq
        return loss_sum / (len(self.x_space) * den_int)
