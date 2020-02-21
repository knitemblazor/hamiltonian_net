# not complete need to write the loss function
from methods import *

net = Net()
input = torch.tensor([[8.5]])
out = net(input)
print(epsilon(beta, out), "done")
