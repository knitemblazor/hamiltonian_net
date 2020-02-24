from methods import *
from neural_net import *
import torch.optim as optim

net = Net()
x_state = space_points_gen()
optimizer = optim.Adam(net.parameters(), lr=0.08)

for i in range(200):
    optimizer.zero_grad()
    loss = error_function(beta, net)
    print("loss:", loss.detach().numpy()[0][0])
    loss.backward()
    optimizer.step()
