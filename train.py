# not complete need to write the loss function
from hamiltonian_net.methods import *
from hamiltonian_net.neural_net import *
import torch.optim as optim

net = Net()
x_state = space_points_gen()
optimizer = optim.SGD(net.parameters(), lr=0.1)

for i in range(400):
    for x in x_state:
        inter = []
        inter.append(x)
        intera = []
        intera.append(inter)
        x = torch.as_tensor(intera).unsqueeze(0)
        optimizer.zero_grad()
        out = net(x)
        loss = error_function(beta, out)
        print(loss)
        loss.backward()
        optimizer.step()



