import torch
import torch.nn as nn
import torch.optim as optim
from autograd import grad


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 8, bias=False)
        self.fc2 = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()
input = torch.tensor([[8.5]])
# out = grad(grad(net))(input)
out = net(input)
print(net.parameters,net.forward(input))

for i in range(66):
    optimizer.zero_grad()
    out = net(input)
    loss = criterion(out, input)
    loss.backward()
    optimizer.step()
    print(input, out)

