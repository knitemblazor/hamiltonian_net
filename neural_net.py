import torch
import torch.nn as nn
from methods import space_points_gen


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 8, bias=False)
        self.fc2 = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


xpointe = space_points_gen()
net = Net()
print(list(net.parameters()))
input = torch.tensor([[8.5]])
out = net(input)

# print(out)
