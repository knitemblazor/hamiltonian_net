from combined import *
from neural_net import *
import torch.optim as optim

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.08)

obj = HamiltonianNet()

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

net.apply(weights_init_normal)

# net.load_state_dict(torch.load("model(1,8,1,sig)(0,1,400)rand_losskl.pth"))
for i in range(6000):
    print("epoch: %d" % i)
    optimizer.zero_grad()
    loss = obj.error_function(net)
    # inter = loss.detach().numpy()[0][0]
    print("loss:", loss.detach().numpy()[0][0])
    loss.backward()
    optimizer.step()
    print("-------*--------")
    torch.save(net.state_dict(), "model(1,8,1,sig)(0,1,400)rand_losskl.pth")
    print("model saved")
