import torch
import torch.optim as optim

from model import Net
from env import Env
from visualize import *

'''
init φ
while True:
    sample task τ
    find φ' for τ using k steps
    φ := φ + ε(φ' - φ)
'''

max_iters = 100
vis_iter = 10
ε = 0.01
α = 0.01
k = 200
p = 10

def copy_params(φ, φ2):
    φ2.load_state_dict(φ.state_dict())

def soft_update(φ, φ2, ε):
    for new_param, old_param in zip(φ2.parameters(), φ.parameters()):
        old_param.data.copy_(old_param.data + (ε * (new_param.data - old_param.data)))

env = Env()
φ = Net()
φ2 = Net()
inner_optimizer = optim.SGD(φ2.parameters(), lr=α)

for iter in range(max_iters):
    copy_params(φ, φ2)
    env.new()
    for step in range(k):
        points = env.sample_points(p)
        sample_loss = env.get_loss(φ2, points)
        # plot_loss(step, sample_loss.item(), 'φ2')

        inner_optimizer.zero_grad()
        sample_loss.backward()
        inner_optimizer.step()
    soft_update(φ, φ2, ε)

    if iter % vis_iter == vis_iter - 1:
        plot_f(env, φ, φ2)




test_net = Net()
test_optimizer = optim.SGD(test_net.parameters(), lr=α)
for step in range(1000):
    points = env.sample_points(p)
    sample_loss = env.get_loss(test_net, points)
    test_optimizer.zero_grad()
    sample_loss.backward()
    test_optimizer.step()
    plot_loss(step, sample_loss.item(), 'from scratch')


phi_optimizer = optim.SGD(φ.parameters(), lr=α)
for step in range(1000):
    points = env.sample_points(p)
    sample_loss = env.get_loss(φ, points)
    phi_optimizer.zero_grad()
    sample_loss.backward()
    phi_optimizer.step()
    plot_loss(step, sample_loss.item(), 'φ')
