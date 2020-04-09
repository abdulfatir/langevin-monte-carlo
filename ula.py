# Generates samples from the heart-shaped density and writes to heart.npy
import numpy as np
import torch
import random
import math
from tqdm import tqdm

def potential(z):
    z = z.view(-1, 2)
    x, y = z[:, 0], z[:, 1]
    u = 0.8 * x ** 2 + (y - ((x**2)**(1/3)))**2
    u = u / 2**2
    return u

def unadjusted_langevin_algorithm(n_samples=100000, step=0.01):
    burn_in = 10000
    Z0 = torch.randn(1, 2)
    Zi = Z0
    samples = []
    for i in tqdm(range(n_samples + burn_in)):
        Zi.requires_grad_()
        u = potential(Zi).mean()
        grad = torch.autograd.grad(u, Zi)[0]
        Zi = Zi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(1, 2)
        samples.append(Zi.detach().numpy())
    return np.concatenate(samples, 0)[burn_in:]


samples = unadjusted_langevin_algorithm()
np.save('heart-ula.npy', samples)
