# Creates an MP4 video using samples from heart.npy 
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import argparse
from tqdm import tqdm
plt.style.use('seaborn-poster')

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=str, required=True, help='Path to the .npy file with samples.')
parser.add_argument('--start', type=int, required=True, help='Start frame number.')
parser.add_argument('--n', type=int, required=True, help='Number of frames.')

args = parser.parse_args()

from_frame = args.start
upto_frame = args.n + from_frame
base_name = args.samples.strip('.npy')
samples = np.load(args.samples)

plt.rcParams["font.family"] = "serif"
fig, axn = plt.subplots(ncols=2, figsize=(15, 8))

def npdensity1(z):
    z = np.reshape(z, [z.shape[0], 2])
    x, y = z[:, 0], z[:, 1]
    u = 0.8 * x ** 2 + (y - ((x**2)**(1/3)))**2
    u = u / 2**2
    return np.exp(-u)

lim = 4

r = np.linspace(-lim, lim, 1000)
x, y = np.meshgrid(r, r)
z = np.vstack([x.flatten(), y.flatten()]).T

q0 = npdensity1(z)
axn[0].pcolormesh(x, y, q0.reshape(x.shape),
                           cmap='viridis')
axn[0].set_aspect('equal', adjustable='box')
axn[0].set_xlim([-lim, lim])
axn[0].set_ylim([-lim, lim])
axn[0].set_title('True Density')

cmap = matplotlib.cm.get_cmap('viridis')
bg = cmap(0.)
axn[1].set_facecolor(bg)
# axn[1].hist2d(samples[:,0], samples[:,1], cmap='viridis', rasterized=False, bins=200)
axn[1].set_aspect('equal', adjustable='box')
axn[1].set_xlim([-lim, lim])
axn[1].set_ylim([-lim, lim])
axn[1].set_title('Empirical Density')

fig.suptitle('Langevin Dynamics Monte Carlo', fontsize=40)

line, = axn[0].plot([], [], lw=2, c='#f3c623')
scat = axn[0].scatter([], [], c='#dd2c00', s=150, marker='*')

def init():
    line.set_data([], [])
    return line,

def random_walk(i):
    i += 1
    if i <= 100:
        z = samples[:i]
    else:
        z = samples[i-100:i]
    line.set_data(z[:, 0], z[:, 1])
    scat.set_offsets(z[-1:])

    axn[1].clear()
#     axn[1].set_facecolor(bg)
    axn[1].set_aspect('equal', adjustable='box')
    axn[1].hist2d(samples[:i, 0], samples[:i, 1], cmap='viridis', rasterized=False, bins=200, density=True)
    axn[1].set_xlim([-lim, lim])
    axn[1].set_ylim([-lim, lim])
    axn[1].set_title('Empirical Density')
    return line, scat, #

anim = animation.FuncAnimation( fig = fig, blit=True, init_func=init, func = random_walk,
                                     interval = 10, frames=range(from_frame, upto_frame))
anim.save(f'{base_name}{from_frame}.mp4', writer='ffmpeg', dpi=200, progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))

