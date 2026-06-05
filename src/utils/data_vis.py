### Visualization utilities for MT data and resistivity models ###
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_single_velocity(data):
    """
    Plot a 2D heatmap of a single resistivity model (jet colormap).

    Args:
        data (torch.Tensor): Resistivity model tensor, expected shape [1, 1, depth, length]
    Note:
        The function has an undefined `ax` variable issue; kept as-is from original code.
    """
    plt.imshow(data[0,0,].detach().cpu().numpy(), cmap='jet')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Depth (m)')
    plt.colorbar()
    plt.show()

def plot_single_mt_2(data):
    """
    Plot a 2D grayscale image of single MT data, adapted for time-offset dimensions.

    Args:
        data (np.ndarray/torch.Tensor): MT data array/tensor of shape [time_samples, num_stations]
    """
    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap='jet', vmin=-1, vmax=2)
    ax.set_aspect(aspect=nx/nz)
    ax.set_xticks(range(0, nx, int(300//(1050/nx)))[:5])
    ax.set_xticklabels(range(0, 1050, 300))
    ax.set_title('Offset (m)', y=1.08)
    ax.set_yticks(range(0, nz, int(200//(1000/nz)))[:5])
    ax.set_yticklabels(range(0, 1000, 200))
    ax.set_ylabel('Time (ms)', fontsize=18)
    fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    plt.show()

def plot_single_v(data):
    """
    Plot a 2D heatmap of a single resistivity model (fixed version with proper ax definition).

    Args:
        data (torch.Tensor): Resistivity model tensor, expected shape [1, 1, depth, length]
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(data[0, 0].detach().cpu().numpy(), cmap='jet')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Depth (m)')
    fig.colorbar(cax, ax=ax)
    plt.show()

def plot_single(data, path):
    """
    Plot a single MT/resistivity data image with standardized axes and save to file.

    Args:
        data (np.ndarray/torch.Tensor): Data to plot, shape [nz, nx]
        path (str): Output image path (including filename, e.g., './output/mt_01.png')
    """
    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap='jet', vmin=-1, vmax=2)
    ax.set_aspect(aspect=nx / nz)

    num_ticks = 5
    x_ticks = np.linspace(0, nx - 1, num_ticks).astype(int)
    x_labels = np.linspace(0, 700, num_ticks).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Length (m)')

    ax.set_yticks(range(0, nz, int(200 // (1000 / nz)))[:5])
    ax.set_yticklabels(range(0, 1000, 200))
    ax.set_ylabel('Time (ms)', fontsize=18)

    plt.show(fig)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
