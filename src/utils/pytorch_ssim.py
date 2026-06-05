# Source: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
# PyTorch-based Structural Similarity (SSIM) metric for evaluating image similarity

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    """
    Generate a 1D Gaussian kernel (weight vector).

    Gaussian formula: G(x) = exp(-(x - mu)^2 / (2*sigma^2)), where mu is the window center.

    Args:
        window_size (int): Gaussian window size (odd, typically 11)
        sigma (float): Standard deviation controlling kernel smoothness
    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel of shape [window_size]
    """
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """
    Create a 2D Gaussian convolution window expanded to the specified number of channels,
    used for subsequent mean/variance convolution computations.

    Args:
        window_size (int): Gaussian window size (same as gaussian function)
        channel (int): Number of image channels (e.g., 1 for grayscale, 3 for RGB)
    Returns:
        Variable(torch.Tensor): 2D Gaussian window of shape [channel, 1, window_size, window_size]
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    """
    Core SSIM computation: computes local mean, variance, and covariance via Gaussian
    convolution, then substitutes into the SSIM formula.

    SSIM formula: SSIM(x,y) = [(2*mu_x*mu_y + C1)(2*sigma_xy + C2)] /
                              [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]
    where:
        mu_x, mu_y: local means of images x and y
        sigma_x^2, sigma_y^2: local variances of images x and y
        sigma_xy: local covariance of x and y
        C1, C2: constants to avoid division by zero (C1=(K1*L)^2, C2=(K2*L)^2,
                L = pixel value range, K1=0.01, K2=0.03)

    Args:
        img1 (torch.Tensor): Input image 1 of shape [B, C, H, W]
        img2 (torch.Tensor): Input image 2 of same shape
        window (torch.Tensor): 2D Gaussian convolution window
        window_size (int): Gaussian window size
        channel (int): Number of image channels
        size_average (bool): If True, average over all pixels to return a scalar;
                             otherwise return per-sample SSIM
    Returns:
        torch.Tensor: SSIM value (scalar or per-sample tensor)
    """
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    """
    PyTorch module wrapping SSIM computation, compatible with nn.Module interface
    for integration into neural networks.
    """
    def __init__(self, window_size = 11, size_average = True):
        """
        Initialize SSIM module.

        Args:
            window_size (int): Gaussian window size (default 11, per SSIM standard)
            size_average (bool): Whether to average results (default True)
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        Forward pass: compute SSIM between two images, auto-adapting to channel count
        and device (CPU/GPU).

        Args:
            img1 (torch.Tensor): Input image 1 of shape [B, C, H, W]
            img2 (torch.Tensor): Input image 2 of same shape
        Returns:
            torch.Tensor: SSIM value (scalar or per-sample tensor)
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    """
    Convenience function: compute SSIM between two images without instantiating the SSIM class.

    Args:
        img1 (torch.Tensor): Input image 1 of shape [B, C, H, W]
        img2 (torch.Tensor): Input image 2 of same shape
        window_size (int): Gaussian window size (default 11)
        size_average (bool): Whether to average results (default True)
    Returns:
        torch.Tensor: SSIM value (scalar or per-sample tensor)
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
