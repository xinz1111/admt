### Core data transformation utilities for MT inversion ###
import torch
import numpy as np
from scipy.ndimage import gaussian_filter


def log_normalize_encode(x, max_log, eps=1e-12):
    """
    Encode raw resistivity values to [-1, 1] via log normalization.

    Compresses the dynamic range of resistivity values ([1, 10000] -> [0, 4] in log10),
    then linearly maps to [-1, 1].

    Normalization flow: log10(x) -> [0, 1] -> [-1, 1]

    Args:
        x: Raw input tensor (resistivity, range [1, 10000], all positive real values)
        max_log: Global maximum of log10(raw data), precomputed externally (<= 4)
        eps: Small value to prevent division by zero
    Returns:
        x_norm: Normalized tensor strictly in [-1, 1]
    """
    log_x = torch.log10(x)
    x_norm_01 = log_x / (max_log + eps)
    x_norm = x_norm_01 * 2 - 1
    return x_norm

def log_normalize_decode(x_norm, max_log, eps=1e-12):
    """
    Decode normalized values back to raw resistivity [1, 10000].

    Reverse of log_normalize_encode: [-1, 1] -> [0, 1] -> inverse log10 -> [1, 10000]

    Args:
        x_norm: Normalized input tensor in [-1, 1]
        max_log: Same global maximum used during encoding
        eps: Same epsilon as encoding, for inverse robustness
    Returns:
        x_original: Recovered raw resistivity tensor in [1, 10000]
    """
    x_norm_01 = (x_norm + 1) / 2
    log_x_original = x_norm_01 * (max_log + eps)
    x_original = torch.pow(10, log_x_original)
    return 1 / x_original

# ===================== Forward response (supports negative values): 2 standalone functions =====================
def forward_normalize(forward_tensor, target_range=(0, 1), global_min=None, global_max=None, eps=1e-8):
    """
    Normalize forward response tensor (supports negative values, returns statistics for denormalization).

    Automatically or manually computes min/max, linearly maps to target range, returns 3 results.

    Args:
        forward_tensor: Forward response tensor (arbitrary shape, supports negative values)
        target_range: Target normalization range, either (0, 1) or (-1, 1)
        global_min: Global minimum (manually specified, for batch consistency)
        global_max: Global maximum (manually specified, for batch consistency)
        eps: Small value to prevent division by zero
    Returns:
        norm_forward: Normalized forward response tensor
        orig_min: Original data minimum (auto-computed or manually specified)
        orig_max: Original data maximum (auto-computed or manually specified)
    """
    if not isinstance(forward_tensor, torch.Tensor):
        forward_tensor = torch.tensor(forward_tensor, dtype=torch.float32)

    assert target_range in [(0, 1), (-1, 1)], "Target range only supports (0,1) or (-1,1)"
    target_min, target_max = target_range

    if global_min is not None and global_max is not None:
        orig_min = torch.tensor(global_min, dtype=torch.float32, device=forward_tensor.device)
        orig_max = torch.tensor(global_max, dtype=torch.float32, device=forward_tensor.device)
    else:
        orig_min = torch.min(forward_tensor)
        orig_max = torch.max(forward_tensor)

    orig_range = orig_max - orig_min + eps
    norm_01 = (forward_tensor - orig_min) / orig_range

    if target_range == (0, 1):
        norm_forward = norm_01
    else:
        norm_forward = norm_01 * 2 - 1

    return norm_forward, orig_min, orig_max

def forward_denormalize(norm_forward_tensor, orig_min, orig_max, target_range=(0, 1), eps=1e-8):
    """
    Denormalize forward response tensor (inverse of forward_normalize, requires original statistics).

    Args:
        norm_forward_tensor: Normalized forward response tensor
        orig_min: Original data minimum from normalization step
        orig_max: Original data maximum from normalization step
        target_range: Target range used during normalization (must match)
        eps: Small value to prevent division by zero
    Returns:
        orig_forward: Recovered raw forward response (may contain negative values)
    """
    if not isinstance(norm_forward_tensor, torch.Tensor):
        norm_forward_tensor = torch.tensor(norm_forward_tensor, dtype=torch.float32)

    assert target_range in [(0, 1), (-1, 1)], "Target range only supports (0,1) or (-1,1)"
    target_min, target_max = target_range

    if not isinstance(orig_min, torch.Tensor):
        orig_min = torch.tensor(orig_min, dtype=torch.float32, device=norm_forward_tensor.device)
    if not isinstance(orig_max, torch.Tensor):
        orig_max = torch.tensor(orig_max, dtype=torch.float32, device=norm_forward_tensor.device)

    orig_range = orig_max - orig_min + eps

    if target_range == (0, 1):
        norm_01 = norm_forward_tensor
    else:
        norm_01 = (norm_forward_tensor + 1) / 2

    orig_forward = norm_01 * orig_range + orig_min
    return orig_forward

class MaxMinNormalizer(object):

    def __init__(self, x, eps=1e-8):
        super(MaxMinNormalizer, self).__init__()
        self.min = torch.min(x)
        self.max = torch.max(x)
        self.eps = eps
        self.range = self.max - self.min + self.eps

    def encode(self, x):
        x_norm = 2 * (x - self.min) / self.range - 1
        return x_norm

    def decode(self, x_norm):
        x_original = (x_norm + 1) * self.range / 2 + self.min
        return x_original


class LogNormalizer(object):
    """
    Log normalization: divide log10(x) by the global maximum of log10(x).

    Encode: d_tilde_i = log10(d_i) / max_j(log10(d_j))
    Decode: d_i = 10^(d_tilde_i * max_j(log10(d_j)))
    """
    def __init__(self, x, eps=1e-8):
        super(LogNormalizer, self).__init__()
        log_x = torch.log10(x)
        self.max_log = torch.max(log_x)
        self.eps = eps

    def encode(self, x):
        log_x = torch.log10(x)
        x_norm = log_x / (self.max_log + self.eps)
        return x_norm

    def decode(self, x_norm):
        log_x_original = x_norm * (self.max_log + self.eps)
        x_original = torch.pow(10, log_x_original)
        return x_original


def v_normalize(v):
    """
    Normalize resistivity values to [-1, 1].

    Args:
        v (np.ndarray/torch.Tensor): Raw resistivity data, typically in range 1500~4500
    Returns:
        np.ndarray/torch.Tensor: Normalized resistivity data in [-1, 1]
    """
    return (((v - 1500) / 3000) * 2) - 1

def v_denormalize(v_norm):
    """
    Denormalize resistivity values from [-1, 1] back to raw range (1500~4500).

    Args:
        v_norm (np.ndarray/torch.Tensor): Normalized resistivity data in [-1, 1]
    Returns:
        np.ndarray/torch.Tensor: Denormalized raw resistivity data
    """
    return ((v_norm + 1) / 2) * 3000 + 1500


def s_normalize_none(s):
    """
    No normalization on MT data, keep original scale.

    Args:
        s (np.ndarray/torch.Tensor): Raw MT data
    Returns:
        np.ndarray/torch.Tensor: Unmodified raw MT data
    """
    return s

def s_normalize(s):
    """
    Normalize MT data to [-1, 1].

    Args:
        s (np.ndarray/torch.Tensor): Raw MT data, typically in range [-20, 60]
    Returns:
        np.ndarray/torch.Tensor: Normalized MT data in [-1, 1]
    """
    return (((s + 20) / 80) * 2) - 1

def s_denormalize(s_norm):
    """
    Denormalize MT data from [-1, 1] back to raw range [-20, 60].

    Args:
        s_norm (np.ndarray/torch.Tensor): Normalized MT data in [-1, 1]
    Returns:
        np.ndarray/torch.Tensor: Denormalized raw MT data
    """
    return ((s_norm + 1) / 2) * 80 - 20

def add_noise_to_mt(y, std):
    """
    Add Gaussian noise to MT data to simulate real-world acquisition noise.

    Args:
        y (torch.Tensor): Raw MT data tensor
        std (float): Gaussian noise standard deviation (std=0 means no noise added)
    Returns:
        torch.Tensor: MT data tensor with added noise
    """
    assert std >= 0, "Noise standard deviation must be >= 0"
    if std == 0:
        return y
    else:
        y = y.detach().cpu().numpy()
        noise = np.random.normal(0, std, y.shape)
        y_noisy = y + noise
        y = torch.tensor(y_noisy).float()
        return y

def prepare_initial_model(v_true, initial_type=None, sigma=None, linear_coeff=1.0):
    """
    Generate initial model from true MT model for inversion initialization.

    Args:
        v_true (torch.Tensor): True MT model tensor, shape typically [batch, 1, height, width]
        initial_type (str): Initial model type: 'smoothed', 'homogeneous', or 'linear'
        sigma (float): Gaussian filter std (only when initial_type='smoothed')
        linear_coeff (float): Linear gradient coefficient (reserved, unused)
    Returns:
        torch.Tensor: Generated initial model tensor, same device and shape as input
    """
    assert initial_type in ['smoothed', 'homogeneous', 'linear'], \
        "Please choose initial model type from: 'smoothed', 'homogeneous', 'linear'"
    v = v_true.clone()
    v_np = v.cpu().numpy()

    if initial_type == 'smoothed':
        v_blurred = gaussian_filter(v_np, sigma=sigma)
    elif initial_type == 'homogeneous':
        min_top_row = np.min(v_np[0, 0, 0, :])
        v_blurred = np.full_like(v_np, min_top_row)
    elif initial_type == 'linear':
        v_min = np.min(v_np)
        v_max = np.max(v_np)
        height = v_np.shape[2]
        depth_gradient = np.linspace(v_min, v_max, height)
        depth_gradient = depth_gradient.reshape(-1, 1)
        v_blurred = np.tile(depth_gradient, (1, v_np.shape[3]))
        v_blurred = v_blurred.reshape(1, 1, height, -1)

    v_blurred = torch.tensor(v_blurred).float().to(v_true.device)
    return v_blurred

def prepare_initial_model1(v_true, initial_type='homogeneous', sigma=None, linear_coeff=1.0,
                         avg_velocity=1e-2, layer_depths=None, layer_velocities=None,
                         v_min_est=1e0, v_max_est=1e-4):
    """
    Generate initial model with extended options (homogeneous, linear gradient, layered, smoothed).

    Extended args:
        input_shape: Model shape [batch, 1, height, width] (derived from v_true)
        avg_velocity: Average conductivity for homogeneous model
        layer_depths: Depth interfaces for layered model
        layer_velocities: Layer conductivities for layered model
        v_min_est: Minimum estimated conductivity for linear gradient (shallow, default 1e-4)
        v_max_est: Maximum estimated conductivity for linear gradient (deep, default 1e0)
    Other args unchanged.

    Returns:
        v_blurred: PyTorch tensor matching v_true shape, suitable as inversion initial model
    """
    assert isinstance(v_true, torch.Tensor), f"Input v_true must be a torch tensor, got {type(v_true)}"
    input_shape = v_true.shape
    valid_types = ['smoothed', 'homogeneous', 'linear', 'layered']
    assert initial_type in valid_types, f"Please choose initial model type from: {valid_types}"

    # 1. Homogeneous initial model (default when no prior)
    if initial_type == 'homogeneous':
        v_blurred = torch.full(input_shape, avg_velocity, dtype=torch.float32, device=v_true.device)

    # 2. Linear gradient initial model (depth-increasing conductivity)
    elif initial_type == 'linear':
        batch, ch, height, width = input_shape
        depth_gradient = torch.logspace(np.log10(v_min_est), np.log10(v_max_est), height,
                                        dtype=torch.float32, device=v_true.device)
        depth_gradient = depth_gradient.reshape(1, 1, height, 1)
        v_blurred = depth_gradient.repeat(batch, ch, 1, width)

    # 3. Layered initial model (with geological layering info)
    elif initial_type == 'layered':
        assert layer_depths is not None and layer_velocities is not None, \
            "Layered model requires layer_depths and layer_velocities"
        batch, ch, height, width = input_shape
        dx = 10
        layer_indices = [int(d / dx) for d in layer_depths]
        assert max(layer_indices) <= height, "Layer depth interfaces exceed model height"
        v_blurred = torch.zeros(input_shape, dtype=torch.float32, device=v_true.device)
        for b in range(batch):
            v_blurred[b, :, :layer_indices[0], :] = layer_velocities[0]
            for i in range(1, len(layer_indices)):
                v_blurred[b, :, layer_indices[i-1]:layer_indices[i], :] = layer_velocities[i]
            v_blurred[b, :, layer_indices[-1]:, :] = layer_velocities[-1]

    # 4. Low-resolution smoothed model (with external low-res model)
    elif initial_type == 'smoothed':
        v_np = v_true.cpu().numpy()
        v_blurred = gaussian_filter(v_np, sigma=sigma)
        v_blurred = torch.tensor(v_blurred).float().to(v_true.device)

    assert isinstance(v_blurred, torch.Tensor), "Returned initial model must be a torch tensor"
    return 1 / ((1 / v_blurred) + 1)

def generate_initial_measured_model(obs_data, core_z_size=53, core_y_size=79):
    """
    Generate a uniform half-space initial model for measured field data (dynamic dimension adaptation).

    Args:
        obs_data: Measured data tensor of shape (4, 60, 15), device must match model
        core_z_size: Number of core region depth grid cells (currently 53 for measured data)
        core_y_size: Number of core region horizontal grid cells (currently 79 for measured data)
    Returns:
        sigma_initial: Uniform conductivity initial model of shape (1, 1, core_z_size, core_y_size)
    """
    rho_te = obs_data[0, :, :]
    rho_tm = obs_data[2, :, :]

    rho_all = torch.cat([rho_te.flatten(), rho_tm.flatten()])
    rho_all = rho_all[rho_all > 0]

    rho_avg = torch.exp(torch.mean(torch.log(rho_all)))

    sigma_initial = torch.ones((core_z_size, core_y_size), device=obs_data.device, dtype=obs_data.dtype)
    sigma_initial = sigma_initial / (rho_avg * 20)

    return sigma_initial.unsqueeze(0).unsqueeze(0)

def missing_trace(y, num_missing):
    """
    Simulate missing MT measurement stations (set specified number of stations to zero).

    Args:
        y (torch.Tensor): Raw MT data tensor of shape [batch_size, num_modes, num_freqs, num_stations]
        num_missing (int): Number of stations to zero out per mode (0 = no missing)
    Returns:
        torch.Tensor: MT data tensor with missing stations zeroed out
    """
    assert num_missing >= 0, "Number of missing stations must be >= 0"
    if num_missing == 0:
        return y
    else:
        y_np = y.detach().cpu().numpy()
        batch_size, num_modes, num_freqs, num_stations = y.shape
        y_missing = y_np.copy()
        for b in range(batch_size):
            for m in range(num_modes):
                missing_indices = np.random.choice(num_stations, num_missing, replace=False)
                y_missing[b, m, :, missing_indices] = 0
        y = torch.tensor(y_missing).float()
    return y
