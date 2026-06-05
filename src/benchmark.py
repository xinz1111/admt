import torch

def total_variation_loss(mu):
    """
    Compute anisotropic total variation loss for a 4D tensor.

    TV loss constrains the smoothness of tensors (images/model parameters) by penalizing
    sharp pixel/value variations. The anisotropic version uses L1 norm (absolute value)
    for horizontal and vertical gradients, commonly used for image denoising and regularization.

    Args:
        mu (torch.Tensor): Input 4D tensor of shape [batch, channels, height, width]
    Returns:
        torch.Tensor: Scalar total variation loss value
    """
    diff_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])
    diff_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])
    tv_loss = torch.mean(diff_x) + torch.mean(diff_y)
    return tv_loss

def tikhonov_loss(mu):
    """
    Compute Tikhonov loss (L2 gradient penalty) for a 4D tensor.

    Also known as L2-norm gradient loss, this is a smoothing regularization that penalizes
    gradient magnitudes. Compared to TV loss (L1), the L2 norm is more sensitive to large
    gradients (squared amplification), favoring global smoothness.

    Args:
        mu (torch.Tensor): Input 4D tensor of shape [batch, channels, height, width]
    Returns:
        torch.Tensor: Scalar Tikhonov loss value
    """
    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :]
    l2_loss_x = torch.mean(diff_x ** 2)
    l2_loss_y = torch.mean(diff_y ** 2)
    l2_loss = l2_loss_x + l2_loss_y
    return l2_loss

def joint_tikhonov_tv_loss(mu, tikhonov_weight=0.5, tv_weight=0.5):
    """
    Compute weighted joint Tikhonov + TV regularization loss for a 4D tensor.

    Combines the strengths of both regularizers: TV loss (L1) preserves edges while
    smoothing locally; Tikhonov loss (L2) suppresses large gradients for global smoothness.
    Default weights are 0.5 each; adjust to emphasize one over the other.

    Args:
        mu (torch.Tensor): Input 4D tensor of shape [batch, channels, height, width]
        tikhonov_weight (float, optional): Weight for Tikhonov loss, default 0.5
        tv_weight (float, optional): Weight for TV loss, default 0.5
    Returns:
        torch.Tensor: Scalar joint regularization loss value
    """
    tik_loss = tikhonov_loss(mu)
    tv_loss = total_variation_loss(mu)
    joint_loss = tikhonov_weight * tik_loss + tv_weight * tv_loss
    return joint_loss
