from tkinter import TRUE
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.benchmark import total_variation_loss, tikhonov_loss


# ===================== Predefined loss function instances =====================
Huber = nn.SmoothL1Loss()  # Huber loss: combines L1 (robust) and L2 (smooth), insensitive to outliers
l1 = nn.L1Loss()            # L1 loss (MAE): focuses on overall error, robust
l2 = nn.MSELoss()           # L2 loss (MSE): more sensitive to large errors, smoother optimization

class ResultsDict:
    """
    Helper class: accumulates loss values and image quality metrics during optimization.

    Core functions: compute MAE/RMSE/SSIM metrics, combine total loss, store per-step results.
    """
    def __init__(self, data_trans_module, ssim_loss, loss_type, regularization_method, reg_lambda):
        """
        Initialize the metrics accumulation class.

        Args:
            data_trans_module: Data transformation module (normalization, noise, etc.)
            ssim_loss: SSIM loss function instance
            loss_type: Observation loss type ('l1'/'l2'/'Huber')
            regularization_method: Regularization method instance
            reg_lambda: Regularization loss weight coefficient
        """
        self.data_trans = data_trans_module
        self.ssim_loss = ssim_loss
        self.loss_type = loss_type
        self.results_dict = self.create_results_dict()
        self.regularization_method = regularization_method

    def create_results_dict(self):
        """Create an empty dictionary for storing per-step optimization metrics."""
        results_dict = {
            'total_losses': [],  # Total loss (obs loss + reg loss * weight)
            'obs_losses': [],    # Observation loss (MT data misfit)
            'reg_losses': [],    # Regularization loss (resistivity model prior)
            'ssim': [],          # Structural similarity index
            'mae': [],           # Mean absolute error
            'rmse': []           # Root mean square error
        }
        return results_dict

    def calculate_metrics(self, mu, mu_true, y):
        """
        Compute resistivity model quality metrics.

        Args:
            mu: Predicted resistivity model tensor (batch, 1, H, W)
            mu_true: True resistivity model tensor (batch, 1, H, W)
            y: MT data tensor (placeholder, not used in computation)
        Returns:
            mae: Mean absolute error (scalar tensor)
            rmse: Root mean square error (float)
            ssim: Structural similarity index (scalar tensor)
        """
        vm_sample_unnorm = mu.detach().to('cpu')
        vm_data_unnorm = mu_true.detach().to('cpu')
        mae = l1(vm_sample_unnorm, vm_data_unnorm)
        mse = l2(vm_sample_unnorm, vm_data_unnorm)
        rmse = np.sqrt(mse.item())
        ssim = self.ssim_loss((vm_sample_unnorm), (vm_data_unnorm))

        return mae, rmse, ssim

    def calcualte_mt_loss(self, predicted_mt, y, loss_type, mode, freq_n):
        """
        Compute MT data observation loss (misfit between predicted and true values).

        Note: function name has a known typo (calcualte -> calculate), kept for consistency.

        Args:
            predicted_mt: Forward-modeled MT data tensor
            y: True/noisy MT data tensor
            loss_type: Loss type (uses self.loss_type; parameter is a placeholder)
        Returns:
            loss: Observation loss value (scalar tensor)
        """
        if mode == "TE":
            y = y[0, 0:2, :freq_n, :]
        elif mode == "TM":
            y = y[0, 2:4, :freq_n, :]
        elif mode == "TETM":
            y = y[0, :4, :freq_n, :]
        if self.loss_type == 'l1':
            loss = l1(y.float(), predicted_mt.float())
        elif self.loss_type == 'l2':
            loss = l2(y.float(), predicted_mt.float())
        elif self.loss_type == 'Huber':
            loss = Huber(y.float(), predicted_mt.float())
        elif self.loss_type == 'ssim':
            loss = self.ssim_loss(y.float(), predicted_mt.float())
        return loss

    def calcualte_raw_reg_loss(self, mu, reg_lambda):
        """
        Compute raw regularization loss (before multiplying by weight coefficient reg_lambda).

        Note: function name has a known typo (calcualte -> calculate), kept for consistency.

        Args:
            mu: Predicted resistivity model tensor
            reg_lambda: Regularization weight (placeholder, applied in total_loss)
        Returns:
            raw_reg_loss: Raw regularization loss value (scalar tensor)
        """
        raw_reg_loss = self.regularization_method.get_reg_loss(mu)
        return raw_reg_loss

    def calcualte_total_loss(self, loss_obs, raw_reg_loss, reg_lambda):
        """
        Compute total loss: observation loss + regularization loss * weight coefficient.

        Note: function name has a known typo (calcualte -> calculate), kept for consistency.

        Args:
            loss_obs: Observation loss value
            raw_reg_loss: Raw regularization loss value
            reg_lambda: Regularization weight coefficient
        Returns:
            total_loss: Total loss value (optimization objective)
        """
        total_loss = loss_obs + reg_lambda * raw_reg_loss
        return total_loss

    def update(self, total_losses, obs_losses, reg_losses, ssim, mae, rmse):
        """
        Append current optimization step metrics to the results dictionary.

        Args:
            total_losses: Current step total loss
            obs_losses: Current step observation loss
            reg_losses: Current step regularization loss
            ssim: Current step SSIM value
            mae: Current step MAE value
            rmse: Current step RMSE value
        """
        self.results_dict['total_losses'].append(total_losses.item())
        self.results_dict['obs_losses'].append(obs_losses.item())
        self.results_dict['reg_losses'].append(reg_losses.item())
        self.results_dict['ssim'].append(ssim.item())
        self.results_dict['mae'].append(mae.item())
        self.results_dict['rmse'].append(rmse)

    def get_results(self):
        """Return all accumulated optimization metrics."""
        return self.results_dict

class Regularization_method:
    """
    Regularization method factory: creates/computes regularization loss by type.

    Supported types: l2 (Tikhonov), tv (total variation), None.
    """
    def __init__(self, regularization_type):
        """
        Initialize the regularization factory.

        Args:
            regularization_type: Regularization type ('l2'/'tv'/None)
        """
        self.regularization_type = regularization_type

    def initialize_regularization(self):
        """Initialize regularization method (fallback; actual computation in get_reg_loss)."""
        if self.regularization_type == 'l2':
            regularization = tikhonov_loss
        elif self.regularization_type == 'tv':
            regularization = total_variation_loss
        else:
            regularization = None
        return regularization

    def get_reg_loss(self, mu):
        """
        Compute regularization loss for the specified type.

        Args:
            mu: Predicted resistivity model tensor (batch, 1, H, W)
        Returns:
            reg_loss: Regularization loss value (scalar tensor)
        """
        if self.regularization_type == 'l2':
            reg_loss = tikhonov_loss(mu)
        elif self.regularization_type == 'tv':
            reg_loss = total_variation_loss(mu)
        else:
            reg_loss = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        return reg_loss

class run_inversion:
    """
    Core inversion class: executes the resistivity model inversion optimization loop.

    Pipeline: MT forward -> loss computation (obs + reg) -> gradient update -> metric evaluation.
    """
    def __init__(self, data_trans_module, data_vis_module, pytorch_ssim_module, regularization):
        """
        Initialize the inversion class.

        Args:
            data_trans_module: Data transformation module (normalization, noise, missing traces)
            data_vis_module: Data visualization module (reserved, unused in original code)
            pytorch_ssim_module: SSIM metric module
            regularization: Initial regularization type ('l2'/'tv'/None)
        """
        self.data_trans = data_trans_module
        self.data_vis = data_vis_module
        self.ssim_loss = pytorch_ssim_module.SSIM(window_size=11)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.regularization_method = Regularization_method(regularization)

    def sample(self, mu, mu_true, y, ts, lr=0.001, reg_lambda=0.01, fwi_forward=None,
            loss_type="l2", noise_std=0, missing_number=0, regularization=None, mode="TETM", freq_n=5):
        """
        Execute the resistivity model inversion optimization main loop (L-BFGS optimizer,
        reuses closure computation results).

        Args:
            mu: Initial resistivity model tensor (batch, 1, H, W) — optimization starting point
            mu_true: True resistivity model tensor (batch, 1, H, W) — for evaluation
            y: MT data tensor (observed data)
            ts: Total optimization steps
            lr: Optimizer learning rate (step size scaling factor in L-BFGS, no scheduling effect)
            reg_lambda: Regularization loss weight coefficient (default 0.01)
            fwi_forward: MT forward operator (callable, input resistivity -> output MT data)
            loss_type: Observation loss type ('l1'/'l2'/'Huber', default 'l2')
            noise_std: Std of noise added to MT data (default 0, no noise)
            missing_number: Number of missing MT traces (default 0, no missing)
            regularization: Override regularization type (takes priority over init value)
        Returns:
            mu: Optimized resistivity model tensor
            final_results: Dictionary of all loss and metric values during optimization
        """
        assert mode in ["TE", "TM", "TETM"], "Forward mode must be 'TE'/'TM'/'TETM'"
        assert mu.shape[0] == y.shape[0], "Resistivity model batch size must match MT data batch size"
        assert loss_type in ["l1", "l2", "Huber", 'ssim'], "Loss function must be 'l1'/'l2'/'Huber'"
        assert regularization in ["l2", "tv", "hybrid", None], "Regularization type must be 'l2'/'tv'/None"
        if fwi_forward is None or not callable(fwi_forward):
            raise ValueError("fwi_forward must be a callable MT forward function")
        fwi_forward = fwi_forward.to(self.device)

        # Override regularization type
        if regularization is not None:
            self.regularization_method = Regularization_method(regularization)

        # Resistivity model normalization
        self.max_log = torch.max(torch.log10(1 / mu_true))
        mu = self.data_trans.log_normalize_encode(1 / mu, self.max_log)
        mu_true = self.data_trans.log_normalize_encode(1 / mu_true, self.max_log)

        # Tensor type conversion with gradient tracking
        mu = mu.float().clone().detach().to(self.device).requires_grad_(True)
        mu_true = mu_true.float().to(self.device)

        # L-BFGS optimizer initialization
        optimizer = torch.optim.LBFGS(
            [mu],
            lr=1.0,
            max_iter=50,
            max_eval=None,
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        # Initialize metrics accumulation class
        results_dict = ResultsDict(self.data_trans, self.ssim_loss, loss_type, self.regularization_method, reg_lambda)

        # MT data preprocessing
        y = self.data_trans.add_noise_to_mt(y, noise_std)
        y = self.data_trans.missing_trace(y, missing_number)
        y = y.float().to(self.device)

        pbar = tqdm(range(ts), desc="L-BFGS Optimizing", unit="step")

        # Main optimization loop
        for l in pbar:
            closure_cache = {}

            def closure():
                optimizer.zero_grad()
                # 1. MT forward
                predicted_mt = fwi_forward(self.data_trans.log_normalize_decode(mu[0, 0, ...], self.max_log))
                # 2. Compute observation loss
                loss_obs = results_dict.calcualte_mt_loss(predicted_mt, y, loss_type, mode, freq_n)
                # 3. Compute regularization loss
                raw_reg_loss = results_dict.calcualte_raw_reg_loss(mu, reg_lambda)
                # 4. Compute total loss
                total_loss = results_dict.calcualte_total_loss(loss_obs, raw_reg_loss, reg_lambda)
                # 5. Backpropagation
                total_loss.backward()

                # Cache computed results (detach gradients, store values only)
                closure_cache['total_loss'] = total_loss.detach()
                closure_cache['loss_obs'] = loss_obs.detach()
                closure_cache['raw_reg_loss'] = raw_reg_loss.detach()
                closure_cache['predicted_mt'] = predicted_mt.detach()

                return total_loss

            # Execute L-BFGS optimization step
            optimizer.step(closure)

            # Reuse cached results to avoid redundant computation
            with torch.no_grad():
                total_loss = closure_cache['total_loss']
                loss_obs = closure_cache['loss_obs']
                raw_reg_loss = closure_cache['raw_reg_loss']
                mae, rmse, ssim = results_dict.calculate_metrics(mu, mu_true, y)

            # Update metrics and progress bar
            results_dict.update(total_loss, loss_obs, raw_reg_loss, ssim, mae, rmse)
            mu.data.clamp_(-1, 1)

            postfix_dict = {
                'total_loss': total_loss.item(),
                'obs_loss': loss_obs.item(),
                'reg_loss': raw_reg_loss.item(),
                'MAE': mae.item(),
                'SSIM': ssim.item(),
                'RMSE': rmse.item(),
                'lr': optimizer.param_groups[0]['lr']
            }
            pbar.set_postfix(postfix_dict)

            log_info = f"Step {l+1}/{ts} | " + " | ".join([f"{k}: {v:.6f}" for k, v in postfix_dict.items()])
            pbar.write(log_info)

        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        final_results = results_dict.get_results()
        return mu, final_results, closure_cache['predicted_mt']
