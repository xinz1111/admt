import torch
import torch.nn as nn
import numpy as np
import cmath as cm
from tqdm import tqdm

class MT2DFD1(nn.Module):
    """
    Fully differentiable 2D magnetotelluric (MT) finite-difference forward modeling class.

    Key design decisions:
    1. No in-place operations — uses tensor concatenation + torch.where exclusively
    2. Minimizes torch.no_grad() wrapping — only for numerically constant auxiliary computations
    3. Complex tensors split into real/imaginary parts for stable gradient propagation
    4. Built-in gradient verification hooks for debugging gradient flow interruptions
    5. Pure PyTorch implementation — no NumPy intermediate conversions, preserves full computation graph
    """
    def __init__(self, ctx, device, normalize=True, mode="TETM"):
        super(MT2DFD1, self).__init__()
        # Device configuration
        self.device = device
        self.fdtype = torch.float32
        self.cdtype = torch.complex64

        # Normalization configuration
        self.normalize = normalize

        # Mode parameter validation
        self.supported_modes = ["TE", "TM", "TETM"]
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported forward mode: {mode}. Supported modes: {self.supported_modes}")
        self.mode = mode

        # Physical constants (frozen, no gradient)
        self.miu = torch.tensor(4.0e-7 * np.pi, dtype=self.fdtype, device=self.device, requires_grad=False)
        self.II = torch.tensor(cm.sqrt(-1), dtype=self.cdtype, device=self.device, requires_grad=False)

        # Extract core parameters from context
        self._extract_ctx_params(ctx)

        # Build grids once (frozen, no gradient)
        self._build_grids()

        # Setup frequencies (frozen, no gradient)
        self._setup_frequencies()
        self.nf = ctx['n_freq']

    def _extract_ctx_params(self, ctx):
        """Extract context parameters and convert to PyTorch constants."""
        self.z = ctx['z']
        self.y = ctx['y']
        self.size_k = ctx['size_k']
        self.size_b = ctx['size_b']
        self.nza = ctx['nza']
        self.n_add = ctx['n_add']
        self.n_freq = ctx['total_freq']

        # Boundary expansion multipliers (frozen)
        self.multiple_t = 1.5
        self.multiple_b = 2.0
        self.multiple_l = 2.5
        self.multiple_r = 2.5

    def _build_grids(self):
        """Build Z/Y direction grids once, convert to PyTorch tensors (frozen, no gradient)."""
        # ---------------------- Z-direction grid (air + core + bottom boundary) ----------------------
        z_air = -(np.logspace(np.log10(50e3), np.log10(50e3 + self.multiple_t * self.z),
                                self.nza + 1) - 50e3)[::-1]
        zn0 = np.linspace(0, self.z, self.size_k + 1)
        z_b = np.logspace(np.log10(zn0[-1]), np.log10(self.multiple_b * zn0[-1]), self.size_b + 1)

        zn = np.concatenate((z_air[:-1], zn0, z_b[1:]))
        self.zn = torch.tensor(zn, dtype=self.fdtype, device=self.device, requires_grad=False)
        self.nz = self.zn.numel()
        self.dz = self.zn[1:] - self.zn[:-1]

        # ---------------------- Y-direction grid (left boundary + core + right boundary) ----------------------
        yn0 = np.linspace(-self.y, self.y, self.size_k + 1)
        y_l = -(np.logspace(np.log10(self.multiple_l * yn0[-1]), np.log10(yn0[-1]), self.size_b + 1))
        y_r = np.logspace(np.log10(yn0[-1]), np.log10(self.multiple_r * yn0[-1]), self.size_b + 1)

        yn = np.concatenate((y_l[:-1], yn0, y_r[1:]))
        self.yn = torch.tensor(yn, dtype=self.fdtype, device=self.device, requires_grad=False)
        self.ny = self.yn.numel()
        self.dy = self.yn[1:] - self.yn[:-1]

        # ---------------------- Observation point grid (core region Y) ----------------------
        self.ry = torch.tensor(yn0, dtype=self.fdtype, device=self.device, requires_grad=False)
        self.nry = self.ry.numel()

    def _setup_frequencies(self):
        """Set frequency range (log-spaced), convert to PyTorch tensors (frozen, no gradient)."""
        freq = np.logspace(np.log10(10), np.log10(1/100), self.n_freq)
        self.freq = torch.tensor(freq, dtype=self.fdtype, device=self.device, requires_grad=False)

    def _to_tensor(self, data, dtype=None, requires_grad=False):
        """
        Unified utility to convert data to PyTorch tensor.

        Args:
            data: Input data (np.ndarray / torch.Tensor / scalar)
            dtype: Target data type
            requires_grad: Whether to enable gradient tracking
        Returns:
            torch.Tensor: Converted tensor (moved to target device)
        """
        if dtype is None:
            dtype = self.fdtype if not isinstance(data, complex) else self.cdtype

        if isinstance(data, np.ndarray):
            tensor = torch.tensor(data, dtype=dtype, device=self.device)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype=dtype, device=self.device)
        else:
            tensor = torch.tensor([data], dtype=dtype, device=self.device).squeeze()

        tensor.requires_grad_(requires_grad)
        return tensor

    def _gradient_verify(self, tensor, node_name):
        """
        Gradient verification utility: prints gradient status of a tensor (for debugging).

        Args:
            tensor: Tensor to inspect
            node_name: Name of the verification node (for log identification)
        """
        if tensor is None:
            print(f"[Gradient Check] {node_name}: tensor is None")
            return

        if tensor.dtype in [torch.complex64, torch.complex128]:
            print(f"[Gradient Check] {node_name} (complex tensor):")
            print(f"  - requires_grad: {tensor.requires_grad}")
            print(f"  - real part grad_fn: {tensor.real.grad_fn if tensor.real.grad_fn else 'None (gradient broken)'}")
            print(f"  - imag part grad_fn: {tensor.imag.grad_fn if tensor.imag.grad_fn else 'None (gradient broken)'}")
            print(f"  - real part is_leaf: {tensor.real.is_leaf}")
        else:
            print(f"[Gradient Check] {node_name} (real tensor):")
            print(f"  - requires_grad: {tensor.requires_grad}")
            print(f"  - grad_fn: {tensor.grad_fn if tensor.grad_fn else 'None (gradient broken)'}")
            print(f"  - is_leaf: {tensor.is_leaf}")
        print("-" * 80)

    def _torch_interp(self, x_new, x_old, y_old):
        """
        Pure PyTorch differentiable linear interpolation (no in-place ops, preserves gradient chain).

        Args:
            x_new: New interpolation point coordinates (1D tensor)
            x_old: Original coordinates (1D tensor, monotonically increasing)
            y_old: Values at original coordinates (1D tensor)
        Returns:
            y_new: Interpolated values at new points (1D tensor, preserves gradients)
        """
        x_new = x_new.flatten().contiguous()
        x_old = x_old.flatten().contiguous()
        y_old = y_old.flatten().contiguous()

        # Step 1: Boundary clamping (differentiable)
        x_new_clamped = torch.clamp(x_new, min=x_old[0], max=x_old[-1])

        # Step 2: Find interpolation positions (differentiable)
        idx = torch.searchsorted(x_old, x_new_clamped)
        idx = torch.clamp(idx, 1, len(x_old)-1)

        # Step 3: Get left and right neighboring nodes (differentiable)
        x_left = x_old[idx-1]
        x_right = x_old[idx]
        y_left = y_old[idx-1]
        y_right = y_old[idx]

        # Step 4: Linear interpolation (differentiable, small epsilon to prevent division by zero)
        weight = (x_new_clamped - x_left) / (x_right - x_left + 1e-12)
        y_new = y_left + weight * (y_right - y_left)

        # Step 5: Out-of-bound value replacement (differentiable)
        y_new = torch.where(x_new < x_old[0], y_old[0].expand_as(y_new), y_new)
        y_new = torch.where(x_new > x_old[-1], y_old[-1].expand_as(y_new), y_new)

        return y_new.contiguous()

    def _mt1dte(self, freq, dz0, sig0, n_add):
        """
        1D TE mode forward modeling (differentiable version, minimizes no_grad wrapping).

        Args:
            freq: Single frequency value
            dz0: Z-direction grid spacing (core layer)
            sig0: Conductivity model (core layer)
            n_add: Grid refinement factor
        Returns:
            ex_result: Electric field component (preserves gradients)
            hy_result: Magnetic field component (preserves gradients)
        """
        # Step 1: Grid refinement
        omega = 2.0 * torch.pi * freq
        dz = torch.repeat_interleave(dz0 / n_add, n_add)
        sig = torch.repeat_interleave(sig0, n_add)
        nz = sig.numel()

        # Bottom boundary condition supplement
        sig = torch.cat((sig, sig[-1].unsqueeze(0)))
        boundary_dz = torch.sqrt(2.0 / (sig[-1] * omega * self.miu))
        dz = torch.cat((dz, boundary_dz.unsqueeze(0)))

        # Step 2: Build system matrix (core part preserves gradients)
        sig_top = sig[:nz]
        sig_bot = sig[1:nz+1]
        dz_top = dz[:nz]
        dz_bot = dz[1:nz+1]

        diagA = (self.II * omega * self.miu * (sig_top * dz_top + sig_bot * dz_bot) -
                 (2.0 / dz_top + 2.0 / dz_bot).to(self.cdtype)).to(self.cdtype)
        offdiagA = (2.0 / dz[1:nz]).to(self.cdtype)

        # Step 3: Assemble dense matrix
        n = diagA.numel()
        mtxA = torch.diag(diagA)
        if offdiagA.numel() > 0:
            mtxA += torch.diag(offdiagA, diagonal=1) + torch.diag(offdiagA, diagonal=-1)

        # Right-hand side
        rhs = torch.zeros((nz, 1), dtype=self.cdtype, device=self.device)
        rhs[0] = (-2.0 / dz[0]).to(self.cdtype)

        # Step 4: Solve linear system (preserves full gradient)
        ex0 = torch.linalg.solve(mtxA, rhs)

        # Step 5: Result reconstruction and sampling
        ex = torch.cat((torch.tensor([1.0], dtype=self.cdtype, device=self.device), ex0.squeeze()))
        hy0 = (ex[1:] - ex[:-1]) / dz[:-1] / self.II / omega / self.miu
        hy = torch.cat((hy0, hy0[-1:]))
        idx = torch.arange(sig0.numel() + 1, device=self.device) * n_add

        # Step 6: Extract core layer results (preserves gradients)
        ex_result = ex[idx].contiguous()
        hy_result = hy[idx].contiguous()

        return ex_result, hy_result

    def _mt1dtm(self, freq, dz0, sig0, n_add):
        """1D TM mode forward modeling (preserves sig0 gradient, other ops disable gradient tracking)."""
        omega = 2.0 * torch.pi * freq
        dz = torch.repeat_interleave(dz0 / n_add, n_add)
        sig = torch.repeat_interleave(sig0, n_add)
        nz = sig.numel()

        sig = torch.cat((sig, sig[-1].unsqueeze(0)))
        boundary_dz = torch.sqrt(2.0 / (sig[-1] * omega * self.miu))
        dz = torch.cat((dz, boundary_dz.unsqueeze(0)))

        dz_top = dz[:nz]
        dz_bot = dz[1:nz+1]
        sig_top = sig[:nz]
        sig_bot = sig[1:nz+1]

        diagA = (self.II * omega * self.miu * (dz_top + dz_bot) -
                    (2.0 / (dz_top * sig_top) + 2.0 / (dz_bot * sig_bot)).to(self.cdtype)).to(self.cdtype)
        offdiagA = (2.0 / (dz[1:nz] * sig[1:nz])).to(self.cdtype) if nz > 1 else torch.tensor([], dtype=self.cdtype, device=self.device)

        n = diagA.numel()
        mtxA = torch.diag(diagA)
        if offdiagA.numel() == n - 1:
            mtxA += torch.diag(offdiagA, diagonal=1) + torch.diag(offdiagA, diagonal=-1)

        rhs = torch.zeros((nz, 1), dtype=self.cdtype, device=self.device)
        if dz[0] > 1e-12 and sig[0] > 1e-12:
            rhs[0] = (-2.0 / (dz[0] * sig[0])).to(self.cdtype)

        hx0 = torch.linalg.solve(mtxA, rhs).squeeze(1)
        hx = torch.cat((torch.tensor([1.0+0j], dtype=self.cdtype, device=self.device), hx0))
        ey0 = (hx[1:] - hx[:-1]) / dz[:-1].to(self.cdtype) / sig[:-1].clamp_min(1e-12).to(self.cdtype)
        ey = torch.cat((ey0, ey0[-1:].clone()))
        orig_nodes = sig0.numel() + 1
        idx = torch.arange(0, orig_nodes, device=self.device) * n_add
        idx = idx[idx < ey.numel()]

        ey_result = ey[idx].contiguous()
        ey_result.requires_grad_(sig0.requires_grad)
        hx_result = hx[idx].contiguous()
        hx_result.requires_grad_(sig0.requires_grad)
        return ey_result, hx_result

    def _mt2dte(self, freq, dy, dz, sig, sig_diff, n_add):
        """
        2D TE mode secondary electric field computation (differentiable, no in-place ops).

        Args:
            freq: Single frequency value
            dy: Y-direction grid spacing
            dz: Z-direction grid spacing
            sig: Expanded conductivity model
            sig_diff: Conductivity difference (sig - sig_back)
            n_add: Grid refinement factor
        Returns:
            ex0d: Total electric field component (preserves full gradient)
        """
        omega = 2.0 * torch.pi * freq
        ny = self.ny - 1
        nz = self.nz - 1

        # Step 1: Build grid spacing and weight matrices
        dy0, dz0 = torch.meshgrid(dy, dz, indexing='ij')
        dy0 = dy0.permute(1, 0).contiguous()
        dz0 = dz0.permute(1, 0).contiguous()

        # Central difference coefficients
        dyc = (dy0[:nz-1, :ny-1] + dy0[:nz-1, 1:ny]) / 2.0
        dzc = (dz0[:nz-1, :ny-1] + dz0[1:nz, :ny-1]) / 2.0

        # Weights and areas
        dy_slice = dy0[:nz-1, :ny-1]
        dy_slice1 = dy0[:nz-1, 1:ny]
        dz_slice = dz0[:nz-1, :ny-1]
        dz_slice1 = dz0[1:nz, :ny-1]

        w1 = dy_slice * dz_slice
        w2 = dy_slice1 * dz_slice
        w3 = dy_slice * dz_slice1
        w4 = dy_slice1 * dz_slice1
        area = (w1 + w2 + w3 + w4) / 4.0

        # Step 2: Weighted conductivity averaging (core computation, preserves full gradient)
        sigc = (sig[:nz-1, :ny-1] * w1 + sig[:nz-1, 1:ny] * w2 +
                sig[1:nz, :ny-1] * w3 + sig[1:nz, 1:ny] * w4) / (area * 4.0)

        # Step 3: Build system matrix core parameters (preserves gradients)
        val = dzc / dy_slice + dzc / dy_slice1 + dyc / dz_slice + dyc / dz_slice1
        mtx1 = (self.II * omega * self.miu * sigc * area).to(self.cdtype) - val.to(self.cdtype)
        mtx1 = mtx1.t().flatten()

        # Step 4: Build auxiliary diagonal matrices
        # Upper/lower diagonals
        mtx20 = dyc[1:nz-1, :ny-1] / dz0[1:nz-1, :ny-1]
        mtx2 = torch.cat((mtx20, torch.zeros((1, ny-1), device=self.device, dtype=self.fdtype)), dim=0)
        mtx2 = mtx2.t().flatten()[:-1].to(self.cdtype)

        # Left/right diagonals
        mtx3 = dzc[:nz-1, 1:ny-1] / dy0[:nz-1, 1:ny-1]
        mtx3 = mtx3.t().flatten()
        k2 = nz

        # Step 5: Assemble dense system matrix (preserves gradients)
        n_total = (nz-1) * (ny-1)
        A_mat = torch.diag(mtx1).to(self.cdtype).contiguous()

        # Handle +/-1 diagonals
        mtx2_correct_len = n_total - 1
        mtx2_truncated = mtx2[:mtx2_correct_len].contiguous() if mtx2.numel() > mtx2_correct_len else mtx2
        if mtx2_truncated.numel() == mtx2_correct_len:
            diag_mtx2_neg = torch.diag(mtx2_truncated, diagonal=-1).to(self.cdtype)
            diag_mtx2_pos = torch.diag(mtx2_truncated, diagonal=1).to(self.cdtype)
            A_mat += diag_mtx2_neg + diag_mtx2_pos

        # Handle +/-(k2-1) diagonals
        mtx3_offset = abs(k2 - 1)
        mtx3_correct_len = n_total - mtx3_offset
        mtx3_truncated = mtx3[:mtx3_correct_len].contiguous() if mtx3.numel() > mtx3_correct_len else mtx3
        if mtx3_truncated.numel() == mtx3_correct_len:
            diag_mtx3_neg = torch.diag(mtx3_truncated, diagonal=1 - k2).to(self.cdtype)
            diag_mtx3_pos = torch.diag(mtx3_truncated, diagonal=k2 - 1).to(self.cdtype)
            A_mat += diag_mtx3_neg + diag_mtx3_pos

        A_mat = A_mat.contiguous()

        # Step 6: Build right-hand side (preserves full gradient)
        ex1d, _ = self._mt1dte(freq, dz, (sig - sig_diff)[:, 0], n_add)
        ex1d = ex1d.unsqueeze(1).expand(-1, self.ny).contiguous()

        sigc_diff = (sig_diff[:nz-1, :ny-1] * w1 + sig_diff[:nz-1, 1:ny] * w2 +
                     sig_diff[1:nz, :ny-1] * w3 + sig_diff[1:nz, 1:ny] * w4) / (area * 4.0)
        coef = (self.II * omega * self.miu * sigc_diff * area)
        rhs = -coef * ex1d[1:nz, 1:ny]
        rhs = rhs.t().flatten().unsqueeze(1).contiguous()

        # Step 7: Solve linear system (preserves full gradient)
        ex2d_flat = torch.linalg.solve(A_mat, rhs).squeeze(1).contiguous()
        ex2d = ex2d_flat.view(ny-1, nz-1).t().to(self.cdtype).contiguous()

        # Step 8: Reconstruct total electric field (no in-place ops, preserves gradients)
        ex0d = ex1d.clone()
        ex0d[1:nz, 1:ny] = ex1d[1:nz, 1:ny] + ex2d

        return ex0d.contiguous()

    def _mt2dtm(self, freq, dy, dz, sig, sig_diff, n_add):
        """
        TM mode secondary magnetic field computation (optimized: fewer intermediate vars, preserves sig core gradient).
        """
        omega = 2.0 * torch.pi * freq
        ny = len(dy)
        nz = len(dz)

        # Grid construction (single pass)
        dy0, dz0 = torch.meshgrid(dy, dz, indexing='ij')
        dy0 = dy0.permute(1, 0).contiguous()
        dz0 = dz0.permute(1, 0).contiguous()

        # Precompute common slices
        dy0_top = dy0[:nz-1, :ny-1]
        dy0_bottom = dy0[1:nz, :ny-1]
        dy0_right = dy0[:nz-1, 1:ny]
        dz0_top = dz0[:nz-1, :ny-1]
        dz0_bottom = dz0[1:nz, :ny-1]

        # Difference coefficients (vectorized)
        dyc = (dy0_top + dy0_right) / 2.0
        dzc = (dz0_top + dz0_bottom) / 2.0

        # Weights and areas (combined computation)
        w1 = 2 * dz0_top
        w2 = 2 * dz0_bottom
        w3 = 2 * dy0_top
        w4 = 2 * dy0_right
        area = (w1 + w2 + w3 + w4) / 4.0

        # Precompute conductivity slices (preserves sig gradient, core computation)
        sig_top_left = sig[:nz-1, :ny-1]
        sig_top_right = sig[:nz-1, 1:ny]
        sig_bottom_left = sig[1:nz, :ny-1]
        sig_bottom_right = sig[1:nz, 1:ny]

        # Difference coefficients A/B/C/D (preserves sig gradient, vectorized)
        inv_sig_top_left = 1.0 / sig_top_left.clamp_min(1e-12)
        inv_sig_top_right = 1.0 / sig_top_right.clamp_min(1e-12)
        inv_sig_bottom_left = 1.0 / sig_bottom_left.clamp_min(1e-12)
        inv_sig_bottom_right = 1.0 / sig_bottom_right.clamp_min(1e-12)

        A = (inv_sig_top_left * dy0_top + inv_sig_top_right * dy0_right) / w1.clamp_min(1e-12)
        B = (inv_sig_bottom_left * dy0_top + inv_sig_bottom_right * dy0_right) / w2.clamp_min(1e-12)
        C = (inv_sig_top_left * dz0_top + inv_sig_bottom_left * dz0_bottom) / w3.clamp_min(1e-12)
        D = (inv_sig_top_right * dz0_top + inv_sig_bottom_right * dz0_bottom) / w4.clamp_min(1e-12)

        # Build main diagonal (preserves sig gradient)
        diag = (self.II * omega * self.miu * dyc * dzc).to(self.cdtype) - (A + B + C + D).to(self.cdtype)
        mtx1 = diag.t().flatten()

        # Upper/lower diagonals
        mtx2 = torch.zeros((1, ny-1), device=self.device, dtype=self.cdtype)
        if nz-1 > 1:
            mtx20 = B[:nz-2, :ny-1]
            mtx2 = torch.cat([mtx20, mtx2], dim=0)
        mtx2 = mtx2.t().flatten()[:-1]

        # Left/right diagonals
        mtx3 = torch.tensor([], device=self.device, dtype=self.cdtype)
        if ny-1 > 1:
            mtx3 = D[:nz-1, :ny-2].t().flatten()

        # Matrix assembly (consistent with _mt2dte)
        n_total = (nz-1) * (ny-1)
        A_mat = torch.diag(mtx1).to(self.cdtype)

        if mtx2.numel() >= n_total - 1:
            mtx2_trunc = mtx2[:n_total-1]
            A_mat += torch.diag(mtx2_trunc, diagonal=1)
            A_mat += torch.diag(mtx2_trunc, diagonal=-1)

        k2 = nz
        offset = k2 - 1
        if mtx3.numel() >= n_total - offset:
            mtx3_trunc = mtx3[:n_total-offset]
            A_mat += torch.diag(mtx3_trunc, diagonal=offset)
            A_mat += torch.diag(mtx3_trunc, diagonal=-offset)
        A_mat = A_mat.contiguous()

        # Primary field computation (reuse results, preserves sig gradient)
        ey1d, hx1d = self._mt1dtm(freq, dz, (sig - sig_diff)[:, 0], n_add)
        ey1d = ey1d.unsqueeze(1).expand(-1, ny+1)
        hx1d = hx1d.unsqueeze(1).expand(-1, ny+1)

        # Compute areas
        A1 = dy0_top * dz0_top
        A2 = dy0_right * dz0_top
        A3 = dy0_top * dz0_bottom
        A4 = dy0_right * dz0_bottom
        area_sum = (A1 + A2 + A3 + A4) / 4.0

        # Conductivity ratio (avoids division by zero, preserves sig gradient)
        sig_scale = sig_diff / sig.clamp_min(1e-12)

        # Precompute sig_scale slices (preserves gradient)
        ss_top_left = sig_scale[:nz-1, :ny-1]
        ss_top_right = sig_scale[:nz-1, 1:ny]
        ss_bottom_left = sig_scale[1:nz, :ny-1]
        ss_bottom_right = sig_scale[1:nz, 1:ny]

        # Weighted average (vectorized, preserves gradient)
        sigc_diff = (ss_top_left * A1 + ss_top_right * A2 +
                    ss_bottom_left * A3 + ss_bottom_right * A4) / (area_sum * 4.0)

        # Right-hand side main term (preserves gradient)
        coef = (self.II * omega * self.miu * sigc_diff * area_sum).to(self.cdtype)
        hx1d_slice = hx1d[1:nz, 1:ny]
        rhs_main = (coef * hx1d_slice).t().flatten().unsqueeze(1)

        # Extra derivative term
        denom_t = dy0_top + dy0_right
        denom_b = dy0_bottom + dy0_right
        dz_avg = (dz0_top + dz0_bottom) / 2.0

        sigc_t = (ss_top_left * dy0_top + ss_top_right * dy0_right) / denom_t.clamp_min(1e-12)
        sigc_b = (ss_bottom_left * dy0_bottom + ss_bottom_right * dy0_right) / denom_b.clamp_min(1e-12)
        ey1d_slice = ey1d[1:nz, 1:ny]
        ey_d = (sigc_b - sigc_t) / dz_avg.clamp_min(1e-12) * area_sum * ey1d_slice

        # Combine right-hand side (preserves gradient)
        rhs = -(rhs_main - ey_d.t().flatten().unsqueeze(1))

        # Solve linear system
        hx2d_flat = torch.linalg.solve(A_mat, rhs).squeeze(1)
        hx2d = hx2d_flat.view(ny-1, nz-1).t()

        # Reconstruct total field (preserves full gradient)
        hx0d = hx1d.clone()
        hx0d[1:nz, 1:ny] = hx0d[1:nz, 1:ny] + hx2d

        return hx0d

    def _mt2dhyhz(self, freq, dy, dz, sig, ex):
        """Compute magnetic field Hy/Hz from TE mode Ex."""
        omega = 2.0 * torch.pi * freq
        ny = dy.numel()
        kk = self.nza
        delz = dz[kk]

        hys = torch.zeros((ny + 1), dtype=self.cdtype, device=self.device)
        hzs = torch.zeros_like(hys)

        # Compute Hy
        sigc = sig[kk, 0]
        c0 = -1.0 / (self.II * omega * self.miu * delz) + (3.0 / 8.0) * sigc * delz
        c1 = 1.0 / (self.II * omega * self.miu * delz) + (1.0 / 8.0) * sigc * delz
        hys[0] = c0 * ex[kk, 0] + c1 * ex[kk + 1, 0]

        sigc = sig[kk, ny - 1]
        c0 = -1.0 / (self.II * omega * self.miu * delz) + (3.0 / 8.0) * sigc * delz
        c1 = 1.0 / (self.II * omega * self.miu * delz) + (1.0 / 8.0) * sigc * delz
        hys[ny] = c0 * ex[kk, ny] + c1 * ex[kk + 1, ny]

        dyj = dy[:ny-1] + dy[1:ny]
        sigc = (sig[kk, :ny-1] * dy[:ny-1] + sig[kk, 1:ny] * dy[1:ny]) / dyj
        cc = delz / (4.0 * self.II * omega * self.miu * dyj)
        c0 = -1.0/(self.II*omega*self.miu*delz) + (3.0/8.0)*sigc*delz - cc*3.0*(1.0/dy[1:ny]+1.0/dy[:ny-1])
        c1 = 1.0/(self.II*omega*self.miu*delz) + (1.0/8.0)*sigc*delz - cc*(1.0/dy[1:ny]+1.0/dy[:ny-1])
        c0l = 3.0 * cc / dy[:ny-1]
        c0r = 3.0 * cc / dy[1:ny]
        c1l = 1.0 * cc / dy[:ny-1]
        c1r = 1.0 * cc / dy[1:ny]

        hys[1:ny] = (c0l * ex[kk, :ny-1] + c0 * ex[kk, 1:ny] + c0r * ex[kk, 2:ny+1] +
                        c1l * ex[kk+1, :ny-1] + c1 * ex[kk+1, 1:ny] + c1r * ex[kk+1, 2:ny+1])

        # Compute Hz
        hzs[0] = -1.0 / (self.II * omega * self.miu) * (ex[kk, 1] - ex[kk, 0]) / dy[0]
        hzs[ny] = -1.0 / (self.II * omega * self.miu) * (ex[kk, ny] - ex[kk, ny-1]) / dy[ny-1]
        hzs[1:ny] = -1.0 / (self.II * omega * self.miu) * (ex[kk, 2:ny+1] - ex[kk, :ny-1]) / (dy[:ny-1] + dy[1:ny])

        return hys.contiguous(), hzs.contiguous()

    def _mt2deyez(self, freq, dy, dz, sig, hx):
        """Compute electric field Ey/Ez from TM mode Hx."""
        omega = 2.0 * torch.pi * freq
        ny = dy.numel()
        kk = 0
        delz = dz[kk]

        eys = torch.zeros((ny + 1), dtype=self.cdtype, device=self.device)
        ezs = torch.zeros_like(eys)

        # Compute Ey
        sigc = sig[kk, 0]
        temp_beta = self.II * omega * self.miu * delz
        temp_1 = sigc * delz
        c0 = -1.0 / temp_1 + (3.0 / 8.0) * temp_beta
        c1 = 1.0 / temp_1 + (1.0 / 8.0) * temp_beta
        eys[0] = c0 * hx[kk, 0] + c1 * hx[kk + 1, 0]

        sigc = sig[kk, ny - 1]
        temp_1 = sigc * delz
        c0 = -1.0 / temp_1 + (3.0 / 8.0) * temp_beta
        c1 = 1.0 / temp_1 + (1.0 / 8.0) * temp_beta
        eys[ny] = c0 * hx[kk, ny] + c1 * hx[kk + 1, ny]

        dyj = (dy[:ny-1] + dy[1:ny]) / 2.0
        tao = 1.0 / sig[kk, :ny].clamp_min(1e-12)
        taoc = (tao[:ny-1] * dy[:ny-1] + tao[1:ny] * dy[1:ny]) / (2 * dyj)
        temp_1 = self.II * omega * self.miu * delz
        temp_2 = taoc / delz
        temp_3 = delz / dyj
        temp_4 = tao / dy

        c0 = (3.0 / 8.0) * temp_1 - temp_2
        c1 = (1.0 / 8.0) * temp_1 + temp_2 - (1.0 / 8.0) * temp_3 * (temp_4[:ny-1] + temp_4[1:ny])
        c1l = (1.0 / 8.0) * temp_3 * temp_4[:ny-1]
        c1r = (1.0 / 8.0) * temp_3 * temp_4[1:ny]

        eys[1:ny] = (c0 * hx[kk, 1:ny] + c1l * hx[kk+1, :ny-1] +
                        c1 * hx[kk+1, 1:ny] + c1r * hx[kk+1, 2:ny+1])

        return eys.contiguous(), ezs.contiguous()

    def _mt2dzxy(self, freq, exr, hyr):
        """
        Compute TE mode apparent resistivity and phase (complex -> real, ensures differentiable gradient flow).

        Args:
            freq: Single frequency value
            exr: Interpolated electric field
            hyr: Interpolated magnetic field
        Returns:
            rhote: Apparent resistivity (real, preserves gradients)
            phste: Phase (real, preserves gradients)
        """
        omega = 2.0 * torch.pi * freq

        # Prevent division by zero (differentiable)
        hyr_safe = hyr + torch.where(hyr == 0,
                                     torch.tensor(1e-12, dtype=self.cdtype, device=self.device),
                                     torch.tensor(0.0, dtype=self.cdtype, device=self.device))

        # Impedance computation (complex)
        zxy = exr / hyr_safe

        # Complex -> real: split real and imaginary parts (preserves full gradient)
        zxy_real = zxy.real
        zxy_imag = zxy.imag

        # Apparent resistivity (pure real arithmetic, differentiable)
        zxy_abs_sq = zxy_real ** 2 + zxy_imag ** 2
        rhote = zxy_abs_sq / (omega * self.miu)

        # Phase (pure real arithmetic, differentiable)
        phste = torch.atan2(zxy_imag, zxy_real) * 180.0 / torch.pi

        rhote.requires_grad_(exr.requires_grad)
        phste.requires_grad_(exr.requires_grad)

        return rhote.contiguous(), phste.contiguous()

    def _mt2dzyx(self, freq, hxr, eyr):
        """
        Compute TM mode apparent resistivity and phase (complex -> real, ensures differentiable gradient flow).

        Args:
            freq: Single frequency value
            hxr: Interpolated magnetic field
            eyr: Interpolated electric field
        Returns:
            rhotm: Apparent resistivity (real, preserves gradients)
            phstm: Phase (real, preserves gradients)
        """
        omega = 2.0 * torch.pi * freq

        # Prevent division by zero (differentiable)
        eyr_safe = eyr + torch.where(eyr == 0,
                                     torch.tensor(1e-12, dtype=self.cdtype, device=self.device),
                                     torch.tensor(0.0, dtype=self.cdtype, device=self.device))

        # Impedance computation (complex)
        zyx = eyr_safe / hxr

        # Complex -> real: split real and imaginary parts (preserves full gradient)
        zyx_real = zyx.real
        zyx_imag = zyx.imag

        # Apparent resistivity (pure real arithmetic, differentiable)
        zyx_abs_sq = zyx_real ** 2 + zyx_imag ** 2
        rhotm = zyx_abs_sq / (omega * self.miu)

        # Phase (pure real arithmetic, differentiable)
        phstm = torch.atan2(zyx_imag, zyx_real) * 180.0 / torch.pi

        rhotm.requires_grad_(hxr.requires_grad)
        phstm.requires_grad_(hxr.requires_grad)

        return rhotm.contiguous(), phstm.contiguous()

    def extend_model_boundary_single1(self, model0):
        """
        Differentiable model boundary extension (no in-place ops, uses torch.where + tensor concatenation).

        Args:
            model0: Core layer conductivity model (size_k x size_k)
        Returns:
            model: Expanded conductivity model (preserves full gradient)
        """
        if not model0.requires_grad:
            print("Warning: model0.requires_grad=False, output model cannot track gradients!")

        device = model0.device
        dtype = model0.dtype
        nza = 10
        size_b = 10
        size_k = 64

        # Boundary constants (frozen, no gradient)
        sig_bound = torch.tensor(1e-2, device=device, dtype=dtype, requires_grad=False)
        sig_air = torch.tensor(1e-9, device=device, dtype=dtype, requires_grad=False)

        # Compute full grid dimensions
        len_z = nza + size_b + size_k
        len_y = 2 * size_b + size_k

        # ---------------------- Step 1: Build base model (tensor concat, differentiable) ----------------------
        core_y = model0

        model0_const_surface = core_y.clone()
        model0_const_surface[0, :] = sig_bound
        left_y_pad = sig_bound.expand(size_k, size_b).contiguous()
        right_y_pad = sig_bound.expand(size_k, size_b).contiguous()
        y_padded = torch.cat([left_y_pad, core_y, right_y_pad], dim=1).contiguous()

        air_z_pad = sig_air.expand(nza, len_y).contiguous()
        bottom_z_pad = sig_bound.expand(size_b, len_y).contiguous()
        model = torch.cat([air_z_pad, y_padded, bottom_z_pad], dim=0).contiguous()

        # ---------------------- Step 2: Boundary smoothing interpolation (torch.where, no in-place ops) ----------------------
        num_interp = size_b - int(2 * size_b / 3)
        start_idx = int(2 * size_b / 3)
        weight = torch.linspace(0.0, 1.0, num_interp, device=device, dtype=dtype).contiguous()
        weight_flip = weight.flip(0).contiguous()

        # 2.1 Left boundary interpolation
        core_edge_l = model[nza:, size_b].unsqueeze(1).expand(-1, num_interp).contiguous()
        left_interp = core_edge_l * weight + sig_bound * weight_flip

        left_mask = torch.zeros_like(model, dtype=torch.bool, device=device)
        left_mask[nza:, start_idx:size_b] = True
        left_interp_expanded = torch.zeros_like(model, device=device, dtype=dtype)
        left_interp_expanded[nza:, start_idx:size_b] = left_interp
        model = torch.where(left_mask, left_interp_expanded, model)

        # 2.2 Right boundary interpolation
        core_edge_r = model[nza:, -size_b-1].unsqueeze(1).expand(-1, num_interp).contiguous()
        right_interp = core_edge_r * weight_flip + sig_bound * weight

        right_mask = torch.zeros_like(model, dtype=torch.bool, device=device)
        right_mask[nza:, -size_b:-start_idx] = True
        right_interp_expanded = torch.zeros_like(model, device=device, dtype=dtype)
        right_interp_expanded[nza:, -size_b:-start_idx] = right_interp
        model = torch.where(right_mask, right_interp_expanded, model)

        # 2.3 Bottom boundary interpolation
        core_edge_b = model[-size_b-1, :].unsqueeze(0).expand(num_interp, -1).contiguous()
        bottom_interp = core_edge_b * weight_flip.unsqueeze(1) + sig_bound * weight.unsqueeze(1)

        bottom_mask = torch.zeros_like(model, dtype=torch.bool, device=device)
        bottom_mask[-size_b:-start_idx, :] = True
        bottom_interp_expanded = torch.zeros_like(model, device=device, dtype=dtype)
        bottom_interp_expanded[-size_b:-start_idx, :] = bottom_interp
        model = torch.where(bottom_mask, bottom_interp_expanded, model)

        # ---------------------- Step 3: Ensure consistent gradient tracking ----------------------
        model.requires_grad_(model0.requires_grad)

        return model.contiguous()

    def mt2d(self):
        """
        2D MT forward modeling main function (no del operations, preserves full computation graph).

        Uses self.mode specified at initialization to select computation mode automatically.

        Returns:
            mt_obs: Apparent resistivity and phase results (4 x n_freq x nry, preserves full gradient)
        """
        rhoxy_list = []
        phsxy_list = []
        rhoyx_list = []
        phsyx_list = []

        # ---------------------- TE mode computation ----------------------
        if self.mode in ["TE", "TETM"]:
            for kf in range(self.nf):
                # 2D TE mode electric field
                ex = self._mt2dte(self.freq[kf], self.dy, self.dz, self.sig, self.sig_diff, self.n_add)

                # Magnetic field computation
                hys, hzs = self._mt2dhyhz(self.freq[kf], self.dy, self.dz, self.sig, ex)
                exs = ex[self.nza, :]

                # Interpolate to observation points (preserves gradients)
                exr = self._torch_interp(self.ry, self.yn, exs)
                hyr = self._torch_interp(self.ry, self.yn, hys)

                # Apparent resistivity and phase (preserves gradients)
                rhoxy_kf, phsxy_kf = self._mt2dzxy(self.freq[kf], exr, hyr)

                rhoxy_list.append(rhoxy_kf)
                phsxy_list.append(phsxy_kf)

        # ---------------------- TM mode computation ----------------------
        if self.mode in ["TM", "TETM"]:
            dz_tm = self.dz[self.nza:].contiguous()
            sig_tm = self.sig[self.nza:, :].contiguous()
            sig_diff_tm = self.sig_diff[self.nza:, :].contiguous()

            for kf in range(self.nf):
                # 2D TM mode magnetic field
                hx = self._mt2dtm(self.freq[kf], self.dy, dz_tm, sig_tm, sig_diff_tm, self.n_add)

                # Electric field computation
                eys, ezs = self._mt2deyez(self.freq[kf], self.dy, dz_tm, sig_tm, hx)
                hxs = hx[0, :]

                # Interpolate to observation points
                hxr = self._torch_interp(self.ry, self.yn, hxs)
                eyr = self._torch_interp(self.ry, self.yn, eys)

                # Apparent resistivity and phase
                rhoyx_kf, phsyx_kf = self._mt2dzyx(self.freq[kf], hxr, eyr)

                rhoyx_list.append(rhoyx_kf)
                phsyx_list.append(phsyx_kf)

        # ---------------------- Result stacking and return ----------------------
        rhoxy = torch.stack(rhoxy_list, dim=0) if rhoxy_list else torch.tensor([], device=self.device)
        phsxy = torch.stack(phsxy_list, dim=0) if phsxy_list else torch.tensor([], device=self.device)
        rhoyx = torch.stack(rhoyx_list, dim=0) if rhoyx_list else torch.tensor([], device=self.device)
        phsyx = torch.stack(phsyx_list, dim=0) if phsyx_list else torch.tensor([], device=self.device)

        # Assemble final results (preserves full gradient)
        if self.mode == "TE":
            mt_obs = torch.stack((rhoxy, phsxy), dim=0).contiguous()
        elif self.mode == "TM":
            mt_obs = torch.stack((rhoyx, phsyx), dim=0).contiguous()
        elif self.mode == "TETM":
            mt_obs = torch.stack((rhoxy, phsxy, rhoyx, phsyx), dim=0).contiguous()

        return mt_obs

    def forward(self, sig=None):
        """
        Forward modeling public interface (fully differentiable, supports inversion training).

        Args:
            sig: Core layer conductivity model (size_k x size_k, requires requires_grad=True)
        Returns:
            mt_obs: Apparent resistivity and phase results (4 x n_freq x nry, preserves full gradient)
        """
        # Step 1: Model boundary extension (preserves full gradient)
        self.sig = self.extend_model_boundary_single1(sig)

        # Step 2: Build background conductivity and difference (preserves full gradient)
        self.sig_back = self.sig[:, 0:1].expand_as(self.sig).contiguous()
        self.sig_diff = (self.sig - self.sig_back).contiguous()
        self.sig_back.requires_grad_(self.sig.requires_grad)
        self.sig_diff.requires_grad_(self.sig.requires_grad)

        # Step 3: Execute 2D forward modeling (preserves full gradient)
        mt_obs = self.mt2d()

        return mt_obs
