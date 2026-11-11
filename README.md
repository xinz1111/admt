# ADMT: A Two-Dimensional Magnetotelluric Inversion Framework Based on Automatic Differentiation Workflow

A physics-driven 2D magnetotelluric (MT) inversion framework. By embedding a differentiable finite-difference forward solver into an optimization loop, it achieves end-to-end resistivity model inversion without surrogate networks.

## Project Structure

```
ADMT-main/
├── .vscode/
│   └── settings.json              # VS Code workspace settings
├── main.py                        # Main entry point: builds the forward solver and runs inversions
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── CHANGES_CN.txt                 # Revision notes for this release (Chinese)
├── LICENSE                        # License file
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── inversion.py               # Inversion loop (AdamW + cosine annealing) and regularization factory
│   ├── MT_forwardSolver.py        # Differentiable 2D MT finite-difference forward solver
│   ├── benchmark.py               # Regularization losses (TV, Tikhonov, joint TV+L2)
│   ├── __pycache__/               # Python bytecode cache
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── data_trans.py          # Data preprocessing: log normalization, noise, missing traces, initial models
│       ├── data_vis.py            # Matplotlib visualization utilities
│       ├── pytorch_ssim.py        # SSIM structural similarity loss / metric
│       └── __pycache__/           # Python bytecode cache
│
└── dataset/                       # Datasets and test data
    ├── MT_Data/                   # Synthetic resistivity/conductivity model generation
    │   ├── gaussian_random_fields.py  # Gaussian random field generator
    │   ├── main_forward.ipynb     # Model generation + forward modeling notebook
    │   ├── __pycache__/           # Python bytecode cache
    │   └── images/                # Model comparison figures (model_compare_000–004, PNG/EPS)
    └── Test_Data/                 # Test data for inversion
        ├── README.md
        ├── obs_modelanddata_64_10_new.npz        # 10-sample synthetic dataset
        ├── obs_modelanddata_regula_64_1.npz      # 1-sample regularized dataset
        └── MT_Data_Test/
            └── obs_modelanddata_64_10_new.npz    # Dataset actually read by main.py
```

The `__pycache__/` directories contain Python bytecode and can be safely deleted. `experiment/` is **not part of the source tree**; it is created at runtime by `main.py` to store inversion results.

## Directory Descriptions

### Root Files

| File | Description |
|------|-------------|
| `main.py` | Command-line entry point: builds the forward solver and runs inversions over `dataset/Test_Data/MT_Data_Test/*.npz` |
| `requirements.txt` | Python dependencies |
| `CHANGES_CN.txt` | Revision notes for this release (Chinese) |
| `LICENSE` | License file |
| `.vscode/settings.json` | VS Code workspace settings (uses conda as the default Python environment/package manager) |

### `src/` — Core Source Code

| File | Description |
|------|-------------|
| `MT_forwardSolver.py` | Differentiable 2D MT finite-difference forward solver (`MT2DFD1` class), supporting TE/TM/TETM modes. Performs mesh construction, linear system solving, boundary extension, and apparent resistivity / phase computation entirely within a PyTorch computation graph |
| `inversion.py` | `run_inversion` optimization loop (AdamW + cosine annealing), `ResultsDict` metric accumulation, and the `Regularization_method` factory. After the last update it runs one extra forward pass so the returned response matches the returned model |
| `benchmark.py` | Regularization loss functions: Total Variation (TV), Tikhonov (L2), and joint TV+L2 |
| `utils/data_trans.py` | Data transformation: log10 encode/decode to `[-1, 1]`, Gaussian noise injection, random missing-trace simulation, initial-model generation (`smoothed` / `homogeneous` / `linear` / `layered`) |
| `utils/data_vis.py` | Visualization helpers: plotting resistivity models and MT responses |
| `utils/pytorch_ssim.py` | SSIM structural similarity loss for evaluating structural consistency between inversion results and ground truth |

### `dataset/` — Datasets

| Path | Description |
|------|-------------|
| `MT_Data/gaussian_random_fields.py` | Gaussian random field generator used to synthesize conductivity structures |
| `MT_Data/main_forward.ipynb` | Notebook that builds the non-uniform grid, generates random conductivity models, runs forward modeling, and saves comparison figures to `MT_Data/images/` |
| `MT_Data/images/` | Pre-generated model comparison figures (`model_compare_*.png` / `.eps`) |
| `Test_Data/MT_Data_Test/` | Test data directory scanned by `main.py` for `*.npz` files |
| `Test_Data/*.npz` | Two additional datasets kept at the `Test_Data/` level; `MT_Data_Test/` holds the copy that `main.py` actually reads |

#### `.npz` data format

Each file contains two arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `sig_model` | `(N, 1, 64, 64)` | Core-region conductivity models, one per sample |
| `obs_data` | `(N, 4, 16, 65)` | Observed MT responses: channels = TE/TM apparent resistivity + phase, 16 frequencies, 65 stations |

### `experiment/` — Experiment Results (generated at runtime)

Each run writes to a directory named from the experiment configuration:

```
experiment/{mode}_{regularization}_lr{lr}_ts{ts}_sigma{sigma}_loss{loss_type}_noisestd{noise_std}_missing{missing_number}/
    {npz_filename_without_suffix}/
        {sample_index}_results.pkl
```

`{regularization}` is replaced by `pure` when no regularization is used. Each `.pkl` stores a dict with:

| Key | Description |
|-----|-------------|
| `mu` | Returned final model in normalized log10 representation |
| `final_results` | Per-step histories (`total_losses`, `obs_losses`, `reg_losses`, `ssim`, `mae`, `rmse`) plus `final_state` |
| `predicted_mt` | Forward response recomputed for the returned final model |
| `final_results['final_state']` | `iteration`, `obs_loss`, `reg_loss`, `total_loss`, `normalization_max_log`, `model_representation` |

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch>=1.12.0`, `numpy>=1.21.0`, `scipy>=1.7.0`, `matplotlib>=3.5.0`, `tqdm>=4.60.0`, `accelerate>=0.15.0`. A CUDA-capable GPU is used automatically when available.

## Usage

```bash
python main.py \
    --lr 0.05 \
    --ts 200 \
    --loss_type l1 \
    --mode TETM \
    --reg_lambda 1 \
    --n_freq 16
```

`main.py` builds a 64×64 core model with 10 boundary and 10 air cells, 16 frequency points, then iterates over every `.npz` file in `dataset/Test_Data/MT_Data_Test/` and runs the inversion.

Note on the current entry point: the `__main__` block sweeps `regularization ∈ {'l2', 'tv'}` × `noise_std ∈ {0, 5, 10}` with these values hard-coded, and inverts sample index `7` of each file (`for j in range(7, 8)`). Running `python main.py` therefore executes the full grid regardless of `--regularization` / `--noise_std`. To run a single configuration, call `main()` directly or edit that block.

### Key Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--regularization` | Regularization type | `l2` | `l2`, `tv`, `None` |
| `--lr` | Learning rate | `0.05` | float |
| `--ts` | Total optimization steps | `200` | int |
| `--sigma` | Gaussian smoothing std for the initial model | `10` | float |
| `--loss_type` | Data misfit loss | `l1` | `l1`, `l2`, `Huber`, `ssim` |
| `--noise_std` | Gaussian noise std added to MT data | `0` | float |
| `--missing_number` | Number of missing MT stations | `0` | int |
| `--initial_type` | Initial model type | `smoothed` | `smoothed`, `homogeneous`, `linear`, `layered` |
| `--method` | Experiment method identifier | `admt` | str |
| `--reg_lambda` | Regularization weight | `1` | float |
| `--mode` | MT forward modeling mode | `TETM` | `TE`, `TM`, `TETM` |
| `--n_freq` | Number of frequency points used | `16` | int |

When calling `run_inversion.sample()` directly, its own defaults are `loss_type='l2'`, `lr=0.001`, `reg_lambda=0.01`; pass explicit values to reproduce the `main.py` configuration.

## Method Overview

This project implements a **physics-driven deep learning inversion** approach. The key idea is embedding a fully differentiable MT finite-difference forward solver (`MT2DFD1`) into a PyTorch optimization loop:

1. **Model Initialization**: Generate the initial model from the true model (Gaussian smoothing by default) and reparameterize it as normalized log10 resistivity in `[-1, 1]`
2. **Forward Modeling**: Decode to conductivity and feed into the MT forward solver to obtain predicted apparent resistivity and phase
3. **Loss Computation**: Total loss = data misfit + λ × regularization (TV or Tikhonov), with the model clamped to `[-1, 1]`
4. **Gradient Backpropagation**: Backpropagate through the full finite-difference solver to update the model parameters
5. **Iterative Refinement**: AdamW with cosine annealing progressively refines the model; a final forward pass recomputes the response of the returned model

Since the forward solver is fully differentiable, gradients flow end-to-end from the data misfit all the way back to the model parameters — no surrogate network training required.

## Revision Notes

See `CHANGES_CN.txt` for the changes in this release, including the final forward pass, the final-state loss record, and the corresponding paper revisions.
