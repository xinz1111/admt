# ADMT: An Intelligent 2D Magnetotelluric Inversion Framework Based on Automatic Differentiation Workflow

A physics-driven 2D magnetotelluric (MT) inversion framework. By embedding a differentiable finite-difference forward solver into an optimization loop, it achieves end-to-end resistivity model inversion without surrogate networks.

## Project Structure

```
├── main_admt_complex_medium.py   # Main entry point: run MT inversion experiments
├── Visualization_models.ipynb    # Jupyter Notebook: visualize and compare inversion results
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── inversion.py              # Inversion with AdamW + cosine annealing scheduler
│   ├── inversion_lbgfs.py        # Inversion with L-BFGS optimizer (currently active)
│   ├── MT_forwardSolver.py       # Differentiable 2D MT finite-difference forward solver
│   ├── benchmark.py              # Regularization losses (TV, Tikhonov)
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── data_trans.py         # Data preprocessing: normalization, noise, missing traces
│       ├── data_vis.py           # Matplotlib visualization utilities
│       └── pytorch_ssim.py       # SSIM structural similarity loss / metric
│
├── dataset/                      # Datasets and test data
│   ├── MT_Data/                  # Synthetic resistivity model generation
│   │   ├── gaussian_random_fields.py  # Gaussian random field generator
│   │   ├── main_forward.ipynb    # Forward modeling notebook
│   │   └── images/               # Model comparison figures (PNG/EPS)
│   └── Test_Data/                # Test data for inversion
│       └── MT_Data_Test/         # .npz files (observed data + true models)
│
└── experiment/                   # Experiment output
    └── complex/                  # Results on complex models (timestamped subdirectories)
```

## Directory Descriptions

### `src/` — Core Source Code

| File | Description |
|------|-------------|
| `MT_forwardSolver.py` | Differentiable 2D MT finite-difference forward solver (`MT2DFD1` class), supporting TE/TM modes. Performs mesh construction, linear system solving, and apparent resistivity / phase computation entirely within a PyTorch computation graph |
| `inversion.py` | Inversion optimizer based on AdamW + cosine annealing learning rate scheduling (`run_inversion` class) |
| `inversion_lbgfs.py` | Inversion optimizer based on L-BFGS + Strong-Wolfe line search. More memory-efficient |
| `benchmark.py` | Regularization loss functions: Total Variation (TV), Tikhonov (L2), and joint TV+L2 |
| `utils/data_trans.py` | Data transformation: log-normal encoding/decoding, Gaussian noise injection, random trace missing simulation, initial model generation |
| `utils/data_vis.py` | Visualization helpers: plotting resistivity models and MT responses |
| `utils/pytorch_ssim.py` | SSIM structural similarity loss for evaluating structural consistency between inversion results and ground truth |

### `dataset/` — Datasets

| Subdirectory | Description |
|--------------|-------------|
| `MT_Data/` | Synthetic data generation. `gaussian_random_fields.py` generates synthetic resistivity models via Gaussian random fields; `images/` stores generated comparison figures |
| `Test_Data/` | Inversion test data. `MT_Data_Test/` contains `.npz` files with observed MT data and corresponding true resistivity models |

### `experiment/` — Experiment Results

| Subdirectory | Description |
|--------------|-------------|
| `complex/` | Results for complex geological models. Each run creates a timestamped subdirectory containing recovered resistivity models and optimization metrics (pickle format) |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main_admt_complex_medium.py \
    --regularization tv \
    --lr 0.001 \
    --ts 200 \
    --loss_type l2 \
    --noise_std 0.1 \
    --mode TETM \
    --reg_lambda 0.01
```

### Key Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `--regularization` | Regularization type | `l2`, `tv`, `None` |
| `--lr` | Learning rate | float |
| `--ts` | Total optimization steps | int |
| `--loss_type` | Data misfit loss | `l1`, `l2`, `Huber`, `ssim` |
| `--noise_std` | MT data noise standard deviation | float |
| `--mode` | MT forward modeling mode | `TE`, `TM`, `TETM` |
| `--reg_lambda` | Regularization weight | float |

## Method Overview

This project implements a **physics-driven deep learning inversion** approach. The key idea is embedding a fully differentiable MT finite-difference forward solver (`MT2DFD1`) into a PyTorch optimization loop:

1. **Model Initialization**: Log-normalize the initial resistivity model to [-1, 1]
2. **Forward Modeling**: De-normalize and feed into the MT forward solver to obtain predicted apparent resistivity and phase
3. **Loss Computation**: Total loss = data misfit + λ × regularization (TV or Tikhonov)
4. **Gradient Backpropagation**: Backpropagate through the full finite-difference solver to update the resistivity model
5. **Iterative Refinement**: L-BFGS or AdamW optimizer progressively refines the model

Since the forward solver is fully differentiable, gradients flow end-to-end from the data misfit all the way back to the resistivity model parameters — no surrogate network training required.
