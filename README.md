# ADMT: An Intelligent 2D Magnetotelluric Inversion Framework Based on Automatic Differentiation Workflow

A physics-driven 2D magnetotelluric (MT) inversion framework. By embedding a differentiable finite-difference forward solver into an optimization loop, it achieves end-to-end resistivity model inversion without surrogate networks.

## Project Structure

```
├── main.py                       # Main entry point: run MT inversion experiments
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # License file
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── inversion.py              # Inversion with AdamW + cosine annealing scheduler
│   ├── MT_forwardSolver.py       # Differentiable 2D MT finite-difference forward solver
│   ├── benchmark.py              # Regularization losses (TV, Tikhonov, joint TV+L2)
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
│       └── MT_Data_Test/         # .npz file (observed data + true models)
│
└── experiment/                   # Experiment output (created at runtime)
    └── <run_tag>/                # One directory per run, named by hyperparameters
        └── <npz_name>/           # One subdirectory per test-data file
            └── <sample>_results.pkl  # Pickled results for one sample
```

## Directory Descriptions

### `src/` — Core Source Code

| File | Description |
|------|-------------|
| `MT_forwardSolver.py` | Differentiable 2D MT finite-difference forward solver (`MT2DFD1` class), supporting TE/TM modes. Performs mesh construction, linear system solving, and apparent resistivity / phase computation entirely within a PyTorch computation graph |
| `inversion.py` | Inversion optimizer based on AdamW + cosine annealing learning rate scheduling (`run_inversion` class) |
| `benchmark.py` | Regularization loss functions: Total Variation (TV), Tikhonov (L2), and joint TV+L2 |
| `utils/data_trans.py` | Data transformation: log-normal encoding/decoding, Gaussian noise injection, random trace missing simulation, initial model generation |
| `utils/data_vis.py` | Visualization helpers: plotting resistivity models and MT responses |
| `utils/pytorch_ssim.py` | SSIM structural similarity loss for evaluating structural consistency between inversion results and ground truth |

### `dataset/` — Datasets

| Subdirectory | Description |
|--------------|-------------|
| `MT_Data/` | Synthetic data generation. `gaussian_random_fields.py` generates synthetic resistivity models via Gaussian random fields; `images/` stores generated comparison figures |
| `Test_Data/` | Inversion test data. `MT_Data_Test/obs_modelanddata_64_10_new.npz` contains observed MT data (`obs_data`) and the corresponding true resistivity models (`sig_model`). `main.py` currently inverts the sample at index `7` |

### `experiment/` — Experiment Results

| Subdirectory | Description |
|--------------|-------------|
| `<run_tag>/` | One directory per run, named `{mode}_{reg}_lr{lr}_ts{ts}_sigma{sigma}_loss{loss}_noisestd{noise_std}_missing{missing}` (e.g. `TETM_l2_lr0.005_ts1000_sigma10_lossl1_noisestd5_missing0`) |
| `<npz_name>/<sample>_results.pkl` | Pickled results for one sample, containing `mu` (recovered resistivity model), `final_results` (per-step loss/metric history), and `predicted_mt` (predicted MT response) |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
    --lr 0.005 \
    --ts 1000 \
    --loss_type l2 \
    --noise_std 0.1 \
    --mode TETM \
    --reg_lambda 0.01
```

> **Note:** `main.py` currently runs both `l2` and `tv` regularization sequentially in a single invocation (see the `regularization_list` loop). The `--regularization` flag is accepted for backward compatibility but does not select the run.

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--regularization` | Regularization type placeholder (accepted but **not used**; `l2` and `tv` are both run) | `l2` |
| `--lr` | Learning rate for the AdamW optimizer | `0.005` |
| `--ts` | Total optimization steps | `1000` |
| `--sigma` | Gaussian smoothing std for the initial model | `10` |
| `--loss_type` | Data misfit loss: `l1`, `l2`, `Huber`, `ssim` | `l1` |
| `--noise_std` | Gaussian noise std added to MT data | `5` |
| `--missing_number` | Number of missing MT traces | `0` |
| `--initial_type` | Initial model type (accepted but **not used**; `smoothed` is hardcoded) | `smoothed` |
| `--method` | Experiment method identifier (accepted but **not used**) | `admt` |
| `--reg_lambda` | Regularization weight coefficient | `1` |
| `--mode` | MT forward modeling mode: `TE`, `TM`, `TETM` | `TETM` |
| `--n_freq` | Number of frequency points used | `16` |

## Method Overview

This project implements a **physics-driven deep learning inversion** approach. The key idea is embedding a fully differentiable MT finite-difference forward solver (`MT2DFD1`) into a PyTorch optimization loop:

1. **Model Initialization**: Convert the resistivity model to conductivity (`1/ρ`), apply log-normalization, and rescale it to `[-1, 1]`
2. **Forward Modeling**: De-normalize the model and feed it into the MT forward solver to obtain predicted apparent resistivity and phase
3. **Loss Computation**: Total loss = data misfit + λ × regularization (TV or Tikhonov)
4. **Gradient Backpropagation**: Backpropagate through the full finite-difference solver to update the model
5. **Iterative Refinement**: AdamW (`weight_decay=1e-2`) with a cosine-annealing learning-rate scheduler progressively refines the model

Since the forward solver is fully differentiable, gradients flow end-to-end from the data misfit all the way back to the resistivity model parameters — no surrogate network training required.
