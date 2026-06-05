import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser
from datetime import datetime
import os
import torch.nn as nn
import src.utils.pytorch_ssim as pytorch_ssim
from src.MT_forwardSolver import MT2DFD1
from src.inversion import run_inversion, Regularization_method
from src.utils import data_trans, data_vis
from accelerate import Accelerator
from torch.optim import Adam
import pickle

# Set computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize loss functions
ssim_loss = pytorch_ssim
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

def main(regularization, lr, ts, sigma, loss_type, noise_std, initial_type, missing_number, method, reg_lambda, mode, freq_n):
    """
    Main function: execute MT inversion core pipeline.

    Args:
        regularization: Regularization type ('l2'/'tv'/None)
        lr: Learning rate for the inversion optimizer
        ts: Number of inversion iteration steps
        sigma: Standard deviation for Gaussian smoothing of initial model
        loss_type: Loss function type ('l1'/'l2'/'ssim')
        noise_std: Standard deviation of Gaussian noise added to MT data
        initial_type: Type of initial resistivity model
        missing_number: Number of missing MT traces
        method: Experiment method identifier
        reg_lambda: Weight coefficient for the regularization term
        mode: MT forward modeling mode ('TE'/'TM'/'TETM')
        freq_n: Number of frequency grid points
    """
    ctx = {
        'z': 75e3,            # Core region depth (75 km)
        'y': 75e3,            # Core region horizontal extent (-75 km to 75 km)
        'size_k': 64,         # Number of core region grid cells
        'size_b': 10,          # Number of boundary layer grid cells
        'total_freq': 16,     # Total frequency grid points
        'n_freq': freq_n,     # Frequency grid points used
        'nza': 10,            # Number of air layer grid cells
        'n_add': 5            # Number of 1D field interpolation points
    }

    # Initialize MT forward solver
    mt_forward = MT2DFD1(
        ctx=ctx,
        device=device,
        normalize=True,
        mode=mode
    )

    dir_identifier = regularization if regularization else 'pure'
    args_str = f"{mode}_{dir_identifier}_lr{lr}_ts{ts}_sigma{sigma}_loss{loss_type}_noisestd{noise_std}_missing{missing_number}"

    # Define data paths
    test_base_dir = "dataset/Test_Data/MT_Data_Test/"
    family_name_list = [file for file in os.listdir(test_base_dir) if file.endswith('.npz')]

    for family_name in family_name_list:
        # Build full path for current file
        test_dir = os.path.join(test_base_dir, family_name)
        # Load MT data and true resistivity model
        mt_data = (np.load(test_dir)['obs_data'])
        resistivity_data = np.load(test_dir)['sig_model']
        print(f"There are {mt_data.shape[0]} data in the file")

        for j in range(7, 8):
            # Extract single sample and convert to PyTorch tensor
            mt_slice = torch.from_numpy((mt_data[j:j+1])).float().to(device)
            res_slice = torch.from_numpy(resistivity_data[j:j+1]).float().to(device)
            initial_model = data_trans.prepare_initial_model1(res_slice, initial_type='smoothed', sigma=sigma)

            # Initialize inversion class
            Inversion = run_inversion(data_trans, data_vis, ssim_loss, regularization)

            # Run inversion sampling
            # Returns: mu = recovered resistivity model, final_results = intermediate metrics
            mu, final_results, predicted_mt = Inversion.sample(
                initial_model,
                res_slice,
                mt_slice,
                ts,
                lr,
                reg_lambda,
                mt_forward,
                loss_type,
                noise_std,
                missing_number,
                regularization,
                mode=mode,
                freq_n=freq_n
            )

            # Create timestamped output directory
            results_dir = f"experiment/{args_str}"
            os.makedirs(results_dir, exist_ok=True)

            family_results_dir = os.path.join(results_dir, family_name.replace('.npz', ''))
            os.makedirs(family_results_dir, exist_ok=True)

            # Save inversion results as pickle file
            with open(os.path.join(family_results_dir, f'{j}_results.pkl'), 'wb') as f:
                pickle.dump({'mu': mu.detach().cpu().numpy(), 'final_results': final_results, 'predicted_mt': predicted_mt.detach().cpu().numpy()}, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--regularization", default='l2', help="Regularization type ('tv'/'l2')")
    parser.add_argument("--lr", type=float, default=0.05, help="Inversion optimizer learning rate")
    parser.add_argument("--ts", type=int, default=1000, help="Number of inversion iteration steps")
    parser.add_argument("--sigma", type=float, default=10, help="Gaussian smoothing std for initial model")
    parser.add_argument("--loss_type", type=str, default='l1', help="Loss function type ('l1'/'l2'/'ssim')")
    parser.add_argument("--noise_std", type=float, default=0, help="Gaussian noise std added to MT data")
    parser.add_argument("--missing_number", type=int, default=0, help="Number of missing MT stations")
    parser.add_argument("--initial_type", type=str, default='smoothed', help="Initial model type (e.g., 'smoothed')")
    parser.add_argument("--method", type=str, default='admt', help="Experiment method identifier")
    parser.add_argument("--reg_lambda", type=float, default=1, help="Regularization weight coefficient")
    parser.add_argument("--mode", type=str, default='TETM', help="MT forward modeling mode (e.g., 'TETM')")
    parser.add_argument("--n_freq", type=int, default=16, help="Number of frequency grid points")
    args = parser.parse_args()

    regularization_list = ['l2', 'tv']
    noise_std_list = [0,5,10]
    for reg in regularization_list:
        print(f"\n==================== Starting all experiments for regularization: {reg} ====================")
        for noise_std in noise_std_list:
            print(f"\n---------- Running: regularization={reg}, noise_std={noise_std:.2f} ----------")
            main(
                regularization=reg,
                lr=args.lr,
                ts=args.ts,
                sigma=args.sigma,
                loss_type=args.loss_type,
                noise_std=noise_std,
                initial_type=args.initial_type,
                missing_number=args.missing_number,
                method=args.method,
                reg_lambda=args.reg_lambda,
                mode=args.mode,
                freq_n=args.n_freq
            )
            print(f"---------- Completed: regularization={reg}, noise_std={noise_std:.2f} ----------")

    print("\n==================== All experiment combinations completed ====================")


# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# im1 = axs[1].imshow((mu[0, 0, ...].detach().cpu().numpy()),vmin=mu_true.min(),vmax=1, cmap='jet')
# im0 = axs[0].imshow((mu_true[0, 0, ...].detach().cpu().numpy()),vmin=mu_true.min(),vmax=1, cmap='jet')
# cbar = fig.colorbar(im0, orientation='horizontal', shrink=0.5, pad=0.12)
# cbar = fig.colorbar(im1, orientation='horizontal', shrink=0.5, pad=0.12)