import logging
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import load_sdf_files
from molecule_utils import center_molecule, compute_local_features, voxelize_molecule, compute_molecular_features
from data_loader import create_data_loaders
from model import Ligand3DVAE
from model_pred import main_prediction
from ANN import create_molecular_data_loaders, MolecularVAE, vae_loss, train_molecular_vae
from visualizations import analyze_distances, visualize_similar_and_dissimilar_pairs, plotly_voxel_full_view, plotly_voxel_nonzero_only, plotly_voxel_pair_comparison
from validate import validate_sdf_files, check_existing_voxel_files_and_missing_sdf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import optuna
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def vae_loss_ligand(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(kl_div) or torch.isinf(kl_div):
        logger.warning("NaN/Inf in KL divergence")
        kl_div = torch.tensor(0.0, device=mu.device)
    return recon_loss + beta * kl_div, recon_loss.item(), kl_div.item()

def main():
    try:
        data_dir = "../data/training_data"
        reference_dir = "../data/reference"
        library_dir = "../data/library"
        voxel_output_dir = "../temp/voxel_grids"
        train_voxel_dir = os.path.join(voxel_output_dir, "train")
        val_voxel_dir = os.path.join(voxel_output_dir, "val")
        results_output_dir = "../output/results"
        model_path = "../output/ligand_autoencoder.pth"
        vae_model_path = "../output/molecular_vae.pth"
        train_features_path = os.path.join(voxel_output_dir, "train_molecular_features.pkl")
        val_features_path = os.path.join(voxel_output_dir, "val_molecular_features.pkl")

        logger.info("Starting molecule data processing")

        bad_files = validate_sdf_files(data_dir)
        if bad_files:
            logger.warning(f"Skipping {len(bad_files)} invalid SDF files: {bad_files}")
        else:
            logger.info("No invalid SDF files detected")

        # Check for missing voxel files corresponding to SDF files
        voxel_done, sdf_to_process, all_voxel_files = check_existing_voxel_files_and_missing_sdf(data_dir, train_voxel_dir, val_voxel_dir)

        if voxel_done:
            logger.info("All .sdf files already processed into voxel grids. Skipping preprocessing.")
            # Load existing voxel files for training and validation
            train_voxel_files = sorted(glob.glob(os.path.join(train_voxel_dir, "train_voxel_*.npy")))
            val_voxel_files = sorted(glob.glob(os.path.join(val_voxel_dir, "val_voxel_*.npy")))
            train_sdf_filenames = [os.path.splitext(os.path.basename(f))[0].replace("train_voxel_", "") + ".sdf" for f in train_voxel_files]
            val_sdf_filenames = [os.path.splitext(os.path.basename(f))[0].replace("val_voxel_", "") + ".sdf" for f in val_voxel_files]
        else:
            logger.info(f"Found {len(sdf_to_process)} .sdf files that need to be voxelized.")
            # Load only unprocessed sdf files
            molecules = load_sdf_files(data_dir)
            train_molecules, val_molecules = train_test_split(molecules, test_size=0.2, random_state=42)

            train_centered_molecules = []
            train_failed = []
            train_sdf_filenames = []

            for mol, _, filename in train_molecules:
                if os.path.join(data_dir, filename) in sdf_to_process:
                    file_name = f"molecule_{mol.GetProp('_Name') if mol.HasProp('_Name') else filename}"
                    centered_result = center_molecule(mol, file_name)
                    if centered_result:
                        local_features = compute_local_features(centered_result[0])
                        if local_features:
                            train_centered_molecules.append((centered_result, local_features, filename))
                            train_sdf_filenames.append(filename)
                        else:
                            logger.warning(f"Skipping molecule {file_name} due to local feature computation failure")
                            train_failed.append(filename)
                    else:
                        logger.warning(f"Skipping molecule {file_name} due to centering failure")
                        train_failed.append(filename)
                else:
                    train_sdf_filenames.append(filename)

            logger.info(f"Processed {len(train_centered_molecules)} training molecules, failed: {len(train_failed)}")

            logger.info("Centering validation molecules and computing local features")
            val_centered_molecules = []
            val_failed = []
            val_sdf_filenames = []
            for mol, _, filename in val_molecules:
                if os.path.join(data_dir, filename) in sdf_to_process:
                    file_name = f"molecule_{mol.GetProp('_Name') if mol.HasProp('_Name') else filename}"
                    centered_result = center_molecule(mol, file_name)
                    if centered_result:
                        local_features = compute_local_features(centered_result[0])
                        if local_features:
                            val_centered_molecules.append((centered_result, local_features, filename))
                            val_sdf_filenames.append(filename)
                        else:
                            logger.warning(f"Skipping molecule {file_name} due to local feature computation failure")
                            val_failed.append(filename)
                    else:
                        logger.warning(f"Skipping molecule {file_name} due to centering failure")
                        val_failed.append(filename)
                else:
                    val_sdf_filenames.append(filename)

            logger.info(f"Processed {len(val_centered_molecules)} validation molecules, failed: {len(val_failed)}")

            logger.info("Voxelizing training molecules")
            os.makedirs(train_voxel_dir, exist_ok=True)
            train_voxel_paths = []
            for _, (centered_mol, local_features, filename) in enumerate(train_centered_molecules):
                file_name = f"molecule_{centered_mol[0].GetProp('_Name') if centered_mol[0].HasProp('_Name') else filename}"
                voxel_grid = voxelize_molecule(centered_mol, local_features, file_name)
                if voxel_grid is not None:
                    filename_base = os.path.splitext(filename)[0]
                    output_path = os.path.join(train_voxel_dir, f"train_voxel_{filename_base}.npy")
                    np.save(output_path, voxel_grid)
                    train_voxel_paths.append(output_path)
                    logger.info(f"Saved training voxel grid for {file_name} to {output_path}")
                else:
                    logger.warning(f"Failed to voxelize training molecule {file_name}")

            logger.info(f"Voxelized {len(train_voxel_paths)} training molecules")

            logger.info("Voxelizing validation molecules")
            os.makedirs(val_voxel_dir, exist_ok=True)
            val_voxel_paths = []
            for _, (centered_mol, local_features, filename) in enumerate(val_centered_molecules):
                file_name = f"molecule_{centered_mol[0].GetProp('_Name') if centered_mol[0].HasProp('_Name') else filename}"
                voxel_grid = voxelize_molecule(centered_mol, local_features, file_name)
                if voxel_grid is not None:
                    filename_base = os.path.splitext(filename)[0]
                    output_path = os.path.join(val_voxel_dir, f"val_voxel_{filename_base}.npy")
                    np.save(output_path, voxel_grid)
                    val_voxel_paths.append(output_path)
                    logger.info(f"Saved validation voxel grid for {file_name} to {output_path}")
                else:
                    logger.warning(f"Failed to voxelize validation molecule {file_name}")

            logger.info(f"Voxelized {len(val_voxel_paths)} validation molecules")
            logger.info("Computing molecular features for training set")
            train_features_df = compute_molecular_features(train_centered_molecules, os.path.dirname(train_features_path))
            train_features_df.to_pickle(train_features_path)
            logger.info(f"Saved training features to {train_features_path}")

            logger.info("Computing molecular features for validation set")
            val_features_df = compute_molecular_features(val_centered_molecules, os.path.dirname(val_features_path))
            val_features_df.to_pickle(val_features_path)
            logger.info(f"Saved validation features to {val_features_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        vae_model, input_size, scalar_columns = train_molecular_vae(
            train_features_path, val_features_path, vae_model_path
        )

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            beta_max = trial.suggest_float("beta_max", 0.1, 2.0)
            latent_dim = trial.suggest_int("latent_dim", 64, 256, step=32)
            num_filters = trial.suggest_categorical("num_filters", [[32, 64, 128], [64, 128, 256], [16, 32, 64]])
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            hidden_size = trial.suggest_int("hidden_size", 256, 1024, step=256)

            train_loader = create_data_loaders(
                train_voxel_dir, batch_size=batch_size, augment_train=True, sdf_filenames=train_sdf_filenames
            )
            val_loader = create_data_loaders(
                val_voxel_dir, batch_size=batch_size, augment_train=False, sdf_filenames=val_sdf_filenames
            )

            model = Ligand3DVAE(
                num_channels=12,
                grid_size=32,
                num_filters=num_filters,
                hidden_size=hidden_size,
                latent_dim=latent_dim
            )
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            num_epochs = 5
            beta_schedule = np.linspace(0, beta_max, num_epochs)
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                total_recon_loss = 0.0
                total_kl_loss = 0.0
                beta = beta_schedule[epoch]

                for voxel_batch, filenames in train_loader:
                    voxel_batch = voxel_batch.to(device)
                    if torch.any(torch.isnan(voxel_batch)) or torch.any(torch.isinf(voxel_batch)):
                        logger.warning("NaN or Inf detected in input voxel batch, skipping")
                        continue

                    reconstructed, mu, logvar, _ = model(voxel_batch)
                    loss, recon_loss, kl_loss = vae_loss_ligand(reconstructed, voxel_batch, mu, logvar, beta=beta)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("NaN or Inf detected in loss, skipping batch")
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_recon_loss += recon_loss
                    total_kl_loss += kl_loss

                avg_total_loss = epoch_loss / len(train_loader)
                logger.info(f"Trial {trial.number} Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_total_loss:.4f}, Recon: {total_recon_loss/len(train_loader):.4f}, KL: {total_kl_loss/len(train_loader):.4f}")

                model.eval()
                val_loss = 0.0
                val_recon_loss = 0.0
                val_kl_loss = 0.0
                with torch.no_grad():
                    for voxel_batch, _ in val_loader:
                        voxel_batch = voxel_batch.to(device)
                        if torch.any(torch.isnan(voxel_batch)) or torch.any(torch.isinf(voxel_batch)):
                            continue
                        reconstructed, mu, logvar, _ = model(voxel_batch)
                        loss, recon_loss, kl_loss = vae_loss_ligand(reconstructed, voxel_batch, mu, logvar, beta=beta)
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        val_loss += loss.item()
                        val_recon_loss += recon_loss
                        val_kl_loss += kl_loss

                avg_val_loss = val_loss / len(val_loader)
                logger.info(f"Trial {trial.number} Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}, Recon: {val_recon_loss/len(val_loader):.4f}, KL: {val_kl_loss/len(val_loader):.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    trial_model_path = f"../output/ligand_autoencoder_trial_{trial.number}.pth"
                    torch.save(model.state_dict(), trial_model_path)
                    logger.info(f"Saved model for Trial {trial.number} to {trial_model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered for Trial {trial.number}")
                        break

                trial.report(avg_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_loss

        study = optuna.create_study(direction="minimize")
        n_trials = 5
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        logger.info(f"Best Ligand3DVAE trial: {best_trial.number}")
        logger.info(f"Best validation loss: {best_trial.value}")
        logger.info(f"Best hyperparameters: {best_trial.params}")

        best_model_path = model_path
        trial_model_path = f"../output/ligand_autoencoder_trial_{best_trial.number}.pth"
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            logger.info(f"Removed existing file {best_model_path}")
        os.rename(trial_model_path, best_model_path)
        logger.info(f"Renamed {trial_model_path} to {best_model_path}")
        with open("../output/ligand_autoencoder_hyperparams.json", "w") as f:
            json.dump(best_trial.params, f)
        logger.info(f"Best Ligand3DVAE model saved to {best_model_path}")
        logger.info(f"Best hyperparameters saved to ../output/ligand_autoencoder_hyperparams.json")

        for trial_num in range(n_trials):
            trial_path = f"../output/ligand_autoencoder_trial_{trial_num}.pth"
            if os.path.exists(trial_path) and trial_path != best_model_path:
                os.remove(trial_path)
        logger.info("Cleaned up trial models")

        best_params = best_trial.params
        model = Ligand3DVAE(
            num_channels=12,
            grid_size=32,
            num_filters=best_params["num_filters"],
            hidden_size=best_params["hidden_size"],
            latent_dim=best_params["latent_dim"]
        )
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        model.eval()
        logger.info(f"Best Ligand3DVAE model loaded from {best_model_path} for inference")

        logger.info("Starting prediction and similarity computation")
        main_prediction(reference_dir, library_dir, model_path, vae_model_path, voxel_output_dir, results_output_dir)
        logger.info("Main execution completed")

        results_dir = "../output/results"
        ref_voxel_dir = "../temp/voxel_grids/reference"
        lib_voxel_dir = "../temp/voxel_grids/library"
        
        mu_manhattan_file = os.path.join(results_dir, "manhattan_distances_ligand_mu.csv")
        visualize_similar_and_dissimilar_pairs(mu_manhattan_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Manhattan")
        
        z_manhattan_file = os.path.join(results_dir, "manhattan_distances_ligand_z.csv")
        visualize_similar_and_dissimilar_pairs(z_manhattan_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Manhattan")
        
        mu_euclidean_file = os.path.join(results_dir, "euclidean_distances_ligand_mu.csv")
        visualize_similar_and_dissimilar_pairs(mu_euclidean_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Euclidean")
        
        z_euclidean_file = os.path.join(results_dir, "euclidean_distances_ligand_z.csv")
        visualize_similar_and_dissimilar_pairs(z_euclidean_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Euclidean")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()