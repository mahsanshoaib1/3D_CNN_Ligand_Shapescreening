import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import optuna
import json
import torch.optim as optim
from typing import Tuple, List, Optional
import os

logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    def __init__(self, data: pd.DataFrame, scalar_columns: List[str], fingerprint_column: str):
        """
        Dataset for molecular features (scalar descriptors and fingerprints).
        
        Args:
            data (pd.DataFrame): DataFrame containing scalar features, Morgan fingerprints, and filenames.
            scalar_columns (List[str]): List of column names for scalar features.
            fingerprint_column (str): Column name for Morgan fingerprints.
        """
        self.data = data
        self.scalar_columns = scalar_columns
        self.fingerprint_column = fingerprint_column
        
        # Verify columns exist
        missing_cols = [col for col in scalar_columns + [fingerprint_column] if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns in DataFrame: {missing_cols}")
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
        # Standardize scalar features
        self.scaled_data = data[scalar_columns].copy()
        self.scaled_data = (self.scaled_data - self.scaled_data.mean()) / self.scaled_data.std()
        
        # Replace NaN with 0 in scaled data
        self.scaled_data = self.scaled_data.fillna(0)
        
        # Store fingerprints
        self.fingerprints = data[fingerprint_column].values
        self.filenames = data['filename'].values
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Get scalar features
        scalars = self.scaled_data.iloc[idx].values.astype(np.float32)
        scalars = torch.tensor(scalars, dtype=torch.float32)
        
        # Get fingerprints
        fingerprint = self.fingerprints[idx].astype(np.float32)
        fingerprint = torch.tensor(fingerprint, dtype=torch.float32)
        
        # Concatenate features
        features = torch.cat([scalars, fingerprint], dim=0)
        
        return features, self.filenames[idx]

def create_molecular_data_loaders(
    train_features_path: str,
    val_features_path: str,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """
    Create DataLoaders for training and validation molecular features.
    
    Args:
        train_features_path (str): Path to training features pickle file.
        val_features_path (str): Path to validation features pickle file.
        batch_size (int): Batch size for DataLoaders.
    
    Returns:
        Tuple[DataLoader, DataLoader, int, List[str]]: Training DataLoader, validation DataLoader,
        input size (scalar features + fingerprint), and list of scalar column names.
    """
    try:
        # Load DataFrames
        train_df = pd.read_pickle(train_features_path)
        val_df = pd.read_pickle(val_features_path)
        
        # Define scalar columns (excluding fingerprint and filename)
        scalar_columns = [
            'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'InertialShapeFactor',
            'SpherocityIndex', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
            'LabuteASA', 'TPSA', 'ExactMolWt', 'MolMR'
        ]
        fingerprint_column = 'MorganFingerprint'
        
        # Verify columns
        for df, name in [(train_df, 'training'), (val_df, 'validation')]:
            missing_cols = [col for col in scalar_columns + [fingerprint_column, 'filename'] if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {name} DataFrame: {missing_cols}")
                raise ValueError(f"Missing columns in {name} DataFrame: {missing_cols}")
        
        # Create datasets
        train_dataset = MolecularDataset(train_df, scalar_columns, fingerprint_column)
        val_dataset = MolecularDataset(val_df, scalar_columns, fingerprint_column)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Calculate input size (number of scalar features + fingerprint length)
        input_size = len(scalar_columns) + len(train_df[fingerprint_column].iloc[0])
        
        logger.info(f"Created molecular DataLoaders with input size: {input_size}")
        return train_loader, val_loader, input_size, scalar_columns
    
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {str(e)}")
        raise

class MolecularVAE(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256], latent_dim=128, dropout_rate=0.3):
        super(MolecularVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.scalar_dim = 14
        self.fingerprint_dim = input_size - self.scalar_dim

        # Encoder
        encoder_layers = []
        prev_size = input_size
        for size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_sizes[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_size = latent_dim
        for size in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_scalar = nn.Linear(hidden_sizes[0], self.scalar_dim)
        self.fc_fingerprint = nn.Linear(hidden_sizes[0], self.fingerprint_dim)

        logger.info(f"Initialized VAE with input_size={input_size}, hidden_sizes={hidden_sizes}, latent_dim={latent_dim}, dropout_rate={dropout_rate}")

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        scalar_out = self.fc_scalar(h)
        fingerprint_out = torch.sigmoid(self.fc_fingerprint(h))
        return torch.cat((scalar_out, fingerprint_out), dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

def vae_loss(recon_batch, batch, mu, logvar, scalar_dim, beta=1.0):
    """
    Compute VAE loss for molecular features (scalar features + fingerprints).
    
    Args:
        recon_batch (torch.Tensor): Reconstructed input from the VAE.
        batch (torch.Tensor): Original input tensor.
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log-variance of the latent distribution.
        scalar_dim (int): Number of scalar features (before fingerprints).
        beta (float): Weight for KL divergence term.
    
    Returns:
        Tuple[torch.Tensor, float, float]: Total loss, reconstruction loss, KL divergence.
    """
    try:
        # Split input into scalar features and fingerprints
        recon_scalars = recon_batch[:, :scalar_dim]
        recon_fingerprints = recon_batch[:, scalar_dim:]
        scalars = batch[:, :scalar_dim]
        fingerprints = batch[:, scalar_dim:]
        
        # Compute MSE for scalar features and BCE for fingerprints
        mse_loss = F.mse_loss(recon_scalars, scalars, reduction='mean')
        bce_loss = F.binary_cross_entropy_with_logits(recon_fingerprints, fingerprints, reduction='mean')
        recon_loss = mse_loss + bce_loss
        
        # Compute KL divergence
        logvar = torch.clamp(logvar, min=-10, max=10)
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            logger.warning("NaN/Inf in KL divergence, setting to 0")
            kl_div = torch.tensor(0.0, device=mu.device)
        
        total_loss = recon_loss + beta * kl_div
        return total_loss, recon_loss.item(), kl_div.item()
    
    except Exception as e:
        logger.error(f"Error in vae_loss: {str(e)}")
        raise

def train_molecular_vae(train_features_path: str, val_features_path: str, model_path: str, device: str = None) -> Tuple[MolecularVAE, int, List[str]]:
    """
    Train the MolecularVAE model using Optuna for hyperparameter optimization.
    
    Args:
        train_features_path (str): Path to training features pickle file.
        val_features_path (str): Path to validation features pickle file.
        model_path (str): Path to save the best model.
        device (str): Device to run training on (cuda or cpu).
    
    Returns:
        Tuple[MolecularVAE, int, List[str]]: Trained model, input size, and scalar columns.
    """

    # At the start of train_molecular_vae
    if os.path.exists(model_path):
        os.remove(model_path)
        logger.info(f"Removed existing model file: {model_path}")

    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def objective(trial):
            # Define hyperparameters
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            beta = trial.suggest_float("beta", 0.1, 1.0)
            latent_dim = trial.suggest_int("latent_dim", 64, 256, step=32)
            hidden_size1 = trial.suggest_int("hidden_size1", 256, 1024, step=256)
            hidden_size2 = trial.suggest_int("hidden_size2", 128, 512, step=128)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
            
            # Create data loaders
            train_loader, val_loader, input_size, scalar_columns = create_molecular_data_loaders(
                train_features_path, val_features_path, batch_size
            )
            scalar_dim = len(scalar_columns)  # Number of scalar features
            
            # Initialize model
            model = MolecularVAE(
                input_size=input_size,
                hidden_sizes=[hidden_size1, hidden_size2],
                latent_dim=latent_dim,
                dropout_rate=dropout_rate
            ).to(device)
            
            logger.info(f"Initialized VAE with input_size={input_size}, hidden_sizes=[{hidden_size1}, {hidden_size2}], latent_dim={latent_dim}, dropout_rate={dropout_rate}")
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            num_epochs = 5
            patience = 10
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                total_recon_loss = 0.0
                total_kl_loss = 0.0
                
                for batch, _ in train_loader:
                    batch = batch.to(device)
                    if torch.any(torch.isnan(batch)) or torch.any(torch.isinf(batch)):
                        logger.warning("NaN or Inf detected in input batch, skipping")
                        continue
                    
                    recon_batch, mu, logvar, _ = model(batch)
                    loss, recon_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, scalar_dim, beta=beta)
                    
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
                    for batch, _ in val_loader:
                        batch = batch.to(device)
                        if torch.any(torch.isnan(batch)) or torch.any(torch.isinf(batch)):
                            continue
                        recon_batch, mu, logvar, _ = model(batch)
                        loss, recon_loss, kl_loss = vae_loss(recon_batch, batch, mu, logvar, scalar_dim, beta=beta)
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
                    trial_model_path = f"../output/molecular_vae_trial_{trial.number}.pth"
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
        study.optimize(objective, n_trials=5)
        
        best_trial = study.best_trial
        logger.info(f"Best MolecularVAE trial: {best_trial.number}")
        logger.info(f"Best validation loss: {best_trial.value}")
        logger.info(f"Best hyperparameters: {best_trial.params}")
        
        # Rename best model, handling existing file
        trial_model_path = f"../output/molecular_vae_trial_{best_trial.number}.pth"
        best_model_path = model_path
        try:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)  # Remove existing file
                logger.info(f"Removed existing model file: {best_model_path}")
            os.rename(trial_model_path, best_model_path)
            logger.info(f"Renamed {trial_model_path} to {best_model_path}")
        except Exception as e:
            logger.error(f"Error renaming model file: {str(e)}")
            raise
        
        # Save hyperparameters
        with open(f"{model_path}_hyperparams.json", "w") as f:
            json.dump(best_trial.params, f)
        logger.info(f"Best hyperparameters saved to {model_path}_hyperparams.json")
        
        # Clean up trial models
        for trial_num in range(20):
            trial_path = f"../output/molecular_vae_trial_{trial_num}.pth"
            if os.path.exists(trial_path) and trial_path != best_model_path:
                os.remove(trial_path)
                logger.info(f"Removed trial model: {trial_path}")
        
        # Load best model
        train_loader, val_loader, input_size, scalar_columns = create_molecular_data_loaders(
            train_features_path, val_features_path, batch_size=16
        )
        best_params = best_trial.params
        model = MolecularVAE(
            input_size=input_size,
            hidden_sizes=[best_params["hidden_size1"], best_params["hidden_size2"]],
            latent_dim=best_params["latent_dim"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        logger.info(f"Best MolecularVAE model loaded from {best_model_path}")
        
        return model, input_size, scalar_columns
    
    except Exception as e:
        logger.error(f"Error in training MolecularVAE: {str(e)}")
        raise