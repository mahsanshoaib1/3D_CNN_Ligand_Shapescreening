import torch
import torch.nn as nn
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ligand3DVAE(nn.Module):
    def __init__(self, num_channels, grid_size=32, num_filters=[32, 64, 128], hidden_size=512, latent_dim=128):
        super(Ligand3DVAE, self).__init__()
        self.grid_size = grid_size
        self.num_filters = num_filters
        encoded_size = grid_size // 8

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(num_channels, num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(num_filters[0], num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(num_filters[1], num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.flat_dim = num_filters[2] * encoded_size * encoded_size * encoded_size
        self.fc_enc1 = nn.Linear(self.flat_dim, hidden_size)
        self.relu_fc = nn.ReLU()

        # Latent space
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, hidden_size)
        self.fc_dec2 = nn.Linear(hidden_size, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(num_filters[2], num_filters[1], kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters[1], num_filters[0], kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters[0], num_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        logger.info(f"Initialized Ligand3DVAE with num_filters={num_filters}, hidden_size={hidden_size}, latent_dim={latent_dim}")

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.relu_fc(self.fc_enc1(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        z = self.reparametrize(mu, logvar)

        x = self.relu_fc(self.fc_dec1(z))
        x = self.fc_dec2(x)
        x = x.view(batch_size, self.num_filters[2], self.grid_size // 8, self.grid_size // 8, self.grid_size // 8)
        x = self.decoder(x)

        return x, mu, logvar, z
    
def compute_cosine_similarity(reference_embeddings: np.ndarray, library_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between reference and library embeddings.

    Args:
        reference_embeddings (np.ndarray): Array of shape (N_ref, D).
        library_embeddings (np.ndarray): Array of shape (N_lib, D).

    Returns:
        np.ndarray: Cosine similarity matrix of shape (N_ref, N_lib).
    """
    try:
        ref_tensor = np.array(reference_embeddings, dtype=float)
        lib_tensor = np.array(library_embeddings, dtype=float)
        cosin_matrix = cosine_similarity(ref_tensor, lib_tensor)
        logger.debug(f"Cosine similarity matrix computed with shape {cosin_matrix.shape}")
        return cosin_matrix
    except Exception as e:
        logger.error(f"Error computing cosine similarity: {str(e)}")
        raise