import logging
import os
import numpy as np
import pandas as pd
import torch
from load_data import load_sdf_files
from molecule_utils import center_molecule, compute_local_features, voxelize_molecule, compute_molecular_features
from data_loader import create_data_loaders
from model import Ligand3DVAE
from ANN import MolecularVAE, create_molecular_data_loaders
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from visualizations import analyze_distances, visualize_similar_and_dissimilar_pairs, plotly_voxel_full_view, plotly_voxel_nonzero_only, plotly_voxel_pair_comparison
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

def load_pretrained_3DVAE_model(model_path: str, num_channels: int = 12, device: str = None) -> Ligand3DVAE:
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load best hyperparameters
        hyperparams_path_3d = "../output/ligand_autoencoder_hyperparams.json"
        if os.path.exists(hyperparams_path_3d):
            with open(hyperparams_path_3d, "r") as f:
                params = json.load(f)
            logger.info(f"Loaded hyperparameters from {hyperparams_path_3d}: {params}")
            num_filters = params["num_filters"]
            hidden_size = params["hidden_size"]
            latent_dim = params["latent_dim"]
        else:
            logger.warning(f"Hyperparameters file {hyperparams_path_3d} not found, using defaults")
            num_filters = [16, 32, 64]
            hidden_size = 1024
            latent_dim = 96

        model = Ligand3DVAE(
            num_channels=num_channels,
            grid_size=32,
            num_filters=num_filters,
            hidden_size=hidden_size,
            latent_dim=latent_dim
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.fc_enc1.parameters():
            param.requires_grad = False
        for param in model.fc_mu.parameters():
            param.requires_grad = False
        for param in model.fc_logvar.parameters():
            param.requires_grad = False

        model.eval()
        logger.info(f"Loaded and froze Ligand3DVAE encoder from {model_path} on {device} with num_filters={num_filters}, hidden_size={hidden_size}, latent_dim={latent_dim}")
        return model
    except Exception as e:
        logger.error(f"Error loading Ligand3DVAE model from {model_path}: {str(e)}")
        raise

def load_pretrained_molecular_vae(model_path: str, input_size: int, device: str = None) -> MolecularVAE:
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load best hyperparameters
        hyperparams_path = f"{model_path}_hyperparams.json"
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, "r") as f:
                params = json.load(f)
            hidden_sizes = [params["hidden_size1"], params["hidden_size2"]]
            latent_dim = params["latent_dim"]
            dropout_rate = params["dropout_rate"]
        else:
            logger.warning(f"Hyperparameters file {hyperparams_path} not found, using defaults")
            hidden_sizes = [512, 256]
            latent_dim = 128
            dropout_rate = 0.3

        model = MolecularVAE(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.fc_mu.parameters():
            param.requires_grad = False
        for param in model.fc_logvar.parameters():
            param.requires_grad = False

        model.eval()
        logger.info(f"Loaded and froze MolecularVAE encoder from {model_path} on {device} with hidden_sizes={hidden_sizes}, latent_dim={latent_dim}, dropout_rate={dropout_rate}")
        return model
    except Exception as e:
        logger.error(f"Error loading MolecularVAE model from {model_path}: {str(e)}")
        raise

def compute_embeddings(model, data_loader, device, use_mu=True, model_type="ligand"):
    embeddings = []
    filenames = []
    model.eval()
    with torch.no_grad():
        for batch, batch_filenames in data_loader:
            batch = batch.to(device)
            if model_type == "ligand":
                _, mu, logvar, z = model(batch)
            else:
                _, mu, logvar, z = model(batch)
            embeddings.append(mu if use_mu else z)
            filenames.extend(batch_filenames)

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    embeddings = normalize(embeddings, norm='l2')
    return embeddings, filenames

def concatenate_embeddings(ligand_embeddings, mol_embeddings, ligand_filenames, mol_filenames, embedding_type):
    common_filenames = list(set(ligand_filenames) & set(mol_filenames))
    if not common_filenames:
        logger.warning("No common filenames found between ligand and molecular embeddings")
        return np.array([]), []
    
    ligand_idx = [ligand_filenames.index(fn) for fn in common_filenames]
    mol_idx = [mol_filenames.index(fn) for fn in common_filenames]
    
    concat_embeddings = np.concatenate(
        [ligand_embeddings[ligand_idx], mol_embeddings[mol_idx]], axis=1
    )
    logger.info(f"Concatenated {embedding_type} embeddings for {len(common_filenames)} common molecules")
    return concat_embeddings, common_filenames

def compute_similarity_metrics(ref_embeddings, lib_embeddings, ref_filenames, lib_filenames, output_dir, embedding_type):
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Cosine similarity
        cos_sim = cosine_similarity(ref_embeddings, lib_embeddings)
        cos_sim_df = pd.DataFrame(cos_sim, index=ref_filenames, columns=lib_filenames)
        cos_sim_df.to_csv(os.path.join(output_dir, f"cosine_similarity_{embedding_type}.csv"))
        logger.info(f"Saved cosine similarity matrix for {embedding_type} to {output_dir}")

        # Euclidean distance
        euclidean_dist = euclidean_distances(ref_embeddings, lib_embeddings)
        euclidean_df = pd.DataFrame(euclidean_dist, index=ref_filenames, columns=lib_filenames)
        euclidean_df.to_csv(os.path.join(output_dir, f"euclidean_distances_{embedding_type}.csv"))
        logger.info(f"Saved Euclidean distance matrix for {embedding_type} to {output_dir}")

        # Manhattan distance
        manhattan_dist = manhattan_distances(ref_embeddings, lib_embeddings)
        manhattan_df = pd.DataFrame(manhattan_dist, index=ref_filenames, columns=lib_filenames)
        manhattan_df.to_csv(os.path.join(output_dir, f"manhattan_distances_{embedding_type}.csv"))
        logger.info(f"Saved Manhattan distance matrix for {embedding_type} to {output_dir}")

         # 4. Element-wise distance vectors
        elementwise_dir = os.path.join("..", "temp", "element_wise", embedding_type)
        os.makedirs(elementwise_dir, exist_ok=True)

        avg_distances = []
        for i, ref_embed in enumerate(ref_embeddings):
            for j, lib_embed in enumerate(lib_embeddings):
                ref_file = os.path.splitext(ref_filenames[i])[0]
                lib_file = os.path.splitext(lib_filenames[j])[0]
                out_file = f"{ref_file}_vs_{lib_file}.npy"
                out_path = os.path.join(elementwise_dir, out_file)

                elementwise_vector = np.abs(ref_embed - lib_embed)  # L1 element-wise difference
                np.save(out_path, elementwise_vector)

                avg_distance = np.mean(elementwise_vector)
                avg_distances.append({
                    "Ref": ref_filenames[i],
                    "Lib": lib_filenames[j],
                    "AvgElementwiseDistance": avg_distance
                })

        # Save average distances to CSV
        avg_df = pd.DataFrame(avg_distances)
        avg_csv_path = os.path.join(output_dir, f"average_elementwise_distance_{embedding_type}.csv")
        avg_df.to_csv(avg_csv_path, index=False)
        logger.info(f"Saved average element-wise distances for {embedding_type} to {avg_csv_path}")

    except Exception as e:
        logger.error(f"Error computing similarity metrics for {embedding_type}: {str(e)}")
        raise


def main_prediction(reference_dir, library_dir, model_path, vae_model_path, voxel_output_dir, results_output_dir):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        ref_voxel_dir = os.path.join(voxel_output_dir, "reference")
        lib_voxel_dir = os.path.join(voxel_output_dir, "library")
        train_features_path = os.path.join(voxel_output_dir, "train_molecular_features.pkl")
        val_features_path = os.path.join(voxel_output_dir, "val_molecular_features.pkl")

        logger.info("Loading and preprocessing reference molecules")
        ref_molecules = load_sdf_files(reference_dir)
        ref_centered_molecules = []
        ref_sdf_filenames = []
        for mol, _, filename in ref_molecules:
            file_name = f"molecule_{mol.GetProp('_Name') if mol.HasProp('_Name') else filename}"
            centered_result = center_molecule(mol, file_name)
            if centered_result:
                local_features = compute_local_features(centered_result[0])
                if local_features:
                    ref_centered_molecules.append((centered_result, local_features, filename))
                    ref_sdf_filenames.append(filename)
                else:
                    logger.warning(f"Skipping reference molecule {file_name} due to local feature computation failure")
            else:
                logger.warning(f"Skipping reference molecule {file_name} due to centering failure")
        
        logger.info(f"Processed {len(ref_centered_molecules)} reference molecules")

        logger.info("Voxelizing reference molecules")
        os.makedirs(ref_voxel_dir, exist_ok=True)
        ref_voxel_paths = []
        for idx, (centered_mol, local_features, filename) in enumerate(ref_centered_molecules):
            file_name = f"molecule_{centered_mol[0].GetProp('_Name') if centered_mol[0].GetProp('_Name') else filename}"
            voxel_grid = voxelize_molecule(centered_mol, local_features, file_name)
            if voxel_grid is not None:
                filename_base = os.path.splitext(filename)[0]
                output_path = os.path.join(ref_voxel_dir, f"ref_voxel_{filename_base}.npy")
                np.save(output_path, voxel_grid)
                ref_voxel_paths.append(output_path)
                logger.info(f"Saved reference voxel grid for {file_name} to {output_path}")
            else:
                logger.warning(f"Failed to voxelize reference molecule {file_name}")
        
        logger.info(f"Voxelized {len(ref_voxel_paths)} reference molecules")

        logger.info("Loading and preprocessing library molecules")
        lib_molecules = load_sdf_files(library_dir)
        lib_centered_molecules = []
        lib_sdf_filenames = []
        for mol, _, filename in lib_molecules:
            file_name = f"molecule_{mol.GetProp('_Name') if mol.HasProp('_Name') else filename}"
            centered_result = center_molecule(mol, file_name)
            if centered_result:
                local_features = compute_local_features(centered_result[0])
                if local_features:
                    lib_centered_molecules.append((centered_result, local_features, filename))
                    lib_sdf_filenames.append(filename)
                else:
                    logger.warning(f"Skipping library molecule {file_name} due to local feature computation failure")
            else:
                logger.warning(f"Skipping library molecule {file_name} due to centering failure")
        
        logger.info(f"Processed {len(lib_centered_molecules)} library molecules")

        logger.info("Voxelizing library molecules")
        os.makedirs(lib_voxel_dir, exist_ok=True)
        lib_voxel_paths = []
        for idx, (centered_mol, local_features, filename) in enumerate(lib_centered_molecules):
            file_name = f"molecule_{centered_mol[0].GetProp('_Name') if centered_mol[0].GetProp('_Name') else filename}"
            voxel_grid = voxelize_molecule(centered_mol, local_features, file_name)
            if voxel_grid is not None:
                filename_base = os.path.splitext(filename)[0]
                output_path = os.path.join(lib_voxel_dir, f"lib_voxel_{filename_base}.npy")
                np.save(output_path, voxel_grid)
                lib_voxel_paths.append(output_path)
                logger.info(f"Saved library voxel grid for {file_name} to {output_path}")
            else:
                logger.warning(f"Failed to voxelize library molecule {file_name}")
        
        logger.info(f"Voxelized {len(lib_voxel_paths)} library molecules")

        logger.info("Computing molecular features for reference and library sets")
        ref_features_df = compute_molecular_features(ref_centered_molecules, os.path.dirname(train_features_path))
        lib_features_df = compute_molecular_features(lib_centered_molecules, os.path.dirname(val_features_path))

        if ref_features_df is None or lib_features_df is None:
            logger.error("Failed to compute molecular features for reference or library sets")
            raise ValueError("Failed to compute molecular features")

        ref_features_df.to_pickle(train_features_path)
        lib_features_df.to_pickle(val_features_path)
        logger.info(f"Saved reference features to {train_features_path}")
        logger.info(f"Saved library features to {val_features_path}")

        logger.info("Loading pretrained models")
        ligand_model = load_pretrained_3DVAE_model(model_path, num_channels=12, device=device)
        
        train_mol_loader, val_mol_loader, input_size, _ = create_molecular_data_loaders(
            train_features_path, val_features_path, batch_size=16
        )
        mol_vae_model = load_pretrained_molecular_vae(vae_model_path, input_size, device)

        logger.info("Computing Ligand3DVAE mu embeddings")
        ref_ligand_mu, ref_filenames = compute_embeddings(ligand_model, create_data_loaders(ref_voxel_dir, batch_size=16, sdf_filenames=ref_sdf_filenames), device, use_mu=True, model_type="ligand")
        lib_ligand_mu, lib_filenames = compute_embeddings(ligand_model, create_data_loaders(lib_voxel_dir, batch_size=16, sdf_filenames=lib_sdf_filenames), device, use_mu=True, model_type="ligand")
        
        logger.info("Computing Ligand3DVAE z embeddings")
        ref_ligand_z, _ = compute_embeddings(ligand_model, create_data_loaders(ref_voxel_dir, batch_size=16, sdf_filenames=ref_sdf_filenames), device, use_mu=False, model_type="ligand")
        lib_ligand_z, _ = compute_embeddings(ligand_model, create_data_loaders(lib_voxel_dir, batch_size=16, sdf_filenames=lib_sdf_filenames), device, use_mu=False, model_type="ligand")

        logger.info("Computing MolecularVAE mu embeddings")
        ref_mol_mu, ref_mol_filenames = compute_embeddings(mol_vae_model, train_mol_loader, device, use_mu=True, model_type="molecular")
        lib_mol_mu, lib_mol_filenames = compute_embeddings(mol_vae_model, val_mol_loader, device, use_mu=True, model_type="molecular")
        
        logger.info("Computing MolecularVAE z embeddings")
        ref_mol_z, _ = compute_embeddings(mol_vae_model, train_mol_loader, device, use_mu=False, model_type="molecular")
        lib_mol_z, _ = compute_embeddings(mol_vae_model, val_mol_loader, device, use_mu=False, model_type="molecular")

        logger.info("Concatenating mu embeddings")
        ref_concat_mu, ref_common_filenames = concatenate_embeddings(ref_ligand_mu, ref_mol_mu, ref_filenames, ref_mol_filenames, "concat_mu")
        lib_concat_mu, lib_common_filenames = concatenate_embeddings(lib_ligand_mu, lib_mol_mu, lib_filenames, lib_mol_filenames, "concat_mu")
        
        logger.info("Concatenating z embeddings")
        ref_concat_z, _ = concatenate_embeddings(ref_ligand_z, ref_mol_z, ref_filenames, ref_mol_filenames, "concat_z")
        lib_concat_z, _ = concatenate_embeddings(lib_ligand_z, lib_mol_z, lib_filenames, lib_mol_filenames, "concat_z")

        compute_similarity_metrics(ref_ligand_mu, lib_ligand_mu, ref_filenames, lib_filenames, results_output_dir, embedding_type='ligand_mu')
        compute_similarity_metrics(ref_ligand_z, lib_ligand_z, ref_filenames, lib_filenames, results_output_dir, embedding_type='ligand_z')
        compute_similarity_metrics(ref_mol_mu, lib_mol_mu, ref_mol_filenames, lib_mol_filenames, results_output_dir, embedding_type='mol_mu')
        compute_similarity_metrics(ref_mol_z, lib_mol_z, ref_mol_filenames, lib_mol_filenames, results_output_dir, embedding_type='mol_z')
        compute_similarity_metrics(ref_concat_mu, lib_concat_mu, ref_common_filenames, lib_common_filenames, results_output_dir, embedding_type='concat_mu')
        compute_similarity_metrics(ref_concat_z, lib_concat_z, ref_common_filenames, lib_common_filenames, results_output_dir, embedding_type='concat_z')

        logger.info("Prediction and similarity computation completed")
    except Exception as e:
        logger.error(f"Error in main prediction: {str(e)}")
        raise

if __name__ == "__main__":
    reference_dir = "../data/reference"
    library_dir = "../data/library"
    model_path = "../output/ligand_autoencoder.pth"
    vae_model_path = "../output/molecular_vae.pth"
    voxel_output_dir = "../temp/voxel_grids"
    results_output_dir = "../output/results"
    
    main_prediction(reference_dir, library_dir, model_path, vae_model_path, voxel_output_dir, results_output_dir)

    results_dir = "../output/results"
    ref_voxel_dir = "../temp/voxel_grids/reference"
    lib_voxel_dir = "../temp/voxel_grids/library"
    
    # Visualize for Element-wise distances (ligand_mu embeddings)
    # mu_elementwise_file = os.path.join(results_dir, "elementwise_distances_ligand_mu.csv")
    # visualize_similar_and_dissimilar_pairs(mu_elementwise_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Element-wise")
    
    # Visualize for Element-wise distances (ligand_z embeddings)
    # z_elementwise_file = os.path.join(results_dir, "elementwise_distances_ligand_z.csv")
    # visualize_similar_and_dissimilar_pairs(z_elementwise_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Element-wise")
    
    # Visualize for Manhattan distances (ligand_mu embeddings)
    mu_manhattan_file = os.path.join(results_dir, "manhattan_distances_ligand_mu.csv")
    visualize_similar_and_dissimilar_pairs(mu_manhattan_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Manhattan")
    
    # Visualize for Manhattan distances (ligand_z embeddings)
    z_manhattan_file = os.path.join(results_dir, "manhattan_distances_ligand_z.csv")
    visualize_similar_and_dissimilar_pairs(z_manhattan_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Manhattan")
    
    # Visualize for Euclidean distances (ligand_mu embeddings)
    mu_euclidean_file = os.path.join(results_dir, "euclidean_distances_ligand_mu.csv")
    visualize_similar_and_dissimilar_pairs(mu_euclidean_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Euclidean")
    
    # Visualize for Euclidean distances (ligand_z embeddings)
    z_euclidean_file = os.path.join(results_dir, "euclidean_distances_ligand_z.csv")
    visualize_similar_and_dissimilar_pairs(z_euclidean_file, ref_voxel_dir, lib_voxel_dir, channel=0, distance_type="Euclidean")