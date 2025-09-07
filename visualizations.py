import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import os
import logging

logger = logging.getLogger(__name__)

def analyze_distances(file_path: str) -> list:
    """
    Analyze distance matrix to return top 10 library molecules by mean distance.
    
    Args:
        file_path (str): Path to the distance matrix CSV file.
    
    Returns:
        list: List of top 10 library molecule filenames.
    """
    try:
        df = pd.read_csv(file_path)
        distance_matrix = df.drop(columns=['Unnamed: 0'])
        mean_distances = distance_matrix.mean()
        min_distances = distance_matrix.min()
        similarity_summary = pd.DataFrame({
            'Mean_Distance': mean_distances,
            'Min_Distance': min_distances, 
            'Library_Molecule': distance_matrix.columns
        })
        sorted_summary = similarity_summary.sort_values(by='Mean_Distance')
        top_10 = sorted_summary["Library_Molecule"].head(10).tolist()
        logger.info(f"Extracted top 10 library molecules from {file_path}")
        return top_10
    except Exception as e:
        logger.error(f"Error analyzing distances in {file_path}: {str(e)}")
        raise

def visualize_similar_and_dissimilar_pairs(distance_file: str, ref_voxel_dir: str, lib_voxel_dir: str, channel: int = None, distance_type: str = "Euclidean") -> None:
    """
    Extract top 10 most similar and top 10 least similar molecule pairs from a distance matrix CSV,
    load their .npy voxel files, and visualize them side by side using plotly_voxel_pair_comparison.
    
    Args:
        distance_file (str): Path to the distance matrix CSV file (e.g., elementwise_distances_mu.csv).
        ref_voxel_dir (str): Directory containing reference .npy voxel files.
        lib_voxel_dir (str): Directory containing library .npy voxel files.
        channel (int, optional): Specific channel to visualize (e.g., 0 for carbon). If None, sum all channels.
        distance_type (str): Type of distance metric used ('Element-wise', 'Manhattan', or 'Euclidean').
    """
    try:
        # Load distance matrix
        df = pd.read_csv(distance_file)
        ref_filenames = df['Unnamed: 0'].tolist()
        lib_filenames = df.columns[1:].tolist()
        distance_matrix = df.drop(columns=['Unnamed: 0']).to_numpy()

        pairs = []
        for i, ref_fn in enumerate(ref_filenames):
            for j, lib_fn in enumerate(lib_filenames):
                if ref_fn != lib_fn:  # Exclude self-pairs
                    pairs.append((distance_matrix[i, j], ref_fn, lib_fn))
        
        # Sort by distance for most similar (ascending) and least similar (descending)
        pairs.sort(key=lambda x: x[0])
        most_similar_pairs = pairs[:10]  # Top 10 smallest distances
        least_similar_pairs = pairs[-10:][::-1]  # Top 10 largest distances (reversed for descending order)
        
        logger.info(f"Identified {len(most_similar_pairs)} most similar and {len(least_similar_pairs)} least similar pairs from {distance_file} using {distance_type} distance")

        # Visualize most similar pairs
        logger.info(f"Visualizing top 10 most similar pairs for {distance_type} distance")
        for idx, (distance, ref_fn, lib_fn) in enumerate(most_similar_pairs, 1):
            logger.info(f"Processing most similar pair {idx}: {ref_fn} vs {lib_fn} ({distance_type} distance: {distance:.4f})")
            
            try:
                ref_idx = int(ref_fn.replace('.sdf', ''))
                lib_idx = int(lib_fn.replace('.sdf', ''))
                ref_voxel_path = os.path.join(ref_voxel_dir, f"ref_voxel_{ref_idx}.npy")  
                lib_voxel_path = os.path.join(lib_voxel_dir, f"lib_voxel_{lib_idx}.npy")
            except ValueError:
                logger.warning(f"Invalid filename format for {ref_fn} or {lib_fn} in {distance_type} pair {idx}, skipping")
                continue

            # Load voxel grids
            try:
                if not os.path.exists(ref_voxel_path):
                    logger.error(f"Reference voxel file {ref_voxel_path} not found for most similar pair {idx}")
                    continue
                if not os.path.exists(lib_voxel_path):
                    logger.error(f"Library voxel file {lib_voxel_path} not found for most similar pair {idx}")
                    continue
                
                ref_voxel = np.load(ref_voxel_path)
                lib_voxel = np.load(lib_voxel_path)
                
                # Validate voxel data
                if np.any(np.isnan(ref_voxel)) or np.any(np.isinf(ref_voxel)):
                    logger.warning(f"NaN/Inf detected in {ref_voxel_path} for most similar pair {idx}, skipping")
                    continue
                if np.any(np.isnan(lib_voxel)) or np.any(np.isinf(lib_voxel)):
                    logger.warning(f"NaN/Inf detected in {lib_voxel_path} for most similar pair {idx}, skipping")
                    continue
                
                # Visualize pair
                logger.info(f"Visualizing most similar pair {idx}: Reference {ref_fn} vs Library {lib_fn} ({distance_type})")
                plotly_voxel_pair_comparison(
                    ref_voxel, lib_voxel, ref_fn, lib_fn, distance, idx, channel, similarity_type=f"Most Similar ({distance_type})"
                )

                if idx == 2:
                    break
                
            except Exception as e:
                logger.error(f"Error processing most similar pair {idx} ({ref_fn} vs {lib_fn}): {str(e)}")
                continue

        # Visualize least similar pairs
        logger.info(f"Visualizing top 10 least similar pairs for {distance_type} distance")
        for idx, (distance, ref_fn, lib_fn) in enumerate(least_similar_pairs, 1):
            logger.info(f"Processing least similar pair {idx}: {ref_fn} vs {lib_fn} ({distance_type} distance: {distance:.4f})")
            
            try:
                ref_idx = int(ref_fn.replace('.sdf', ''))
                lib_idx = int(lib_fn.replace('.sdf', ''))
                ref_voxel_path = os.path.join(ref_voxel_dir, f"ref_voxel_{ref_idx}.npy")  
                lib_voxel_path = os.path.join(lib_voxel_dir, f"lib_voxel_{lib_idx}.npy")
            except ValueError:
                logger.warning(f"Invalid filename format for {ref_fn} or {lib_fn} in least similar pair {idx}, skipping")
                continue

            # Load voxel grids
            try:
                if not os.path.exists(ref_voxel_path):
                    logger.error(f"Reference voxel file {ref_voxel_path} not found for least similar pair {idx}")
                    continue
                if not os.path.exists(lib_voxel_path):
                    logger.error(f"Library voxel file {lib_voxel_path} not found for least similar pair {idx}")
                    continue
                
                ref_voxel = np.load(ref_voxel_path)
                lib_voxel = np.load(lib_voxel_path)
                
                # Validate voxel data
                if np.any(np.isnan(ref_voxel)) or np.any(np.isinf(ref_voxel)):
                    logger.warning(f"NaN/Inf detected in {ref_voxel_path} for least similar pair {idx}, skipping")
                    continue
                if np.any(np.isnan(lib_voxel)) or np.any(np.isinf(lib_voxel)):
                    logger.warning(f"NaN/Inf detected in {lib_voxel_path} for least similar pair {idx}, skipping")
                    continue
                
                # Visualize pair
                logger.info(f"Visualizing least similar pair {idx}: Reference {ref_fn} vs Library {lib_fn} ({distance_type})")
                plotly_voxel_pair_comparison(
                    ref_voxel, lib_voxel, ref_fn, lib_fn, distance, idx, channel, similarity_type=f"Least Similar ({distance_type})"
                )

                if idx == 2:
                    break
                
            except Exception as e:
                logger.error(f"Error processing least similar pair {idx} ({ref_fn} vs {lib_fn}): {str(e)}")
                continue

        logger.info(f"Completed visualization of top 10 most similar and least similar molecule pairs for {distance_type} distance")
        
    except Exception as e:
        logger.error(f"Error in visualize_similar_and_dissimilar_pairs for {distance_file} ({distance_type}): {str(e)}")
        raise

def plotly_voxel_full_view(voxel_grid: np.ndarray, channel: int = None) -> None:
    """
    Visualize a 3D voxel grid including all voxels (including zeros) using Plotly.
    
    Args:
        voxel_grid (np.ndarray): 4D voxel grid (X, Y, Z, C).
        channel (int, optional): Specific channel to visualize. If None, sum over all channels.
    """
    try:
        if channel is not None:
            grid = voxel_grid[..., channel]
        else:
            grid = voxel_grid.sum(axis=-1)

        grid_shape = grid.shape
        x, y, z = np.meshgrid(
            np.arange(grid_shape[0]),
            np.arange(grid_shape[1]),
            np.arange(grid_shape[2]),
            indexing='ij'
        )

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = grid.flatten()

        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=values,
                colorscale='Viridis',
                opacity=0.6,
                cmin=0,
                cmax=1
            )
        ))

        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'),
            title="3D Full Voxel Grid (Including Zeros)"
        )
        output_file = "full_voxel_grid.html"
        fig.write_html(output_file)
        logger.info(f"Saved full voxel grid visualization to {output_file}")
        fig.show()
        logger.info("Generated full voxel grid visualization")
    except Exception as e:
        logger.error(f"Error in plotly_voxel_full_view: {str(e)}")
        raise

def plotly_voxel_nonzero_only(voxel_grid: np.ndarray, channel: int = None, filename: str = "Unknown", pair_number: int = 0, molecule_type: str = "Unknown") -> None:
    """
    Visualize only non-zero voxels of a 3D voxel grid using Plotly with annotated title.
    
    Args:
        voxel_grid (np.ndarray): 4D voxel grid (X, Y, Z, C).
        channel (int, optional): Specific channel to visualize. If None, sum over all channels.
        filename (str): Name of the molecule file (e.g., 0.sdf).
        pair_number (int): Pair number in the top 10 similar pairs.
        molecule_type (str): Type of molecule ('Reference' or 'Library').
    """
    try:
        if channel is not None:
            grid = voxel_grid[..., channel]
        else:
            grid = voxel_grid.sum(axis=-1)

        mask = grid != 0
        x, y, z = np.where(mask)
        values = grid[mask]

        if len(x) == 0:
            logger.warning(f"No non-zero voxels found for {filename} (Pair {pair_number}, {molecule_type})")
            return

        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=values,
                colorscale='Viridis',
                opacity=0.6,
                showscale=True
            )
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, 32], title='X'),
                yaxis=dict(range=[0, 32], title='Y'),
                zaxis=dict(range=[0, 32], title='Z'),
            ),
            title=f"3D Voxel Grid — {molecule_type} Molecule: {filename} (Pair {pair_number}, Channel: {channel if channel is not None else 'Summed'})"
        )

        output_file = f"voxel_nonzero_{molecule_type.lower()}_{filename.replace('.sdf', '')}_pair_{pair_number}.html"
        fig.write_html(output_file)
        logger.info(f"Saved plot for {molecule_type} molecule: {filename} (Pair {pair_number}) to {output_file}")
        fig.show()
        logger.info(f"Generated plot for {molecule_type} molecule: {filename} (Pair {pair_number})")
        
    except Exception as e:
        logger.error(f"Error visualizing {molecule_type} molecule {filename} (Pair {pair_number}): {str(e)}")

def plotly_voxel_pair_comparison(
    ref_voxel: np.ndarray, 
    lib_voxel: np.ndarray, 
    ref_filename: str, 
    lib_filename: str, 
    distance: float, 
    pair_number: int, 
    channel: int = None,
    similarity_type: str = "Unknown"
) -> None:
    """
    Visualize a pair of molecules side by side for comparison using Plotly subplots.
    
    Args:
        ref_voxel (np.ndarray): 4D voxel grid for the reference molecule (X, Y, Z, C).
        lib_voxel (np.ndarray): 4D voxel grid for the library molecule (X, Y, Z, C).
        ref_filename (str): Filename of the reference molecule (e.g., 0.sdf).
        lib_filename (str): Filename of the library molecule (e.g., 0.sdf).
        distance (float): Distance between the molecule pair.
        pair_number (int): Pair number in the top 10 similar or dissimilar pairs.
        channel (int, optional): Specific channel to visualize. If None, sum over all channels.
        similarity_type (str): Type of similarity (e.g., 'Most Similar (Element-wise)').
    """
    try:
        # Prepare voxel grids
        ref_grid = ref_voxel[..., channel] if channel is not None else ref_voxel.sum(axis=-1)
        lib_grid = lib_voxel[..., channel] if channel is not None else lib_voxel.sum(axis=-1)

        # Create masks for non-zero voxels
        ref_mask = ref_grid != 0
        lib_mask = lib_grid != 0

        ref_x, ref_y, ref_z = np.where(ref_mask)
        lib_x, lib_y, lib_z = np.where(lib_mask)
        ref_values = ref_grid[ref_mask]
        lib_values = lib_grid[lib_mask]

        if len(ref_x) == 0:
            logger.warning(f"No non-zero voxels found for Reference molecule {ref_filename} (Pair {pair_number}, {similarity_type})")
            return
        if len(lib_x) == 0:
            logger.warning(f"No non-zero voxels found for Library molecule {lib_filename} (Pair {pair_number}, {similarity_type})")
            return

        # Create subplot figure
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Reference: {ref_filename} (Channel: {channel if channel is not None else 'Summed'})",
                f"Library: {lib_filename} (Channel: {channel if channel is not None else 'Summed'})"
            ],
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            horizontal_spacing=0.1
        )

        # Add reference molecule scatter
        fig.add_trace(
            go.Scatter3d(
                x=ref_x, y=ref_y, z=ref_z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=ref_values,
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True
                ),
                name=f"Reference: {ref_filename}"
            ),
            row=1, col=1
        )

        # Add library molecule scatter
        fig.add_trace(
            go.Scatter3d(
                x=lib_x, y=lib_y, z=lib_z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=lib_values,
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True
                ),
                name=f"Library: {lib_filename}"
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"{similarity_type} Pair {pair_number} Comparison (Distance: {distance:.4f})",
            scene1=dict(
                xaxis=dict(range=[0, 32], title='X'),
                yaxis=dict(range=[0, 32], title='Y'),
                zaxis=dict(range=[0, 32], title='Z')
            ),
            scene2=dict(
                xaxis=dict(range=[0, 32], title='X'),
                yaxis=dict(range=[0, 32], title='Y'),
                zaxis=dict(range=[0, 32], title='Z')
            ),
            showlegend=False
        )

        # Save the plot
        similarity_type_clean = similarity_type.replace(" ", "_").replace("(", "").replace(")", "")
        output_file = f"voxel_pair_{similarity_type_clean}_pair_{pair_number}_ref_{ref_filename.replace('.sdf', '')}_lib_{lib_filename.replace('.sdf', '')}.html"
        fig.write_html(output_file)
        logger.info(f"Saved side-by-side comparison plot for {similarity_type} Pair {pair_number}: {ref_filename} vs {lib_filename} to {output_file}")
        fig.show()
        logger.info(f"Generated side-by-side comparison plot for {similarity_type} Pair {pair_number}: {ref_filename} vs {lib_filename}")
        
    except Exception as e:
        logger.error(f"Error visualizing {similarity_type} Pair {pair_number} ({ref_filename} vs {lib_filename}): {str(e)}")

if __name__ == "__main__":
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