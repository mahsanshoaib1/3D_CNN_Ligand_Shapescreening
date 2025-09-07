import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
from typing import Tuple, Optional, Dict
from Bio.PDB import PDBParser, DSSP
from collections import Counter
import os
from rdkit.Chem import rdFingerprintGenerator
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import pandas as pd

# Ensure the output directory exists
log_dir = os.path.abspath('../output')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'output.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def center_molecule(mol: Chem.Mol, file_name: str) -> Optional[Tuple[Chem.Mol, np.ndarray, np.ndarray]]:
    """
    Center the molecule by translating its center of mass to (0, 0, 0).
    
    Args:
        file_name (str): Name of the SDF file for logging purposes.
    
    Returns:
        Tuple[Chem.Mol, np.ndarray, np.ndarray]: Tuple containing centered molecule,
        centered coordinates, and center of mass, or None if processing fails.
    """
    if mol is None:
        logger.warning(f"Invalid molecule provided for {file_name}")
        return None
    
    try:
        conformer = mol.GetConformer()
        coords = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        center_of_mass = np.mean(coords, axis=0)
        centered_coords = coords - center_of_mass
        
        for i in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(i, centered_coords[i])
        
        logger.info(f"Centered molecule from {file_name} with center of mass: {center_of_mass}")
        return mol, centered_coords, center_of_mass
    
    except ValueError as e:
        logger.error(f"Error processing {file_name}: No 3D conformation - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {file_name}: {str(e)}")
        return None

def compute_molecular_features(centered_mol_tuples, output_dir):
    """
    Compute molecular features for a list of centered molecules, including Morgan fingerprints.
    
    Args:
        centered_mol_tuples (List[Tuple[Chem.Mol, np.ndarray, np.ndarray]]): List of tuples containing 
            centered molecule, centered coordinates, and center of mass.
        output_dir (str): Directory for logging purposes.
    
    Returns:
        pd.DataFrame: DataFrame containing features for all molecules, or None if no features are computed.
    """
    all_feats = []
    # Check the structure of first element in the list
    logging.info(f"Length of tuple[0]: {len(centered_mol_tuples[0])}")
    logging.info(f"Type of tuple[0][0]: {type(centered_mol_tuples[0][0])}")  # helpful debug

# Unpack based on structure
    if len(centered_mol_tuples[0]) == 3 and isinstance(centered_mol_tuples[0][0], tuple):
        for idx, ((mol, centered_coords, center_of_mass), local_features, filename) in enumerate(centered_mol_tuples):
            if mol is None:
                logger.warning(f"Invalid molecule at index {idx} for {filename}")
                continue
            if not isinstance(mol, Chem.Mol):
                logger.warning(f"Expected RDKit Mol, got {type(mol)} for {filename}")
                continue
      
            feats = {}
            try:
                # Shape and inertia descriptors
                feats["RadiusOfGyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
                feats["Asphericity"] = rdMolDescriptors.CalcAsphericity(mol)
                feats["Eccentricity"] = rdMolDescriptors.CalcEccentricity(mol)
                feats["InertialShapeFactor"] = rdMolDescriptors.CalcInertialShapeFactor(mol)
                feats["SpherocityIndex"] = rdMolDescriptors.CalcSpherocityIndex(mol)

                # Principal moments of inertia
                feats["PMI1"] = rdMolDescriptors.CalcPMI1(mol)
                feats["PMI2"] = rdMolDescriptors.CalcPMI2(mol)
                feats["PMI3"] = rdMolDescriptors.CalcPMI3(mol)

                # Normalized principal ratios (NPR)
                feats["NPR1"] = rdMolDescriptors.CalcNPR1(mol)
                feats["NPR2"] = rdMolDescriptors.CalcNPR2(mol)

                # Surface and polar area
                feats["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
                feats["TPSA"] = Descriptors.TPSA(mol)

                # Estimated molecular weight and volume
                feats["ExactMolWt"] = Descriptors.ExactMolWt(mol)
                feats["MolMR"] = Descriptors.MolMR(mol)

                # Morgan fingerprints (ECFP, radius=2) using MorganGenerator
                morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fingerprint = morgan_gen.GetFingerprint(mol)
                feats["MorganFingerprint"] = np.array(fingerprint, dtype=np.int8)

                # Add filename to features
                feats["filename"] = filename
                all_feats.append(feats)
                logger.debug(f"Computed features for molecule {filename}")

            except Exception as e:
                logger.error(f"Error computing features for {filename}: {str(e)}")
                continue

    else:
        for idx, (mol, _, _, filename) in enumerate(centered_mol_tuples):
            feats = {}
            try:
                # Shape and inertia descriptors
                feats["RadiusOfGyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
                feats["Asphericity"] = rdMolDescriptors.CalcAsphericity(mol)
                feats["Eccentricity"] = rdMolDescriptors.CalcEccentricity(mol)
                feats["InertialShapeFactor"] = rdMolDescriptors.CalcInertialShapeFactor(mol)
                feats["SpherocityIndex"] = rdMolDescriptors.CalcSpherocityIndex(mol)
                # Principal moments of inertia
                feats["PMI1"] = rdMolDescriptors.CalcPMI1(mol)
                feats["PMI2"] = rdMolDescriptors.CalcPMI2(mol)
                feats["PMI3"] = rdMolDescriptors.CalcPMI3(mol)
                # Normalized principal ratios (NPR)
                feats["NPR1"] = rdMolDescriptors.CalcNPR1(mol)
                feats["NPR2"] = rdMolDescriptors.CalcNPR2(mol)
                # Surface and polar area
                feats["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
                feats["TPSA"] = Descriptors.TPSA(mol)
                # Estimated molecular weight and volume
                feats["ExactMolWt"] = Descriptors.ExactMolWt(mol)
                feats["MolMR"] = Descriptors.MolMR(mol)
                # Morgan fingerprints (ECFP, radius=2) using MorganGenerator
                morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fingerprint = morgan_gen.GetFingerprint(mol)
                feats["MorganFingerprint"] = np.array(fingerprint, dtype=np.int8)
                # Add filename to features
                feats["filename"] = filename
                all_feats.append(feats)
                logger.debug(f"Computed features for molecule {filename}")
            except Exception as e:
                logger.error(f"Error computing features for {filename}: {str(e)}")
                continue
                
    if not all_feats:
        logger.error("No molecular features computed for any molecule")
        return None
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_feats)
    logger.info(f"Computed features for {len(all_feats)} molecules")
    return features_df
    
# Define hydrophobic atoms (approximate)
HYDROPHOBIC_ATOMS = {'C', 'S', 'F', 'Cl', 'Br', 'I'}

# SMARTS for H-bond donor and acceptor
HBD_SMARTS = Chem.MolFromSmarts('[!H0;#7,#8]')  # Donor: N/O with at least one H
HBA_SMARTS = Chem.MolFromSmarts('[$([O,S;H1;v2]),$([O,S;H0;v2;!$(*~[N,O,S])])]')  # Acceptor

def compute_local_features(mol: Chem.Mol) -> List[Dict]:
    """
    Compute atomic-level shape-based features excluding atom type and 3D properties.

    Returns:
        List[Dict]: List of dictionaries with atomic features.
    """
    if mol is None:
        return []

    mol = Chem.AddHs(mol)

    # Check for valid conformer
    if mol.GetNumConformers() == 0:
        try:
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result != 0:
                logger.warning("Failed to embed 3D conformer.")
                return []
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except ValueError as e:
        logger.warning(f"UFF optimization failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error during UFF optimization: {e}")
        return []

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        logger.warning(f"Gasteiger charge computation failed: {e}")
        return []

    # Identify donor/acceptor atoms
    hbd_matches = set(i[0] for i in mol.GetSubstructMatches(HBD_SMARTS))
    hba_matches = set(i[0] for i in mol.GetSubstructMatches(HBA_SMARTS))

    features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        feature = {
            "AtomicNum": atom.GetAtomicNum(),
            "GasteigerCharge": float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0,
            "FormalCharge": atom.GetFormalCharge(),
            "IsAromatic": atom.GetIsAromatic(),
            "IsHydrophobic": atom.GetSymbol() in HYDROPHOBIC_ATOMS,
            "IsHBDonor": idx in hbd_matches,
            "IsHBAcceptor": idx in hba_matches
        }
        features.append(feature)

    return features
  
def voxelize_molecule(centered_mol_tuple: Tuple[Chem.Mol, np.ndarray, np.ndarray], local_features: List[Dict], file_name: str, grid_size: int = 32, voxel_size: float = 1.0) -> Optional[np.ndarray]:
    """
    Voxelize a centered molecule using Gaussian distributions with multi-channel encoding for atom types and local atomic features.
    
    Args:
        centered_mol_tuple (Tuple[Chem.Mol, np.ndarray, np.ndarray]): Tuple containing centered molecule,
        centered coordinates, and center of mass.
        features (Dict): Dictionary of computed molecular features (used for MorganFingerprint).
        local_features (List[Dict]): List of dictionaries containing local atomic features.
        file_name (str): Name of the SDF file for logging purposes.
        grid_size (int): Number of voxels per dimension (default: 32).
        voxel_size (float): Size of each voxel in Ångströms (default: 1.0).
    
    Returns:
        np.ndarray: Multi-channel 3D voxel grid (grid_size x grid_size x grid_size x (num_atom_types + num_local_features)),
        or None if processing fails.
    """
    mol, centered_coords, _ = centered_mol_tuple

    # Define atom types and local feature keys
    atom_types = ['C', 'N', 'O', 'S', 'H']
    local_feature_keys = [
        "AtomicNum", "GasteigerCharge", "FormalCharge", "IsAromatic",
        "IsHydrophobic", "IsHBDonor", "IsHBAcceptor"
    ]

    num_channels = len(atom_types) + len(local_feature_keys)
    voxel_grid = np.zeros((grid_size, grid_size, grid_size, num_channels), dtype=np.float32)
    
    # Define normalization ranges for local features
    local_feature_stats = {
        "AtomicNum": {"min": 1, "max": 100},  # Atomic numbers range (H=1 to heavier elements)
        "GasteigerCharge": {"min": -1.0, "max": 1.0},  # Typical range for partial charges
        "FormalCharge": {"min": -2, "max": 2},  # Typical formal charge range
        "IsAromatic": {"min": 0, "max": 1},  # Binary (0 or 1)
        "IsHydrophobic": {"min": 0, "max": 1},  # Binary
        "IsHBDonor": {"min": 0, "max": 1},  # Binary
        "IsHBAcceptor": {"min": 0, "max": 1}  # Binary
    }

    # Normalize local features
    try:
        normalized_local_features = []
        for atom_features in local_features:
            normalized_atom = {}
            for key in local_feature_keys:
                if key in atom_features and key in local_feature_stats:
                    raw_val = atom_features[key]
                    min_val = local_feature_stats[key]['min']
                    max_val = local_feature_stats[key]['max']
                    if max_val > min_val:
                        norm_val = (raw_val - min_val) / (max_val - min_val)
                    else:
                        norm_val = 0.0
                    normalized_atom[key] = norm_val
                else:
                    normalized_atom[key] = 0.0  # Fallback for missing keys
            normalized_local_features.append(normalized_atom)
    except Exception as e:
        logger.error(f"Error normalizing local features for {file_name}: {str(e)}")
        return None

    # Define grid boundaries (centered at origin)
    half_box = grid_size * voxel_size / 2
    grid_coords = np.linspace(-half_box, half_box, grid_size)
    
    # Van der Waals radii for common atoms (in Ångströms)
    vdw_radii = {
        'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'H': 1.2, 'F': 1.47,
        'Cl': 1.75, 'Br': 1.85, 'I': 1.98
    }
    
    # Process each atom
    for i, atom in enumerate(mol.GetAtoms()):
        atom_symbol = atom.GetSymbol()
        radius = vdw_radii.get(atom_symbol, 1.5)  # Default radius if not found
        sigma = radius / 4.0  # Gaussian standard deviation based on radius
        
        # Atom coordinates
        x, y, z = centered_coords[i]
        threshold = 1e-2  # Threshold for Gaussian contribution
        
        # Compute Gaussian contribution for each voxel
        for xi, x_val in enumerate(grid_coords):
            for yi, y_val in enumerate(grid_coords):
                for zi, z_val in enumerate(grid_coords):
                    dist_sq = (x - x_val) ** 2 + (y - y_val) ** 2 + (z - z_val) ** 2
                    gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
                    if gaussian < threshold:  # Threshold
                        gaussian = 0.0      
                    
                    # Atom-type channels
                    if atom_symbol in atom_types:
                        channel_idx = atom_types.index(atom_symbol)
                        voxel_grid[xi, yi, zi, channel_idx] += gaussian
                    
                    # Local feature channels
                    if gaussian > threshold:  # Skip tiny values for speed
                        for fi, feature in enumerate(local_feature_keys):
                            if i < len(normalized_local_features):  # Ensure index is valid
                                voxel_grid[xi, yi, zi, len(atom_types) + fi] += gaussian * normalized_local_features[i][feature]
                            else:
                                logger.warning(f"Local feature index {i} out of range for {file_name}")

    logger.debug(f"Voxelized molecule from {file_name} with grid shape: {voxel_grid.shape}")
    return voxel_grid


# def visualize_voxel_grid(voxel_grid, threshold=0.1, channel=None):
#     """
#     Visualize a 3D voxel grid (single channel or sum over all).
    
#     Args:
#         voxel_grid (np.ndarray): 4D voxel grid (X, Y, Z, C)
#         threshold (float): Minimum voxel intensity to show
#         channel (int or None): Index of the channel to show. If None, all channels are summed.
#     """
#     assert voxel_grid.ndim == 4, "Voxel grid must be 4D"

#     if channel is not None:
#         assert 0 <= channel < voxel_grid.shape[3], "Invalid channel index"
#         grid = voxel_grid[..., channel]
#         title = f"Voxel Grid - Channel {channel}"
#     else:
#         grid = voxel_grid.sum(axis=-1)
#         title = "Voxel Grid - Summed Channels"

#     # Apply threshold to make a binary mask
#     mask = grid > threshold

#     if not np.any(mask):
#         print("No voxels above threshold. Try lowering the threshold.")
#         return

#     # Create 3D voxel plot
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.voxels(mask, facecolors='skyblue', edgecolors='gray', alpha=0.6)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(title)
#     plt.tight_layout()
#     plt.show()

# voxel_grid = voxelize_molecule(centered_result, local_features, "file_name")
# visualize_voxel_grid(voxel_grid, threshold=0.05)               # Summed view
# visualize_voxel_grid(voxel_grid, threshold=0.05, channel=0)    # Carbon only
# visualize_voxel_grid(voxel_grid, threshold=0.05, channel=1)    # Nitrogen only

