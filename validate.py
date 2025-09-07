import os
import glob
import logging
from rdkit import Chem

logger = logging.getLogger(__name__)

def validate_sdf_files(folder_path):
    """
    Validates .sdf files in the given folder:
    1. Checks if the molecule can be parsed by RDKit.
    2. Checks if it has at least one conformer.
    3. Checks if Z-coordinates are non-zero (true 3D structure).

    Returns:
        bad_files: List of invalid .sdf file names.
    """
    bad_files = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".sdf"):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            suppl = Chem.SDMolSupplier(file_path, removeHs=False)
            mol = next(iter(suppl), None)

            if mol is None:
                logger.warning(f"{file_name}: Cannot be parsed. Skipping.")
                bad_files.append(file_name)
                continue

            if mol.GetNumConformers() == 0:
                logger.warning(f"{file_name}: No conformer found. Skipping.")
                bad_files.append(file_name)
                continue

            conf = mol.GetConformer()
            z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]

            if all(z == 0.0 for z in z_coords):
                logger.warning(f"{file_name}: All Z-coordinates are zero. Not a valid 3D molecule. Skipping.")
                bad_files.append(file_name)
                continue

        except Exception as e:
            logger.warning(f"{file_name}: Error during validation - {e}")
            bad_files.append(file_name)

    if not bad_files:
        logger.info("All SDF files are valid and contain 3D coordinates.")
    else:
        logger.error(f"Invalid SDF files found: {bad_files}")

    return bad_files

def check_existing_voxel_files_and_missing_sdf(data_dir, train_voxel_dir, val_voxel_dir):
    """
    Compare existing .npy voxel files with available .sdf files and determine if preprocessing is needed.

    Returns:
        voxel_done: Boolean indicating if all .sdf files have corresponding voxel files.
        sdf_to_process: List of .sdf file paths that need preprocessing.
        all_voxel_files: List of all existing voxel file paths (train and val).
    """
    # Get all .sdf files
    sdf_files = sorted(glob.glob(os.path.join(data_dir, "*.sdf")))
    sdf_filenames = [os.path.basename(f) for f in sdf_files]
    sdf_indices = {os.path.splitext(os.path.basename(f))[0] for f in sdf_files}

    # Get all training .npy voxel files
    train_voxel_files = sorted(glob.glob(os.path.join(train_voxel_dir, "train_voxel_*.npy")))
    train_voxel_indices = set()
    for vf in train_voxel_files:
        try:
            idx = os.path.splitext(os.path.basename(vf).replace("train_voxel_", ""))[0]
            train_voxel_indices.add(idx)
        except ValueError:
            logger.warning(f"Invalid voxel filename: {vf}")

    # Get all validation .npy voxel files
    val_voxel_files = sorted(glob.glob(os.path.join(val_voxel_dir, "val_voxel_*.npy")))
    val_voxel_indices = set()
    for vf in val_voxel_files:
        try:
            idx = os.path.splitext(os.path.basename(vf).replace("val_voxel_", ""))[0]
            val_voxel_indices.add(idx)
        except ValueError:
            logger.warning(f"Invalid voxel filename: {vf}")

    # Combine all voxel indices
    all_voxel_indices = train_voxel_indices | val_voxel_indices
    all_voxel_files = train_voxel_files + val_voxel_files

    # Determine missing sdf indices
    missing_indices = sdf_indices - all_voxel_indices
    if not missing_indices:
        logger.info("All .sdf files have corresponding voxel files. Skipping preprocessing.")
        return True, [], all_voxel_files

    sdf_to_process = [f for f in sdf_files if os.path.splitext(os.path.basename(f))[0] in missing_indices]
    logger.info(f"{len(sdf_to_process)} .sdf files need preprocessing: {sdf_to_process}")
    return False, sdf_to_process, all_voxel_files