import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate
import logging
import glob


logger = logging.getLogger(__name__)

class VoxelDataset(Dataset):
    def __init__(self, voxel_dir, augment=False, sdf_filenames=None):
        """
        Initialize the dataset for voxel files.
        
        Args:
            voxel_dir (str): Path to directory containing .npy voxel files.
            augment (bool): Whether to apply data augmentation (random rotations).
            sdf_filenames (List[str]): List of SDF filenames corresponding to voxel files.
        """
        
        self.voxel_dir = voxel_dir
        self.augment = augment
        self.sdf_filenames = sdf_filenames or []
        # Load only voxel files
        self.voxel_files = sorted([f for f in glob.glob(os.path.join(voxel_dir, "*.npy"))])
        
        # Check for NaN/Inf and filter out invalid files
        self.voxel_files, self.sdf_filenames, invalid_files = self._check_and_filter_npy_files(self.voxel_files, self.sdf_filenames)
        
        if not self.voxel_files:
            logger.warning(f"No data .npy files found in {voxel_dir}")
        else:
            logger.info(f"Loaded {len(self.voxel_files)} data voxel files from {voxel_dir}")
        
        if len(self.sdf_filenames) != len(self.voxel_files):
            logger.warning(f"Mismatch between number of voxel files ({len(self.voxel_files)}) and SDF filenames ({len(self.sdf_filenames)})")

    def _check_and_filter_npy_files(self, voxel_files, sdf_filenames):
        """
        Check .npy files for NaN or Inf values, log invalid files, and filter them out.
        
        Args:
            voxel_files (List[str]): List of paths to .npy files.
            sdf_filenames (List[str]): List of corresponding SDF filenames.
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Filtered voxel files, filtered SDF filenames, and list of invalid files.
        """
        valid_voxel_files = []
        valid_sdf_filenames = []
        invalid_files = []
        
        for idx, voxel_path in enumerate(voxel_files):
            try:
                voxel_data = np.load(voxel_path)
                if np.any(np.isnan(voxel_data)) or np.any(np.isinf(voxel_data)):
                    invalid_files.append(os.path.basename(voxel_path))
                    logger.warning(f"NaN or Inf detected in {voxel_path}, discarding")
                    continue
                valid_voxel_files.append(voxel_path)
                if idx < len(sdf_filenames):
                    valid_sdf_filenames.append(sdf_filenames[idx])
            except Exception as e:
                invalid_files.append(os.path.basename(voxel_path))
                logger.error(f"Error loading {voxel_path}: {str(e)}, discarding")
                continue
        
        if invalid_files:
            logger.info(f"Found {len(invalid_files)} invalid .npy files with NaN/Inf: {invalid_files}")
        else:
            logger.info("No invalid .npy files with NaN/Inf found")
        
        return valid_voxel_files, valid_sdf_filenames, invalid_files

    def __len__(self):
        return len(self.voxel_files)
    
    def __getitem__(self, idx):
        try:
            # Load voxel grid
            voxel = np.load(self.voxel_files[idx])
            
            # Ensure voxel is in correct format (C, D, H, W)
            if voxel.ndim == 4:
                voxel = np.transpose(voxel, (3, 0, 1, 2))  # (D, H, W, C) -> (C, D, H, W)
            
            # Apply data augmentation if enabled
            if self.augment:
                voxel = self._augment_voxel(voxel)
            
            # Use SDF filename if available, otherwise fall back to voxel filename
            filename = self.sdf_filenames[idx] if idx < len(self.sdf_filenames) else os.path.basename(self.voxel_files[idx])
            return torch.tensor(voxel, dtype=torch.float32), filename
        
        except Exception as e:
            logger.error(f"Error loading voxel file {self.voxel_files[idx]}: {str(e)}")
            raise
    
    def _augment_voxel(self, voxel):
        """
        Apply random rotations to the voxel grid for data augmentation.
        
        Args:
            voxel (np.ndarray): Voxel grid of shape (C, D, H, W).
        
        Returns:
            np.ndarray: Augmented voxel grid.
        """
        try:
            # Random rotation angles for each axis (-30 to 30 degrees)
            angles = np.random.uniform(-30, 30, 3)
            for axis, angle in enumerate(angles):
                for c in range(voxel.shape[0]):
                    voxel[c] = rotate(voxel[c], angle, axes=(axis, (axis + 1) % 3), reshape=False, mode='constant', cval=0.0)
            return voxel
        except Exception as e:
            logger.error(f"Error during augmentation: {str(e)}")
            return voxel

def create_data_loaders(voxel_dir, batch_size=1, augment_train=False, sdf_filenames=None):
    """
    Create DataLoaders datasets.
    
    Args:
        voxel_dir (str): Path to directory containing .npy voxel files.
        batch_size (int): Batch size for DataLoaders.
        augment_train (bool): Whether to apply augmentation to training data.
        sdf_filenames (List[str]): List of SDF filenames for molecules.
    
    Returns:
        tuple: loader
    """
    try:
        # Create datasets
        dataset = VoxelDataset(voxel_dir, augment=augment_train, sdf_filenames=sdf_filenames)
        
        # Create DataLoaders with custom collate function to handle (tensor, filename) tuples
        def collate_fn(batch):
            tensors, filenames = zip(*batch)
            return torch.stack(tensors), list(filenames)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        logger.info(f"Created DataLoader with {len(dataset)} samples")
        return loader
    
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {str(e)}")
        raise