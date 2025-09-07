import os
import logging
from rdkit import Chem
from typing import List, Tuple, Optional
from Bio.PDB import PDBParser

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('../output/data_processing.log'),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

def load_sdf_files(data_dir: str, sdf_files: Optional[List[str]] = None) -> List[Tuple[Chem.Mol, int, str]]:
    """
    Load SDF files from data directory or a provided list of file paths.
    Handles SDF files with multiple molecules separated by $$$$.
    Returns filenames along with molecule objects.
    
    Args:
        data_dir (str): Path to directory containing SDF files.
        sdf_files (Optional[List[str]]): List of specific SDF file paths to load. If None, load all SDF files in data_dir.
    
    Return: List[Tuple[Chem.Mol, int, str]]: List of tuples containing (molecule, label, filename).
    """
    data = []
    
    try:
        # Validate directory existence
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} does not exist")
            raise FileNotFoundError(f"Directory {data_dir} not found")
        
        # Determine which files to process
        if sdf_files is None:
            sdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.sdf')]
        
        # Process specified SDF files
        logger.info(f"Loading SDF files from data directory: {data_dir}")
        for filepath in sdf_files:
            filename = os.path.basename(filepath)
            try:
                supplier = Chem.SDMolSupplier(filepath, removeHs=False)
                for mol_idx, mol in enumerate(supplier):
                    if mol is not None:
                        data.append((mol, 1, filename))
                        logger.info(f"Loaded molecule {mol_idx+1} from {filepath}")
                    else:
                        logger.warning(f"Failed to load molecule {mol_idx+1} from {filepath}")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded inhibitors {len(data)} molecules")
        return data
    
    except Exception as e:
        logger.error(f"Critical error in load_sdf_files: {str(e)}")
        raise