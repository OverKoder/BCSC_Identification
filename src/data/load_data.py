import scanpy

from src.utils.paths import path_exists

def load_data(path: str):
    """
    Loads h5ad data and return the AnnData object

    Args:
        path (str): Path to the data

    Raises:
        TypeError: If path is not a string.
    
    Returns:
        AnnData: The data
    Note: The object weights around 100GB in memory and takes around 10 minutes to load
    """
    data = None
    
    if path_exists(path):
        data = scanpy.read(path)

    return data