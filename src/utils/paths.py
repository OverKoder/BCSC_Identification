import os

def path_exists(path:str) -> bool:
    """ 
    Checks if a path exists.

    Args:
        path (str): Path to check.
    
    Raises:
        TypeError: If path is not a string.
    
    Returns:
        bool: True if path exists, False otherwise.
    """
    # Type checking
    if not isinstance(path, str): raise TypeError("path must be a string")

    return True if os.path.isfile(path) else False
