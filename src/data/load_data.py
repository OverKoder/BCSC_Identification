from typing import Union

from src.ml.sparsedataset import SparseDataset
from src.data.process_data import format_cells

import scanpy
import numpy as np
from sklearn.model_selection import train_test_split
from torch import Generator, from_numpy
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.sparse import csr_matrix, csc_matrix, load_npz

def load_data(path: str):
    """
    Loads an AnnData object with the provided path

    Args:
        path (str): Path to the data.

    Raises:
        TypeError: If path is not a string.

    Returns
        scipy.sparse.csr_matrix: The data as a sparse matrix.
    """

    if not isinstance(path, str): raise TypeError("'path' is not a string.")

    # Use scanpy
    if path.endswith('h5ad'):

        # Load h5ad
        data = scanpy.read_h5ad(path).X

        # Transform to csr
        if not isinstance(data, csr_matrix):
            data = csr_matrix(data)

    # Load with scipy
    elif path.endswith('npz'):

        # Load npz
        data = load_npz(path)

        # Transform to csr
        if not isinstance(data, csr_matrix):
            data = csr_matrix(data)

    return data.astype(np.float32)
    

def build_labels(annotation: Union[list, np.ndarray], positives: list):
    """
    Given the annotation of the cells in the data, builds the labels
    used in the Machine Learning models.

    Args:
        annotation (list): Annotation of the cells (usually extracted from anndata.obs['annotation'] or similar)
        positives (list): List of the cells that should be taking as positive (stem-cell like) examples. The list must
        contain the names (str) of such cells in the annotation.

    Raises:
        TypeError: 'annotation' is not a list.
        TypeError: 'positives' is not a list.
        ValueError: The names contained in 'positives' are not in 'annotation'.

    Returns:
        labels: (np.ndarray) array containing the labels for the models.
    """
    if not (isinstance(annotation, list) or isinstance(annotation, np.ndarray)): raise TypeError("'annotation' is not a list or a np.ndarray.")
    if not isinstance(positives, list): raise TypeError("'positives' is not a list.")
    if not all([name in annotation for name in positives]): raise ValueError("The names inside 'positives' are not inside the annotation list, please double check.")
    
    return np.array([[1 if annot in positives else 0 for annot in annotation]]).T.astype(np.float32)

def get_gene_slice(feature_names: Union[list, np.ndarray], gene_slice: Union[list, np.ndarray]):
    """
    Given the feature (genes) names and the gene slice (list of genes) returns a list of indexes to slice
    the original data matrix

    Args:
        feature_names (list or np.ndarray): List of the gene (or feature) names (e.g. feature_names = ['CD44', 'SOX4']), usually
        taken from anndata.var.
        gene_slice (list or np.ndarray): List of the gene (or feature) names (e.g. gene_slice = ['CD44']), to slice, genes not in feature_names
        will not be used to slice.

    Raises:
        TypeError: 'feature_names' is not a list or np.ndarray.
        TypeErrpr: 'gene_slice' is not a list or np.ndarray.
    
    Returns: 
        gene_slice (list): List of indexes of the genes (columns) to slice
    """

    if not (isinstance(feature_names, list) or isinstance(feature_names, np.ndarray)): raise TypeError("'feature_names' is not a list or np.ndarray.")
    if not (isinstance(gene_slice, list) or isinstance(gene_slice, np.ndarray)): raise TypeError("'gene_slice' is not a list or np.ndarray.")

    # Make a dictionary of gene:index for efficiency
    feature_dict = {gene:index for gene, index in zip(feature_names, range(len(feature_names)))}
    
    index_list = []
    for gene in gene_slice:

        try:
            # Add gene to slice
            index_list.append(feature_dict[gene])
        except:
            pass

    return index_list

def get_dataloaders(
        data_path: str, annotation: Union[list, np.ndarray], positives: list, ratios: list, feature_names: Union[list, np.ndarray],
        batch_size: int, cell_slice: Union[list, np.ndarray] = None, gene_slice: Union[list, np.ndarray] = None,
        do_format_cells: bool = True, random_state: int = 42
    ):
    """
    Loads and returns the dataloaders for PyTorch

    Args:
        data_path (str): Path to the data (an .h5ad or .mtx). WARNING: In the AnnData data matrix the cells must be located by rows
        and the genes by columns.
        annotation (list or np.ndarray): Annotation of the cells (usually extracted from anndata.obs['annotation'] or similar)
        positives (list): List of the cells that should be taking as positive (stem-cell like) examples. The list must
        contain the names (str) of such cells in the annotation.
        ratios (list): List containing the ratios of the dataset split (e.g. ratios = [0.7, 0.15, 0.15] means 70% training data, 15% validation
        and 15% test).
        feature_names (list or np.ndarray): List of the gene (or feature) names (e.g. feature_names = ['CD44', 'SOX4']), usually
        taken from anndata.var.
        batch_size (int): Batch size for experiments.
        cell_slice (list or np.ndarray): Array containing integer indexes to slice the original data matrix into a subset of it, in order to match
        the annotation (if not provided uses the whole data matrix). Default is None.
        gene_slice (list or np.ndarray): Array containing string which are the genes that will be sliced from the original data matrix into a subset
        of it. Default is None.
        do_format_cells (bool): Whether to call the full_format preprocessing function for cells or not.
        random_state (int): Seed for randomization. Default is 42.

    Raises:
        TypeError: 'data_path' is not a string.
        TypeError: 'annotation' is not a list.
        TypeError: 'positives' is not a list.
        TypeError: 'ratios' is not a list or does not have length == 3.
        TypeError: 'feature_names' is not a list or np.ndarray.
        TypeError: 'batch_size is not an int.
        TypeError: 'cell_slice' is not a list or np.ndarray.
        TypeError: 'gene_slice' is not a list or np.ndarray.
        TypeError: 'do_format_cells' is not a bool.
        TypeError: 'random_state is not an int.

    Returns
        train_dataloader (DataLoader): Training dataloader.
        val_dataloader (DataLoader): Validation dataloader.
        test_dataloader (DataLoader): Test dataloader.
    """
    if not isinstance(data_path, str): raise TypeError("'data_path' is not a string.")
    if not (isinstance(annotation, list) or isinstance(annotation, np.ndarray)): raise TypeError("'annotation' is not a list or a np.ndarray.")
    if not isinstance(positives, list): raise TypeError("'positives' is not a list.")
    if not (isinstance(ratios, list) and len(ratios) == 3): raise TypeError("'ratios' is not a list or does not have length == 3.")
    if not (isinstance(feature_names, list) or isinstance(feature_names, np.ndarray)): raise TypeError("'feature_names' is not a list or np.ndarray.")
    if not isinstance(batch_size, int): raise TypeError("'batch_size is not an int")
    if cell_slice is not None and not (isinstance(cell_slice, list) or isinstance(cell_slice, np.ndarray)): raise TypeError("'cell_slice' is not a list or np.ndarray")
    if gene_slice is not None and not (isinstance(gene_slice, list) or isinstance(gene_slice, np.ndarray)): raise TypeError("'gene_slice' is not a list or np.ndarray")
    if not isinstance(do_format_cells, int): raise TypeError("'do_format_cells is not a bool")
    if not isinstance(random_state, int): raise TypeError("'random_state is not an int")

    # Load sparse data matrix
    sparse_data = load_data(data_path)

    # Get labels
    labels = build_labels(annotation, positives)
    
    # Slice data if need, in order to match some samples with the annotation provided
    if cell_slice is not None:
        sparse_data = sparse_data[cell_slice, :]

    # Slice genes if needed, to ignore some genes or make some tests.
    if gene_slice is not None:
        gene_slice = get_gene_slice(feature_names, gene_slice)
        sparse_data = sparse_data[:, gene_slice]

    # Format the cells
    if do_format_cells:
        sparse_data = format_cells(sparse_data, feature_names, show_progress = True)

    # Check if there are the same samples as labels
    if sparse_data.shape[0] != labels.shape[0]:
        print("WARNING! The number of samples in the dataset: " + str(sparse_data.shape[0]) + " does not match the number of labels: " + str(labels.shape[0]))

    # Split data
    train_dataset, val_dataset, test_dataset = random_split(SparseDataset(sparse_data, labels), lengths = ratios, generator = Generator().manual_seed(random_state))

    # Create dataloaders
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    
    return train_dataloader, val_dataloader, test_dataloader