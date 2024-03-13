from collections import Counter
import numpy as np
from numpy import ndarray
from typing import Tuple, Union
from scanpy import AnnData, pp, tl
from math import ceil

import pandas as pd
from torch import Tensor, from_numpy
import pickle as pk
from scipy.sparse import csc_matrix
from tqdm import tqdm


def get_gene_index(data: AnnData, gene_name: str) -> int:
    """
    Returns the gene index give the name of the gene

    Args:
        data (AnnData): The data
        gene_name: The gene name

    Raises:
        TypeError: The data is not an AnnData object
        TypeError: gene_name is not an string
    """

    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")
    if not isinstance(gene_name, str): raise TypeError("'gene_name' must be an string")
    
    try:
        return int(data.var[data.var['_index'] == gene_name.upper()].index[0])
    
    except:
        raise IndexError('Gene does not exist.')

def get_gene_nonzero_values(data: AnnData, gene_name: str) -> int:
    """
    Returns the amount of cells (rows) in the data that have nonzero values of
    a given gene

    Args:
        data (AnnData): The data
        gene_name: The gene name:

    Raises:
        TypeError: The data is not an AnnData object
        TypeError: gene_name is not an string

    Returns:
        count (int): The count of nonzero values of a given gene
        percentage (float): Percentage of cells that have non-zero value of that gene
    """

    # Get gene index in data
    gene_idx = get_gene_index(data, gene_name)

    # Count number of non_zeros
    count = (data.X[:, gene_idx] != 0).sum()

    return count, round(count / data.X.shape[0], ndigits = 3) * 100

def run_louvain(data: AnnData, 
    n_neighbors: int, n_pcs: int, random_state_neigh: int, metric: str,
    resolution: float, random_state_lou: int):
    """
    Runs the Louvain clustering algorithm on the single cell data.

    Args: 
        n_neighbors (int): Number of neighbors to compute per cell.
        n_pcs (int): Number of PCA dimensions.
        random_state_neigh (int): Seed for initialization for neighbor computing.
        metric (str): A metric that returns a distance consult scanpy.pp.neighbors to see the available metrics
        resolution (float): controls the granularity of the community detection process. Higher values make more and smaller clusters with similar
        data points, lower values make less and larger cluster with high variability.
        random_state_lou (int): Seed for initialization of optimization algorithm.

    Raises:
        TypeError: n_neighbors is not an int.
        TypeError: n_pcs is not an int.
        TypeError: random_state_neigh is not an int.
        TypeError: metric is not a str.
        TypeError: resolution is not a float.
        TypeError: random_state_lou is not an int.
        ValueError: n_neighbors has to be higher than 0 and lower than number of samples.
        ValueError: n_pcs has to be higher than 0 but lower than number of features.

    Returns:
        Neighbor data is stored in data.ubs['neighbors'], distances in data.obsp['distances'], this stores distances for each pair of neighbors.
        Connectivities in data.obsp['connectivities'], Weights should be interpreted as connectivities.
        Louvain clustering, stored in data.obs['louvain']
    """

    if not isinstance(n_neighbors, int): raise TypeError('n_neighbors must be an int.')
    if not isinstance(n_pcs, int): raise TypeError('n_pcs must be an int.')
    if not isinstance(random_state_neigh, int): raise TypeError('random_state_neigh must be an int.')
    if not isinstance(metric, str): raise TypeError('n_neighbors must be a str.')
    if not isinstance(resolution, float): raise TypeError('resolution must be a float.')
    if not isinstance(random_state_lou, int): raise TypeError('random_state_lou must be an int.')

    if not n_neighbors > 0 and n_neighbors <= data.X.shape[0]: raise ValueError('n_neighbors must be between 0 and the number of samples.')
    if (not n_pcs > 0 and n_pcs <= data.X.shape[1]) and n_pcs is not None: raise ValueError('n_pcs must be between 0 and the number of features.')

    # Compute neighbors
    pp.neighbors(data, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state_neigh, metric = metric)

    # Run Louvain
    tl.louvain(data, resolution = resolution, random_state=random_state_lou)

    return

def mean_cluster_values(data: AnnData, path: str, gene_list: list):
    """
    Generates a .csv file containing the mean values of each gene in all clusters.
    The Louvain algorithm must have been executed before calling this function.

    Args:
        data (AnnData): The data
        path (str): Path to save the .csv file
        gene_list: List of genes to compute mean

    Raises:
        TypeError: The data is not an AnnData object
        TypeError: path is not an string
        TypeError: gene_list is not a list
        RuntimeError: The Louvain algorithm has not been run before this functions
    """
    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")
    if not isinstance(path, str): raise TypeError("'path' must be an string")
    if not isinstance(gene_list, list): raise TypeError("'gene_list' must be an string")

    try:
        data.obs['louvain']
    except:
        raise RuntimeError("The Louvain algorithm must be run before calling this function: scanpy.pp.louvain")
    
    # DataFrame
    df = {key:[] for key in data.obs['louvain'].cat.categories}

    # Get number of clusters
    for cluster_id in data.obs['louvain'].cat.categories:

        # Get all cells from same cluster
        slice = data.X[data.obs['louvain'] == cluster_id]

        # For all genes
        for gene in gene_list:

            try:
                mean = slice[:, get_gene_index(data, gene)].mean()
                df[cluster_id].append(round(mean,3))

            # Gene does not exist in data, a NaN is then placed in that position
            except:
                df[cluster_id].append('NaN')

    # Convert to DataFrame
    df = pd.DataFrame(df)
    df.index = gene_list

    # Save DataFrame
    df.to_csv(path + '.csv')
    return

def fold_change_values(data: AnnData, mean_path:str, save_path: str):
    """
    Generates a .csv file containing the fold change values.
    The Louvain algorithm must have been executed before calling this function.

    Args:
        data (AnnData): The data
        mean_path (str): Path to the .csv file with the mean values
        save_path (str): Path to save the .csv file

    Raises:
        TypeError: The data is not an AnnData object
        TypeError: path is not an string
        TypeError: path is not an string
        RuntimeError: The Louvain algorithm has not been run before this functions
    """
    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")
    if not isinstance(mean_path, str): raise TypeError("'mean_path' must be an string")
    if not isinstance(save_path, str): raise TypeError("'save_path' must be an string")
    
    try:
        data.obs['louvain']
    except:
        raise RuntimeError("The Louvain algorithm must be run before calling this function: scanpy.pp.louvain")
    
    # DataFrames
    df_mean = pd.read_csv(mean_path, index_col=0)
    df_save = {key:[] for key in data.obs['louvain'].cat.categories}

    # Iterate rows (each gene)
    for row in df_mean.iterrows():
        
        # Convert row to numpy for processing
        row = row[1].to_numpy()

        for idx in range(len(row)):
            fold_change = row[idx] - ((row.sum() - row[idx]) / (len(row) - 1))
            df_save[str(idx)].append(round(fold_change,3))

    

    # Convert to DataFrame
    df_save = pd.DataFrame(df_save)
    df_save.index = df_mean.index

    # Save DataFrame
    df_save.to_csv(save_path + '.csv')
    return

def format_cells(data: np.ndarray, feature_names: Union[list, np.ndarray], show_progress: bool = False) -> np.ndarray:
    """
    Prepares the data matrix in order to be able to be used by the ML models by making a new data matrix where the genes are
    located in the way that the models accept. Since each dataset measures different genes and records them in different positions
    it makes it impossible for (some) ML Models to compute predictions with different sizes of input and positions of features.

    Args:
        data (np.ndarray): The data matrix of shape (C x G) where C is the number of cells and G is the number of genes.
        feature_names (list or np.ndarray): List of the gene (or feature) names (e.g. feature_names = ['CD44', 'SOX4']), usually
        taken from anndata.var.
        show_progress (bool): Whether to show progress or not

    Raises:
        TypeError: 'data' is not a np.ndarray.
        TypeError: 'feature_names' is not a list or np.ndarray.
        TypeError: 'show_progress' is not a bool.

    Returns:
        np.ndarray: The new data matrix (with zeroes on the gene that are missing).
    """

    if not isinstance(data, np.ndarray): raise TypeError("'data' is not a np.ndarray.")
    if not (isinstance(feature_names, list) or isinstance(feature_names, np.ndarray)): raise TypeError("'feature_names' is not a list or np.ndarray.")
    if not isinstance(show_progress, bool): raise TypeError("'show_progress' is not a bool.")

    # Load a dictionary of genes -> index to know where to place the genes in the
    # new data matrix
    format_dict = pk.load(open('objects/all_genes_dict.pk','rb'))
    
    # Transform the gene names to indexes
    new_genes_indexes, original_genes_indexes = [], []
    for index, gene in enumerate(feature_names):

        # Check if the gene exists in the format_dict
        try:
            new_genes_indexes.append(format_dict[gene])
            original_genes_indexes.append(index)
            
        # If the gene is not recorded, pass
        except:
            pass

    # New data matrix and convert the previouse one to compressed sparse column for efficiency
    if show_progress:
        print("Converting to sparse matrix for efficiency...")
    data = csc_matrix(data)
    new_data = csc_matrix(np.zeros((data.shape[0], len(format_dict))))

    # Transfer recorded genes from original data matrix to the new one
    # For some easion scipy crashes when trying to transfer a lot of columns
    # so we transfer them in batches

    if show_progress:
        for i in tqdm(range((len(original_genes_indexes) // 1000) + 1), desc = "Creating new data matrix..."):
            new_data[:, new_genes_indexes[i * 1000: (i+1) * 1000]] = data[:, original_genes_indexes[i * 1000: (i+1) * 1000]]

    else:
        for i in range((len(original_genes_indexes) // 1000) + 1):
            new_data[:, new_genes_indexes[i * 1000: (i+1) * 1000]] = data[:, original_genes_indexes[i * 1000: (i+1) * 1000]]

    return new_data.toarray().astype(np.float32)