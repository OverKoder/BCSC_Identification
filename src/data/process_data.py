from scanpy import AnnData, pp, tl
from collections import Counter
from src.utils.paths import path_exists

import numpy as np
import pandas as pd
from numpy import ndarray
from torch import Tensor, from_numpy
from typing import Tuple

from math import ceil
from src.data.globals import LABEL_2_ID1, LABEL_2_ID2, LABEL_2_ID3

def get_min_max_features(data: AnnData) -> Tuple[int, int]:
    """
    Returns the minimum and maximum value of non-zero features of all cells in the data

    Args:
        data (AnnData): The data

    Raises:
        TypeError: The data is not an AnnData object

    Returns:
        min (int): The minimum value of non-zero features
        max (int): The maximum value of non-zero features
    """

    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")

    return data.obs.loc[:,'nFeature_RNA'].min(), data.obs.loc[:,'nFeature_RNA'].max()

def get_class_counts(data: AnnData, id: str) -> dict:
    """
    Returns the number of samples per class

    Args:
        data (AnnData): The data

    Raises:
        TypeError: The data is not an AnnData object
        TypeError: Id is not an string
        ValueError: Id does not have the expected value

    Returns:
        dist (dict): The class distributions
    """
    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")
    if not isinstance(id, str): raise TypeError("'id' must be an string")

    if not id in ['main.ids', 'main.ids.2', 'main.ids.3']: raise ValueError("'id' can only be " + ' or '.join(['main.ids', 'main.ids.2', 'main.ids.3']) )

    return dict(Counter(data.obs.loc[:,id]))

def _get_all_from_class(data: AnnData, id: int, label: int) -> ndarray:
    """
    Returns all samples from a single class

    Args:
        data (AnnData): The data
        id (int): The id of the cell, can only be from 1 to 3.
        label (int): Class label
    """

    if id == 1:
        new_data = data.X[data.obs.loc[:,'main.ids'] == LABEL_2_ID1[label]]
        new_labels = np.full(new_data.shape[0], label)
        return new_data, new_labels

    elif id == 2:
        new_data = data.X[data.obs.loc[:,'main.ids.2'] == LABEL_2_ID2[label]]
        new_labels = np.full(new_data.shape[0], label)
        return new_data, new_labels
    
    else:
        new_data = data.X[data.obs.loc[:,'main.ids.3'] == LABEL_2_ID3[label]]
        new_labels = np.full(new_data.shape[0], label)
        return new_data, new_labels

def get_all_from_classes(data: AnnData, id: int, labels: list, return_tensor: bool) -> ndarray or Tensor:
    """
    Returns all samples from a list of labels
    Args:
        data (AnnData): The data
        id (int): The id of the cell, can only be from 1 to 3.
        label (int): Class label
        return_tensor (bool): Whether to return a numpy array (False) or a torch Tensor (True)
    Raises:
        TypeError: The data is not an AnnData object
        TypeError: id is not an int
        TypeError: labels is not a list
        TypeError: return_tensor is not a bool
        ValueError: id is not within the range 1 to 3

    """

    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")
    if not isinstance(id, int): raise TypeError("'id' must be an integer")
    if not isinstance(labels, list): raise TypeError("'labels' must be a list")
    if not isinstance(return_tensor, int): raise TypeError("'return_tensor' must be an boolean")
    if not id in [1,2,3]: raise ValueError("'id' must can only have value from 1 to 3")
    if not all([label >= 0 and label <= 26 for label in labels]): raise ValueError("All labels must be from 0 to 26")
    
    new_data = np.array([])
    new_labels = np.array([])

    # For each label
    for label in labels:
        add_data, add_labels = _get_all_from_class(data, id, label)

        new_data = np.vstack([new_data, add_data]) if new_data.size else add_data
        new_labels = np.concatenate((new_labels, add_labels))

    if return_tensor:
        return from_numpy(new_data), from_numpy(new_labels)
    
    else:
        return new_data, new_labels

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

def get_nonzero_gene_values(data: AnnData, gene_name: str) -> int:
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
    if not n_pcs > 0 and n_pcs <= data.X.shape[1]: raise ValueError('n_pcs must be between 0 and the number of features.')

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

    if not path_exists(mean_path):
        raise FileNotFoundError(mean_path,"does not exist")
    
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

def split_np_array(array: ndarray, ratios: list) -> ndarray:
    """
    Splits a numpy array according to some ratios in a list

    Args:
        array (ndarray): Array to split
        ratios (list): List of ratios

    Raises
        TypeError: array is not an ndarray
        TypeError: ratios is not a list
        ValueError: ratios does not sum up to 1 or is has not length 3
    """
    if not isinstance(array, ndarray): raise TypeError("'array' must be a ndarray")
    if not isinstance(ratios, list): raise TypeError("'ratios' must be a list")
    
    if not (len(ratios) == 3 and sum(ratios) == 1): raise ValueError("'ratio' must be of length 3 and sum up to 1")

    raise NotImplementedError

def train_test_split(data: AnnData, ratio: list, return_tensor: bool) -> ndarray or Tensor:
    """
    Creates and returns the training, validation and test splits

    Args:
        data (AnnData): The data
        ratio (list): The ratio to split
        return_tensor (bool): Whether to return a numpy array (False) or a torch Tensor (True)

    Raises:
        TypeError: The data is not an AnnData object
        TypeError: ratio is not a list
        TypeError: return_tensor is not a bool
        ValueError: ratio does not sum up to 1

    Returns:
        train_data (ndarray or tensor): The training data
        val_data (ndarray or tensor): The validation data
        test_data (ndarray or tensor): The test data
    """

    if not isinstance(data, AnnData): raise TypeError("'data' must be an AnnData object")
    if not isinstance(ratio, list): raise TypeError("'ratio' must be a list")
    if not isinstance(return_tensor, bool): raise TypeError("'return_tensor' must be a boolean")

    if not (len(ratio) == 3 and sum(ratio) == 1): raise ValueError("'ratio' must be of length 3 and sum up to 1")

    raise NotImplementedError

    # Number of samples
    n_samples = np.array([ i for i in range(data.X.shape[0]) ])

    # Shuffle samples, and add seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(n_samples)

    # Split (return tensor or numpy)
    if return_tensor:
        return from_numpy(data.X[n_samples[ : int( ratio[0] * len(n_samples) ) ]]), from_numpy(data.X[n_samples[ int( ratio[0] * len(n_samples) ): int( ratio[1] * len(n_samples) ) ]]), from_numpy(data.X[n_samples[ int( ratio[1] * len(n_samples) ): int( ratio[2] * len(n_samples) ) ]])

    else:
        return data.X[n_samples[ : int( ratio[0] * len(n_samples) ) ]], data.X[n_samples[ int( ratio[0] * len(n_samples) ): int( ratio[1] * len(n_samples) ) ]], data.X[n_samples[ int( ratio[1] * len(n_samples) ): int( ratio[2] * len(n_samples) ) ]]

