from collections import Counter
import numpy as np
from numpy import ndarray
from typing import Tuple, Union
from math import ceil

import pandas as pd
from torch import Tensor, from_numpy
import pickle as pk
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from tqdm import tqdm


def get_format_index(gene: str, gene_format: dict, ensg_mapping: pd.DataFrame) -> int:
    """
    Given a gene name, returns the index of the column to which the gene expression data
    should be placed. If the gene is not recorded in the gene_format dictionary then it is skipped

    Args:
        gene (str): The name of the gene (accepts ENSG gene name)
        gene_format (dict): Dictionary of the gene format (usually already loaded).
        ensg_mapping (pd.DataFrame): DataFrame which maps ENGS genes to gene names (usually already loaded).

    Raises:
        TypeError: 'gene'is not a str

    Returns:
        index (int): The index of the column where the gene expression data should be placed, returns -1 if 
        the gene is not recorded in the gene_format diciontary

    """

    if not isinstance(gene, str): raise TypeError("'gene' is not a str")

    # First transform 'ENSG' type genes with the mapping
    if gene.startswith('ENSG'):
        
        # Transform 'ENSG'
        try:
            gene = ensg_mapping.loc[gene]['Gene name']

        # If 'ENSG' type gene is not in the mapping it is skipped
        except:
            return -1
        
    # Sometimes there are repeated because of the synonyms, in the case
    # that is not unique we make it unique
    if not isinstance(gene, str):
        gene_list = gene.unique()
    else:
        gene_list = [gene]
        
    for gene in gene_list:
        # Now check if it exists directly in the format dictionary
        try:
            index = gene_format[gene]
            return index
        # Gene is not in the gene_format dictionary
        except:
            pass

        # Check if the gene has synonyms in the dictionary
        synonyms = ensg_mapping[ensg_mapping['Gene Synonym'] == gene]['Gene name'].unique()
        for synonym in synonyms:

            # Check if the synonym exists in the format dictionary
            try:
                index = gene_format[synonym]
                return index
            
            # Synonym does not exist
            except:
                pass
    
    # If the gene is not already in the dictionary, and has no synonyms in it, skip it
    return -1

    # If the 
def format_cells(data: np.ndarray, feature_names: Union[list, np.ndarray], format_dict: str) -> csr_matrix:
    """
    Prepares the data matrix in order to be able to be used by the ML models by making a new data matrix where the genes are
    located in the way that the models accept. Since each dataset measures different genes and records them in different positions
    it makes it impossible for (some) ML Models to compute predictions with different sizes of input and positions of features.

    Args:
        data (np.ndarray, csr_matrix): The data matrix of shape (C x G) where C is the number of cells and G is the number of genes.
        feature_names (list or np.ndarray): List of the gene (or feature) names (e.g. feature_names = ['CD44', 'SOX4']), usually
        taken from anndata.var.
        format_dict (str): Path to a dictionary which marks the format of the cells

    Raises:
        TypeError: 'data' is not a np.ndarray.
        TypeError: 'feature_names' is not a list or np.ndarray.
        TypeError: 'format_dict' is not a str.

    Returns:
        np.ndarray: The new data matrix (with zeroes on the gene that are missing).
    """

    if not (isinstance(data, np.ndarray) or isinstance(data, csr_matrix)): raise TypeError("'data' is not a np.ndarray or a csr matrix.")
    if not (isinstance(feature_names, list) or isinstance(feature_names, np.ndarray)): raise TypeError("'feature_names' is not a list or np.ndarray.")
    if not isinstance(format_dict, str): raise TypeError("'format_dict' is not a str.")

    # Load a dictionary of genes -> index to know where to place the genes in the
    # new data matrix and the ENSG mapping
    format_dict = pk.load(open(format_dict,'rb'))
    ensg_mapping = pd.read_csv('objects/gene_mapping.csv', index_col = 1)

    # Transform the gene names to indexes
    new_genes_indexes, original_genes_indexes = [], []
    for original_index, gene in enumerate(feature_names):

        index = get_format_index(gene, format_dict, ensg_mapping)
        if index != -1:
            new_genes_indexes.append(index)
            original_genes_indexes.append(original_index)

    # New data matrix and convert the previouse one to compressed sparse column for efficiency
    print("Converting to sparse matrix for efficiency...")
    data = csc_matrix(data)
    new_data = lil_matrix((data.shape[0], len(format_dict)), dtype = np.float32)

    # Transfer recorded genes from original data matrix to the new one
    # For some easion scipy crashes when trying to transfer a lot of columns
    # so we transfer them in batches

    for i in tqdm(range((len(original_genes_indexes) // 1000) + 1), desc = "Creating new data matrix..."):
        new_data[:, new_genes_indexes[i * 1000: (i+1) * 1000]] = data[:, original_genes_indexes[i * 1000: (i+1) * 1000]]


    return csr_matrix(new_data)