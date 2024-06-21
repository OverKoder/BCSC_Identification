from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import random, coo_matrix, csr_matrix
from scipy.sparse import vstack as csr_vstack
from tqdm import tqdm

class SparseDataset(Dataset):
    """
    Custom Dataset class using scipy sparse matrices, the implementation requires the data to by
    in CSR format.
    """
    def __init__(self,
        data: csr_matrix, 
        labels: np.ndarray, 
        ):
        
        if not isinstance(data, csr_matrix): raise TypeError("'data' must be a CSR matrix.")
        if not isinstance(labels, np.ndarray): raise TypeError("'labels' must be a np.ndarray.")

        self.data = data
        self.labels = labels

    def __getitem__(self, index: int):
        return self.data[index,:].toarray()[0], self.labels[index]

    def __len__(self):
        return self.data.shape[0]
    
    def update_data_and_labels(self, new_data: csr_matrix, new_labels: np.ndarray):
        """
        Updates the data and labels of the dataset and replaces them

        Args:
            new_data (csr_matrix): New data matrix
            new_labels (np.ndarray): New labels
        """
        if not isinstance(new_data, csr_matrix): raise TypeError("'data' must be a CSR matrix.")
        if not isinstance(new_labels, np.ndarray): raise TypeError("'labels' must be a np.ndarray.")

        self.data = csr_vstack((self.data, new_data))
        self.labels = np.vstack((self.labels, new_labels))

        return

