from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import random, coo_matrix, csr_matrix, vstack
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

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)
    
def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch