from collections import Counter

import pandas as pd
from tqdm import tqdm
from scanpy import AnnData
from scipy.sparse import csr_matrix
import numpy as np 


class ORIGINS():

    def __init__(self, anndata: AnnData, path: str, show_progress: bool = False):
        """
        Constructor for ORIGINS class

        Args:
            anndata (AnnData): An AnnData object where the data matrix X has cells by rows and genes by columns. Also,
            there must be a pandas.core.series.Series object in the "var" attribute with the name "features" (anndata.var['features'])
            which has the names (strings) of the genes recorded in the RNA-sea single cell data.

            Please make sure that ALL genes recorded in the data
            are stored in this Series object, or the output of ORIGINS will be wrong (check if the length of the Series
            equals the number of genes in the data matrix). And please respect the order of genes, make sure each gene
            references the correct column (e.g. anndata.var['features'].values = ['CD44','ALDH'], gene in position 0,
            CD44 means that column 0 in the original data matrix contains the expression of CD44 of all cells)

            Lastly, in the "obs" attribute with the name "annotation" there must be a list of the name of the annotation
            of each cell (cell type). (anndata.obs['annotation']).

            path (str): Path to .csv file with the interactions. The .csv file must have only 2 columns (with names 'V1' and 'V2')
            with the genes that interact by rows (one row is two genes that interact)

            show_progress: Whether to show the progress on screen (True) or not (False)

        Raises:
            ValueError: anndata does not have an attribute "var" with names "features"
        """
        self.data = anndata.X 
        self.annotation = anndata.obs['annotation'].values
        self.df_interactions = pd.read_csv(path, index_col = 0)
        self.show_progress = show_progress

        try:
            self.features = list(anndata.var['features'].values)

        except:
            raise ValueError("Make sure that anndata, has an attribute 'var' with the names 'features' (anndata.var['features'])")

        return
    
    def _get_useful_genes(self):
        """
        When running ORIGINS, most of the genes recorded in the RNA-seq single cell data
        have no interactions recorded (missing genes) in order to reduce spatial complexity (memory)
        we preprocess the data (with the metadata) to take only the genes that have recorded
        interactions.
        """

        useful_genes = []

        if self.show_progress:

            # Go through all the genes that have any interactions recorded and check if they are also
            # in the gene list
            for gene in tqdm(self.df_interactions['V1'].unique(), desc = "Filtering genes..."):

                try:
                    
                    # Check if the gene is in the original data
                    idx = self.features.index(gene)

                    # Append it as a useful gene
                    useful_genes.append((gene, idx))

                except:

                    # If the gene is not in the data, ignore it
                    pass

        else:

            # Go through all the genes that have any interactions recorded and check if they are also
            # in the gene list
            for gene in self.df_interactions['V1'].unique():

                try:
                    
                    # Check if the gene is in the original data
                    idx = self.features.index(gene)

                    # Append it as a useful gene
                    useful_genes.append((gene, idx))

                except:

                    # If the gene is not in the data, ignore it
                    pass
        
        self.useful_genes_name, self.useful_genes_idx = zip(*useful_genes)

        # Generate a dictionary which relates genes to the index of the gene in the original matrix
        self.gene2idx = {key:value for key,value in zip(self.useful_genes_name, range(len(self.useful_genes_name)))}

        # Now, slice the data matrix taking only the useful genes and convert it to a sparse matrix for efficiency
        self.data = self.data[:, self.useful_genes_idx]
        self.data = csr_matrix(self.data)

        return 


    def _create_mask_matrix(self):
        """
        This implementation of ORIGINS uses a mask matrix, which is a matrix where each column has the length of the
        number of useful genes in the data, and contains zeroes in the positions of the genes which do not interact
        with the current gene and ones in the position of the genes that have PPI interaction (e.g. column 0 hypothetically
        represents the mask for gene 'CD44' then it will have ones in the position of all the genes that have a PPI interaction
        with the gene 'CD44' and the rest zeroes)
        """
        mask_matrix = np.zeros((len(self.gene2idx), len(self.gene2idx)))

        if self.show_progress:

            # For all the useful genes
            for gene_i in tqdm(self.gene2idx, desc = "Building mask matrix..."):
                
                # Get the genes that have a PPI interaction with gene_i
                for gene_j in self.df_interactions['V1'][self.df_interactions['V2'] == gene_i].values:

                    try:
                        # If gene_j does interact with gene_i add a 1
                        mask_matrix[self.gene2idx[gene_j]] [self.gene2idx[gene_i]] = 1
                    except:
                        # If gene_j does not interact with gene_i leave the 0
                        pass

        else:

            # For all the useful genes
            for gene_i in self.gene2idx:

                # Get the genes that have a PPI interaction with gene_i
                for gene_j in self.df_interactions['V1'][self.df_interactions['V2'] == gene_i].values:

                    try:
                        # If gene_j does interact with gene_i add a 1
                        mask_matrix[self.gene2idx[gene_j]] [self.gene2idx[gene_i]] = 1
                    except:
                        # If gene_j does not interact with gene_i leave the 0
                        pass

        self.mask_matrix = csr_matrix(mask_matrix)

        return

    def _origins(self):

        result = []

        if self.show_progress:
            for row in tqdm(range(self.data.get_shape()[0]), desc = "Computing ORIGINS..."):

                interactions = self.data[row].dot(self.mask_matrix)

                activity_cell = interactions.dot(self.data[row].transpose())

                result.append(activity_cell.toarray()[0])
        
        else:
            for row in range(self.data.get_shape()[0]):

                interactions = self.data[row].dot(self.mask_matrix)

                activity_cell = interactions.dot(self.data[row].transpose())

                result.append(activity_cell.toarray()[0])

        result = np.array(result)

        min_result = result.min()
        max_result = result.max()

        # Scale the result according to the original algorithm (however the original algorithm)
        # does not guarantee that this does not result in a division by zero
        self.result = (result - min_result) / (max_result - min_result)
        
        return self.result

    def run(self):

        # First, get the useful genes
        self._get_useful_genes()

        # Then, build the mask matrix
        self._create_mask_matrix()

        # Lastly, run origins
        result = self._origins()
        
        return result

    def get_counts(self, threshold: float = 0.1):
        """
        Returns the count of cells identified as lowly differentiated by ORIGINS
        under a given threshold

        Args:
            threshold (float): Threshold which marks which cells are taken as lowly differentiated (e.g. a threshold
            of 0.2 means that all cells with ORIGINS score < 0.2 will be taken as lowly differentiated)

        Returns:
            counts (Counter): The counts
        """
        # Indexes of the cells that have a ORIGINS score < threshold
        indexes = [index for index in range(len(self.result)) if self.result[index] <= threshold]

        # Transform cells to annotation
        cells = self.annotation[indexes]

        # Return count
        counts = Counter(cells)

        return counts