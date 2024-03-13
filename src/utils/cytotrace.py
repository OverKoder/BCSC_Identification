import cellrank as cr
from cellrank.kernels import CytoTRACEKernel
import scvelo as scv
from scanpy import AnnData
from collections import Counter

class CytoTRACE():

    def __init__(self,anndata: AnnData):
        """
        Constructor for CytoTRACE class

        Args:
            anndata (AnnData): An AnnData object where the data matrix X has cells by columns and genes by rows.
            There must be one extra attribute in the object:
            - In the "var" attribute with the name "annotation" there must be a list of the name of the annotation
            of each cell (cell type)
        """
        self.anndata = anndata
        self.anndata.layers["spliced"] = anndata.X
        self.anndata.layers["unspliced"] = anndata.X

        return
    
    def run(self, n_neighbors: int = 30, n_pcs: int = 30):
        """
        Runs CytoTRACE on the AnnData provided in the constructor

        Args:
            n_neighbors: Number of neighbors to use to compute moments for velocity estimation. 
                First-/second-order moments are computed for each cell across its nearest neighbors.
                Default is 30

            n_pcs:Number of principal components to use. Default is 30.
        """
        self.anndata.layers["spliced"] = self.anndata.X
        self.anndata.layers["unspliced"] = self.anndata.X

        # Computes moments for velocity estimation
        # First-/second-order moments are computed for each cell across its nearest neighbors,
        scv.pp.moments(self.anndata, n_pcs=n_pcs, n_neighbors=n_neighbors)

        # Compute CytoTRACE
        self.ctk = CytoTRACEKernel(self.anndata)
        self.ctk = self.ctk.compute_cytotrace()

        return

    def get_gene_correlation(self):
        """
        Return the pearson correlation index of stemness of each gene
        """
        return self.anndata.var['ct_gene_corr']
    
    def get_score(self):
        """
        Return the CytoTRACE score
        """
        return self.anndata.obs['ct_score']
    
    def get_counts(self, threshold: float = 0.1):
        """
        Returns the count of cells identified with low differentiation potential by CytoTRACE
        under a given threshold

        Args:
            threshold (float): Threshold which marks which cells are taken with low differentiation potential (e.g. a threshold
            of 0.2 means that all cells with CytoTRACE score > 0.2 will be taken with lowldifferentiation potential)

        Returns:
            counts (Counter): The counts
        """

        # Indexes of the cells that have a CytoTRACE score > threshold
        indexes = [index for index in range(len(self.anndata.obs['ct_score'])) if self.anndata.obs['ct_score'][index] <= threshold]

        # Transform cells to annotation
        cells = self.anndata.obs["annotation"][indexes]

        # Return count
        return Counter(cells)