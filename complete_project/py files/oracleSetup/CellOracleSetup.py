import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import setuptools
import site
from CellOracle import celloracle as co

import logging



"""This class is to setup celloracle with the base grn and train it on the scRNA-seq data
it can load it in the base grn, but not the links! this is to just setup celloracle with the scRNA-seq data (or load in) and retrain the cluster speciric grns and save  to a file
inference and other stuff and loading in back links is done somewhere else"""

class Setup:

    def __init__(self, base_grn:pd.DataFrame, scRNA_dat:sc.AnnData, cluster_name:str, embedding_name:str,output_dir:str,load_dir:str,load_from_file_cellOc:bool = True, load_from_file_links:bool = True):
        logging.info("Creating CellOracle object")
        self.load_dir = load_dir
        self.base_GRN  = base_grn
        self.scRNA_dat = scRNA_dat
        self.cluster_name = cluster_name
        self.embedding_name = embedding_name
        self.ouput_dir = output_dir
        self.celloracle = self._create_cell_oc(load_from_file_cellOc)
        print("CellOracle object created, now creating cluster specific grns")
        print("This may take a while")
        self.cluster_grns = self._create_links(load_from_file_links)



    def _create_cell_oc(self,  load_from_file:bool = True) -> co.Oracle:
        if load_from_file:
            return self._load_cell_oc()
        else:
            return self._initialize_cell_oc()

    def _create_links(self, load_from_file:bool = True) -> co.Links:
        if load_from_file:
            return self._load_links()
        else:
            return self._construct_network_specific_grns()

    def _load_cell_oc(self) -> co.Oracle:
        return co.load_hdf5(os.path.join(self.load_dir, 'cellOC.celloracle.oracle'))

    def _load_links(self) -> co.Links:
        if os.path.exists(os.path.join(self.load_dir, 'filtered_links.celloracle.links')):
            return co.load_hdf5(os.path.join(self.load_dir, 'filtered_links.celloracle.links'))
        return co.load_hdf5(os.path.join(self.load_dir, 'links.celloracle.links'))

    def _initialize_cell_oc(self) -> co.Oracle:
        celloracle = co.Oracle()
        n_cells_downsample = 30000
        if self.scRNA_dat.shape[0] > n_cells_downsample:
            # Let's dowmsample into 30K cells
            sc.pp.subsample(self.scRNA_dat, n_obs=n_cells_downsample, random_state=123)

        celloracle.import_anndata_as_normalized_count(adata=self.scRNA_dat, cluster_column_name=self.cluster_name, embedding_name=self.embedding_name)
        celloracle.import_TF_data(TF_info_matrix=self.base_GRN)
        # Select important PCs
        celloracle.perform_PCA_mod()
        # plt.plot(np.cumsum(celloracle.pca.explained_variance_ratio_)[:100])
        # n_comps = np.where(np.diff(np.diff(np.cumsum(celloracle.pca.explained_variance_ratio_)) > 0.002))[0][0]
        # plt.axvline(n_comps, c="k")
        # plt.show()
        n_comps = 50
        print(n_comps)
        n_comps = min(n_comps, 50)
        n_cell = celloracle.adata.shape[0]
        print(f"cell number is :{n_cell}")
        k = int(0.025 * n_cell)
        print(f"Auto-selected k is :{k}")
        celloracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=4)
        celloracle.to_hdf5(os.path.join(self.ouput_dir,"cellOC.celloracle.oracle"))
        return celloracle

    def _construct_network_specific_grns(self) -> co.Links:
        links = self.celloracle.get_links(cluster_name_for_GRN_unit=self.cluster_name, alpha=10, verbose_level=10)
        links.to_hdf5(file_path=os.path.join(self.ouput_dir, 'links.celloracle.links'))
        links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
        links.to_hdf5(file_path=os.path.join(self.ouput_dir, 'filtered_links.celloracle.links'))
        print("links comp done")
        return links

    # def get_cluster_grns(self) -> mco.Links:
    #     return self.cluster_grns

    # def count_base_grn_overlap(self, base_grn: pd.DataFrame, scRNA_anndata: sc.AnnData) -> int:
    #     """
    #     Count how many genes overlap between the base GRN data frame (via 'gene_short_name')
    #     and the scRNA-seq AnnData object's var.index.
    #
    #     Parameters:
    #     -----------
    #     base_grn : pd.DataFrame
    #         Must contain a column named 'gene_short_name' representing target genes.
    #     scRNA_anndata : sc.AnnData
    #         Single-cell RNA-seq data. The AnnData object should have var.index
    #         (or var["symbol"]) containing the gene identifiers.
    #
    #     Returns:
    #     --------
    #     overlap_count : int
    #         Number of genes that overlap between 'base_grn["gene_short_name"]'
    #         and 'scRNA_anndata.var.index'.
    #     """
    #     # Extract unique gene names from base_grn's "gene_short_name" column
    #     grn_genes = set(base_grn["gene_short_name"].unique())
    #
    #     # Extract gene names from the AnnData object's var index
    #     adata_genes = set(scRNA_anndata.var.index.values)
    #
    #     # Determine the intersection
    #     overlap = grn_genes.intersection(adata_genes)
    #
    #     # Return the size of the intersection
    #     return len(overlap)
