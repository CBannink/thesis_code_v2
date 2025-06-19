# -*- coding: utf-8 -*-

import logging
import warnings
from copy import deepcopy
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from typing import Dict, Any, List, Union, Tuple
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.neighbors import KDTree
from umap import UMAP
import torch
import psutil
import umap
from sklearn.decomposition import PCA



from sklearn.decomposition import PCA

from ..utility.hdf5_processing import dump_hdf5, load_hdf5

from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from .sankey import sankey
from .markov_simulation import _walk
from .oracle_utility import (_adata_to_matrix, _adata_to_df,
                             _adata_to_color_dict, _get_clustercolor_from_anndata,
                             _numba_random_seed, _linklist2dict,
                             _decompose_TFdict, _is_perturb_condition_valid,
                             _calculate_relative_ratio_of_simulated_value,
                             _check_color_information_and_create_if_not_found)
from .oracle_GRN import _do_simulation, _do_simulation_torch,_do_simulation_numpy,_getCoefMatrix, _coef_to_active_gene_list, _shuffle_celloracle_GRN_coef_table, _correct_coef_table, _do_simulation_cupy
from .modified_VelocytoLoom_class import modified_VelocytoLoom
from ..network_analysis.network_construction import get_links
from ..visualizations.oracle_object_visualization import Oracle_visualization
from ..version import __version__

CONFIG = {"N_PROP_MIN": 1,
          "N_PROP_MAX": 5,
          "OOD_WARNING_EXCEEDING_PERCENTAGE": 50}


def update_adata(adata):
    # Update Anndata
    # Anndata generated with Scanpy 1.4 or less should be updated with this function
    # This function will be depricated in the future.

    try:
        lo = adata.uns['draw_graph']['params']['layout']
        if isinstance(lo, np.ndarray):
            lo = lo[0]
        adata.uns['draw_graph']['params']['layout'] = lo
    except:
        pass



def load_oracle(file_path):

    """
    Load oracle object saved as hdf5 file.

    Args:
        file_path (str): File path to the hdf5 file.


    """

    if os.path.exists(file_path):
        pass
    else:
        raise ValueError("File not found. Please check if the file_path is correct.")

    try:
        obj = load_hdf5(filename=file_path, obj_class=Oracle, ignore_attrs_if_err=["knn", "knn_smoothing_w", "pca"])

    except:
        print("Found serious error when loading data. It might be because of discrepancy of dependent library. You are trying to load an object which was generated with a library of different version.")
        obj = load_hdf5(filename=file_path, obj_class=Oracle, ignore_attrs_if_err=["knn", "knn_smoothing_w", "pca"])

        return None
    # Update Anndata
    update_adata(obj.adata)

    return obj


class Oracle(modified_VelocytoLoom, Oracle_visualization):
    """
    Oracle is the main class in CellOracle. Oracle object imports scRNA-seq data (anndata) and TF information to infer cluster-specific GRNs. It can predict the future gene expression patterns and cell state transitions in response to  the perturbation of TFs. Please see the CellOracle paper for details.
    The code of the Oracle class was made of the three components below.

    (1) Anndata: Gene expression matrix and metadata from single-cell RNA-seq are stored in the anndata object. Processed values, such as normalized counts and simulated values, are stored as layers of anndata. Metadata (i.e., Cluster info) are saved in anndata.obs. Refer to scanpy/anndata documentation for detail.

    (2) Net: Net is a custom class in celloracle. Net object processes several data to infer GRN. See the Net class documentation for details.

    (3) VelycytoLoom: Calculation of transition probability and visualization of directed trajectory graph will be performed in the same way as velocytoloom. VelocytoLoom is class from Velocyto, a python library for RNA-velocity analysis. In celloracle, we use some functions in velocytoloom for the visualization.


    Attributes:
        adata (anndata): Imported anndata object
        cluster_column_name (str): The column name in adata.obs containing cluster info
        embedding_name (str): The key name in adata.obsm containing dimensional reduction cordinates

    """

    def __init__(self):
        self.celloracle_version_used = __version__
        self.adata = None
        self.cluster_column_name = None
        self.embedding_name = None
        self.ixs_markvov_simulation = None
        self.cluster_specific_TFdict = None
        self.cv_mean_selected_genes = None
        self.TFdict = {}
        self.coef_matrix_per_cluster= None

        #general variables needed for inference opt
        self.embedding_neighbor_name = None
        self.embedding_neighbor_sparse_name = None
        self.cluster_label_to_idx_dict = None
        self.prev_perturbed_value_batch=None
        self.init_called=False
        self.gene_to_index_dict = None

        self.is_cupy = False
        #if using numpy approach (smaller optimization), is used for cupy as well
        self.X = None
        self.imputed_count = None
        self.coef_matrix_per_cluster_np_dict = None
        self.AI_input = None

        #cupy approach
        self.X_CUPY = None
        self.imputed_count_CUPY = None
        self.coef_matrix_per_cluster_CUPY_dict = None
        self.AI_input_CUPY = None
        self.embedding_knn_sparse_cp = None
        self.umap_neighbors_cp = None
        self.embedding_cp = None
        self.cluster_label_to_idx_dict_cp = None

        #if using torch.tensor approach (currently not viable)
        self.torch_X = None
        self.torch_imputed_count = None
        self.coef_matrix_per_cluster_tensor_dict = None
        # self.save_simulated_counts=None
        # self.save_simulated_counts_single = None


    ############################
    ### 0. utility functions ###
    ############################
    def copy(self):
        """
        Deepcopy itself.
        """
        return deepcopy(self)

    def to_hdf5(self, file_path):
        """
        Save object as hdf5.

        Args:
            file_path (str): file path to save file. Filename needs to end with '.celloracle.oracle'
        """
        if file_path.endswith(".celloracle.oracle"):
            pass
        else:
            raise ValueError("Filename needs to end with '.celloracle.oracle'")

        compression_opts = 5
        dump_hdf5(obj=self, filename=file_path,
                  data_compression=compression_opts,  chunks=(2048, 2048),
                  noarray_compression=compression_opts, pickle_protocol=4)


    def _generate_meta_data(self):
        info = {}
        if hasattr(self, "celloracle_version_used"):
            info["celloracle version used for instantiation"] = self.celloracle_version_used
        else:
            info["celloracle version used for instantiation"] = "NA"

        if self.adata is not None:
            info["n_cells"] = self.adata.shape[0]
            info["n_genes"] = self.adata.shape[1]
            info["status - Gene expression matrix"] = "Ready"
        else:
            info["n_cells"] = "NA"
            info["n_genes"] = "NA"
            info["status - Gene expression matrix"] = "Not imported"

        info["cluster_name"] = self.cluster_column_name
        info["dimensional_reduction_name"] = self.embedding_name

        if hasattr(self, "all_target_genes_in_TFdict"):
            pass
        else:
            if self.adata is not None:
                self._process_TFdict_metadata(verbose=False)
        if self.adata is not None:
            info["n_target_genes_in_TFdict"] = f"{len(self.all_target_genes_in_TFdict)} genes"
            info["n_regulatory_in_TFdict"] = f"{len(self.all_regulatory_genes_in_TFdict)} genes"
            info["n_regulatory_in_both_TFdict_and_scRNA-seq"] = f"{self.adata.var['isin_TFdict_regulators'].sum()} genes"
            info["n_target_genes_both_TFdict_and_scRNA-seq"] = f"{self.adata.var['isin_TFdict_targets'].sum()} genes"


        if len(self.TFdict.keys()) > 0:
            info["status - BaseGRN"] = "Ready"
        else:
            info["status - BaseGRN"] = "Not imported"

        if hasattr(self, "pcs"):
            info["status - PCA calculation"] = "Done"
        else:
            info["status - PCA calculation"] = "Not finished"

        if hasattr(self, "knn"):
            info["status - Knn imputation"] = "Done"
        else:
            info["status - Knn imputation"] = "Not finished"

        if hasattr(self, "k_knn_imputation"):
            info["k_for_knn_imputation"] =  self.k_knn_imputation
        else:
            info["k_for_knn_imputation"] =  "NA"

        if hasattr(self, "coef_matrix_per_cluster") | hasattr(self, "coef_matrix"):
            info["status - GRN calculation for simulation"] = "Done"
        else:
            info["status - GRN calculation for simulation"] = "Not finished"
        return info

    def __repr__(self):
        info = self._generate_meta_data()

        message = "Oracle object\n\n"
        message += "Meta data\n"
        message_status = "Status\n"
        for key, value in info.items():
            if key.startswith("status"):
                message_status += "    " + key.replace("status - ", "") + ": " + str(value) + "\n"
            else:
                message += "    " + key + ": " + str(value) + "\n"

        message += message_status

        return message

    ###################################
    ### 1. Methods for loading data ###
    ###################################
    def _process_TFdict_metadata(self, verbose=True):

        # Make list of all target genes and all reguolatory genes in the TFdict
        self.all_target_genes_in_TFdict, self.all_regulatory_genes_in_TFdict = _decompose_TFdict(TFdict=self.TFdict)

        # Intersect gene between the list above and gene expression matrix.
        self.adata.var["symbol"] = self.adata.var.index.values
        self.adata.var["isin_TFdict_targets"] = self.adata.var.symbol.isin(self.all_target_genes_in_TFdict)
        self.adata.var["isin_TFdict_regulators"] = self.adata.var.symbol.isin(self.all_regulatory_genes_in_TFdict)

        #n_target = self.adata.var["isin_TFdict_targets"].sum()
        #if n_target == 0:
            #print("Found no overlap between TF info (base GRN) and your scRNA-seq data. Please check your data format and species.")
        if verbose:
            n_reg = self.adata.var["isin_TFdict_regulators"].sum()
            if n_reg == 0:
                print("Found No overlap between TF info (base GRN) and your scRNA-seq data. Please check your data format and species.")

            elif n_reg < 50:
                print(f"Total number of TF was {n_reg}. Although we can go to the GRN calculation with this data, but the TF number is small." )



    def import_TF_data(self, TF_info_matrix=None, TF_info_matrix_path=None, TFdict=None):
        """
        Load data about potential-regulatory TFs.
        You can import either TF_info_matrix or TFdict.
        For more information on how to make these files, please see the motif analysis module within the celloracle tutorial.

        Args:
            TF_info_matrix (pandas.DataFrame): TF_info_matrix.

            TF_info_matrix_path (str): File path for TF_info_matrix (pandas.DataFrame).

            TFdict (dictionary): Python dictionary of TF info.
        """

        if self.adata is None:
            raise ValueError("Please import scRNA-seq data first.")

        if len(self.TFdict) != 0:
            print("TF dict already exists. The old TF dict data was deleted. \n")

        if not TF_info_matrix is None:
            tmp = TF_info_matrix.copy()
            tmp = tmp.drop(["peak_id"], axis=1)
            tmp = tmp.groupby(by="gene_short_name").sum()
            self.TFdict = dict(tmp.apply(lambda x: x[x>0].index.values, axis=1))

        if not TF_info_matrix_path is None:
            tmp = pd.read_parquet(TF_info_matrix_path)
            tmp = tmp.drop(["peak_id"], axis=1)
            tmp = tmp.groupby(by="gene_short_name").sum()
            self.TFdict = dict(tmp.apply(lambda x: x[x>0].index.values, axis=1))

        if not TFdict is None:
            self.TFdict=TFdict.copy()

        # Update summary of TFdata
        self._process_TFdict_metadata()




    def updateTFinfo_dictionary(self, TFdict={}):
        """
        Update a TF dictionary.
        If a key in the new TF dictionary already exists in the old TF dictionary, old values will be replaced with a new one.

        Args:
            TFdict (dictionary): Python dictionary of TF info.
        """

        self.TFdict.update(TFdict)

        # Update summary of TFdata
        self._process_TFdict_metadata()

    def addTFinfo_dictionary(self, TFdict):
        """
        Add new TF info to pre-existing TFdict.
        Values in the old TF dictionary will remain.

        Args:
            TFdict (dictionary): Python dictionary of TF info.
        """
        for tf in TFdict:
            if tf in self.TFdict.keys():
                targets = self.TFdict[tf]
                targets = list(TFdict[tf]) + list(targets)
                targets = np.unique(targets)
                self.TFdict.update({tf: targets})
            else:
                self.TFdict.update({tf: TFdict[tf]})

        # Update summary of TFdata
        self._process_TFdict_metadata()


    def get_cluster_specific_TFdict_from_Links(self, links_object, ignore_warning=False):

        """
        Extract TF and its target gene information from Links object.
        This function can be used to reconstruct GRNs based on pre-existing GRNs saved in Links object.

        Args:
            links_object (Links): Please see the explanation of Links class.

        """
        # Check cluster unit in oracle object is same as cluster unit in links_object
        clusters_in_oracle_object = sorted(self.adata.obs[self.cluster_column_name].unique())
        clusters_in_link_object = sorted(links_object.cluster)
        if (self.cluster_column_name == links_object.name) & (clusters_in_link_object == clusters_in_oracle_object):
            pass
        else:
            if ignore_warning:
                pass
            else:
                raise ValueError("Clustering unit does not match. Please prepare links object that was made with same cluster data.")

        self.cluster_specific_TFdict = {}

        for i in links_object.filtered_links:
            self.cluster_specific_TFdict[i] = _linklist2dict(links_object.filtered_links[i])

    def import_anndata_as_raw_count(self, adata, cluster_column_name=None, embedding_name=None,
                                    transform="natural_log"):
        """
        Load scRNA-seq data. scRNA-seq data should be prepared as an anndata object.
        Preprocessing (cell and gene filtering, dimensional reduction, clustering, etc.) should be done before loading data.
        The method imports RAW GENE COUNTS because unscaled and uncentered gene expression data are required for the GRN inference and simulation.
        See tutorial notebook for the details about how to process scRNA-seq data.

        Args:
            adata (anndata): anndata object that stores scRNA-seq data.

            cluster_column_name (str): the name of column containing cluster information in anndata.obs.
                Clustering data should be in anndata.obs.

            embedding_name (str): the key name for dimensional reduction information in anndata.obsm.
                Dimensional reduction (or 2D trajectory graph) should be in anndata.obsm.

            transform (str): The method for log-transformation. Chose one from "natural_log" or "log2".

        """
        if adata.X.min() < 0:
            raise ValueError("gene expression matrix (adata.X) does not seems to be raw_count because it contains negavive values.")

        if (adata.shape[1] < 1000) | (adata.shape[1] > 4000):
            print(f"{adata.shape[1]} genes were found in the adata. Note that Celloracle is intended to use around 1000-3000 genes, so the behavior with this number of genes may differ from what is expected.")

        # store data
        self.adata = adata.copy()

        # update anndata format
        update_adata(self.adata)

        self.cluster_column_name = cluster_column_name
        self.embedding_name = embedding_name
        self.embedding = self.adata.obsm[embedding_name].copy()

        #if hasattr(self.adata, "raw"):
        #    self.adata.X = self.adata.raw.X.copy()

        # store raw count data
        self.adata.layers["raw_count"] = self.adata.X.copy()

        # log transformation
        if transform == "log2":
            self.adata.X = np.log2(self.adata.X + 1)
        elif transform == "natural_log":
            sc.pp.log1p(self.adata)

        self.adata.layers["normalized_count"] = self.adata.X.copy()

        # update color information
        _check_color_information_and_create_if_not_found(adata=self.adata,
                                                         cluster_column_name=cluster_column_name,
                                                         embedding_name=embedding_name)
        col_dict = _get_clustercolor_from_anndata(adata=self.adata,
                                                  cluster_name=self.cluster_column_name,
                                                  return_as="dict")
        self.colorandum = np.array([col_dict[i] for i in self.adata.obs[self.cluster_column_name]])

        # variable gene detection for the QC of simulation
        n = min(adata.shape[1], 1000) - 1

        self.score_cv_vs_mean(n, plot=False, max_expr_avg=35)
        self.high_var_genes = self.cv_mean_selected_genes.copy()
        self.cv_mean_selected_genes = None

        self.adata.var["symbol"] = self.adata.var.index.values
        self.adata.var["isin_top1000_var_mean_genes"] = self.adata.var.symbol.isin(self.high_var_genes)


    def import_anndata_as_normalized_count(self, adata, cluster_column_name=None, embedding_name=None, test_mode=False):
        """
        Load scRNA-seq data. scRNA-seq data should be prepared as an anndata object.
        Preprocessing (cell and gene filtering, dimensional reduction, clustering, etc.) should be done before loading data.
        The method will import NORMALIZED and LOG TRANSFORMED data but NOT SCALED and NOT CENTERED data.
        See the tutorial for more details on how to process scRNA-seq data.

        Args:
            adata (anndata): anndata object containing scRNA-seq data.

            cluster_column_name (str): the name of column containing cluster information in anndata.obs.
                Clustering data should be in anndata.obs.

            embedding_name (str): the key name for dimensional reduction information in anndata.obsm.
                Dimensional reduction (or 2D trajectory graph) should be in anndata.obsm.

            transform (str): The method for log-transformation. Chose one from "natural_log" or "log2".
        """
        if adata.X.min() < 0:
            raise ValueError("Gene expression matrix (adata.X) contains negavive values. Please use UNSCALED and UNCENTERED data.")

        if (adata.shape[1] < 1000) | (adata.shape[1] > 4000):
            print(f"{adata.shape[1]} genes were found in the adata. Note that Celloracle is intended to use around 1000-3000 genes, so the behavior with this number of genes may differ from what is expected.")

        # Store data
        self.adata = adata.copy()

        # Update anndata format
        update_adata(self.adata)

        self.cluster_column_name = cluster_column_name
        self.embedding_name = embedding_name
        self.embedding = self.adata.obsm[embedding_name].copy()

        # store raw count data
        #self.adata.layers["raw_count"] = adata.X.copy()

        # normalization and log transformation
        self.adata.layers["normalized_count"] = self.adata.X.copy()

        # update color information
        if not test_mode:
            _check_color_information_and_create_if_not_found(adata=self.adata,
                                                             cluster_column_name=cluster_column_name,
                                                             embedding_name=embedding_name)
            col_dict = _get_clustercolor_from_anndata(adata=self.adata,
                                                      cluster_name=self.cluster_column_name,
                                                      return_as="dict")
            self.colorandum = np.array([col_dict[i] for i in self.adata.obs[self.cluster_column_name]])

            # variable gene detection for the QC of simulation
            n = min(adata.shape[1], 1000) - 1

            self.score_cv_vs_mean(n, plot=False, max_expr_avg=35)
            self.high_var_genes = self.cv_mean_selected_genes.copy()
            self.cv_mean_selected_genes = None

            self.adata.var["symbol"] = self.adata.var.index.values
            self.adata.var["isin_top1000_var_mean_genes"] = self.adata.var.symbol.isin(self.high_var_genes)


    def change_cluster_unit(self, new_cluster_column_name):
        """
        Change clustering unit.
        If you change cluster, previous GRN data and simulation data will be delated.
        Please re-calculate GRNs.
        """

        # 1. Check new cluster information exists in anndata.
        if new_cluster_column_name in self.adata.obs.columns:
            _check_color_information_and_create_if_not_found(adata=self.adata,
                                                             cluster_column_name=new_cluster_column_name,
                                                             embedding_name=self.embedding_name)
        else:
            raise ValueError(f"{new_cluster_column_name} was not found in anndata")


        # 2. Reset previous GRN data and simoulation data
        attributes_delete = ['ixs_markvov_simulation', 'colorandum' ,"alpha_for_trajectory_GRN",
                             'GRN_unit', 'coef_matrix_per_cluster',"perturb_condition",
                             'corr_calc', 'embedding_knn', 'sampling_ixs', 'corrcoef', 'corrcoef_random',
                             'transition_prob', 'transition_prob_random', 'delta_embedding', 'delta_embedding_random',
                             'total_p_mass', 'flow_embedding', 'flow_grid', 'flow',
                             'flow_norm', 'flow_norm_magnitude', 'flow_rndm', 'flow_norm_rndm',
                             'flow_norm_magnitude_rndm']

        attributes = list(self.__dict__.keys())
        for i in attributes:
            if i in attributes_delete:
                delattr(self, i)

        # 4. Update cluster info
        self.cluster_column_name = new_cluster_column_name

        # 3. Update color information
        col_dict = _get_clustercolor_from_anndata(adata=self.adata,
                                                  cluster_name=new_cluster_column_name,
                                                  return_as="dict")
        self.colorandum = np.array([col_dict[i] for i in self.adata.obs[new_cluster_column_name]])

    def update_cluster_colors(self, palette):

        """
        Update color information stored in the oracle object.
        The color information is overwritten.
        """

        sc.pl.embedding(self.adata,
                        basis=self.embedding_name,
                        color=self.cluster_column_name,
                        palette=palette)

        col_dict = _get_clustercolor_from_anndata(adata=self.adata,
                                                  cluster_name=self.cluster_column_name,
                                                  return_as="dict")
        self.colorandum = np.array([col_dict[i] for i in self.adata.obs[self.cluster_column_name]])




    ####FUNCTION THAT WE NOT USE### we keep it for future reference
    def _precompute_umap(self, n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=0) :
        """
        Compute umap embedding based on the precomputed pca embedding and save the reducer to transform later shifted pca embeddings to umap space"""

        if "X_pca" not in self.adata.obsm:
            raise ValueError("PCA embedding is missing. Please compute PCA embedding first.")
        sc.pp.neighbors(self.adata, n_neighbors=15)
        sc.tl.umap(self.adata)
        # plot umap
        sc.pl.umap(self.adata, color='celltype')
        # self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state)
        # self.embedding_umap = self.reducer.fit_transform(self.adata.obsm["X_pca"])
        # self.adata.obsm["X_umap"] = self.embedding_umap

    def _precompute_kda_tree_on_pca_embedding(self):
        """
        Precompute KDTree for fast neighbor queries.
        """
        if "X_pca" not in self.adata.obsm:
            raise ValueError("PCA embedding is missing. Please compute PCA embedding first.")
        self._pca_kdtree = KDTree(self.adata.obsm["X_pca"])


    def _precompute_PCA_embedding(self, n_components=50):
        """
        Computes PCA embedding for the AnnData object.
        The PCA embedding is stored in adata.obsm['X_pca'].

        Parameters
        ----------
        n_components: int
            Number of PCA components to compute.
        """
        # Run PCA (if not already computed)
        #test which layer in self.adata is equal to .X
        #error is : ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().
        if "X_pca" not in self.adata.obsm:
            sc.tl.pca(self.adata, n_comps=n_components, svd_solver='arpack')

    def _set_embedding_name(self, embedding_name):
        """
        Set the column name containing cluster information.
        """
        self.embedding_name = embedding_name

    def set_adata(self, adata):
        self.adata = adata
        self.init_called= False
        self._init_np()

        ###END OF NOT USED FUNCTION####

    ####DEBUG FUNCTIONS####
    def analyze_umap(self):
        umap_embedding = self.adata.obsm["X_umap"]  # Get UMAP embedding (assuming it's in adata.obsm['X_umap'])
        umap_min_coords = umap_embedding.min(axis=0)  # Minimum coordinates along each UMAP dimension
        umap_max_coords = umap_embedding.max(axis=0)  # Maximum coordinates along each UMAP dimension

        print(f"UMAP1 Min/Max: {umap_min_coords[0]}, {umap_max_coords[0]}")  # Range of UMAP1
        print(f"UMAP2 Min/Max: {umap_min_coords[1]}, {umap_max_coords[1]}")  # Range of UMAP2

    def return_delta(self):
        return self.delta_embedding

    def plot_umap_shifts_with_multiple_tfs_multiple_perturbs(self, shift_dict, tf_dict):
        """
        Generates a scatter plot of UMAP embedding and visualizes shifts for multiple primary cells.
        Allows for an arbitrary number of shifts starting from multiple cell types.

        Args:
            shift_dict (dict): Dictionary where keys are indices of primary cells and values are lists of shifted coordinates (list of np.ndarray of shape (2,)).
            tf_dict (dict): Dictionary where keys are indices of primary cells and values are lists of TFs being perturbed in sequence.
        """
        # 1. Extract UMAP embedding and cell types
        adata = self.adata
        umap_embedding = adata.obsm["X_umap"]
        cell_types = adata.obs["celltype"]

        # 2. Create scatter plot for all cells without coloring by cell type
        plt.figure(figsize=(10, 8))
        plt.scatter(
            umap_embedding[:, 0],  # UMAP1 coordinates for all cells
            umap_embedding[:, 1],  # UMAP2 coordinates for all cells
            s=10,  # Marker size for background cells
            color="lightgray",  # Uniform color for all background cells
            alpha=0.5  # Transparency for background cells
        )

        # 3. Plot original and shifted primary cells with annotations
        for primary_index, shifted_coords in shift_dict.items():
            original_coord = umap_embedding[primary_index]
            current_coord = original_coord

            # Annotate the starting cell with its cell type
            cell_type = cell_types[primary_index]
            tfs = tf_dict.get(primary_index, ["Unknown TF"])
            # Scatter plot of the starting point (red dot)
            plt.scatter(
                current_coord[0],
                current_coord[1],
                s=100,
                c='red',
                marker='o',
                label='Original Primary Cell' if 'Original Primary Cell' not in plt.gca().get_legend_handles_labels()[
                    1] else None,
            )
            # plt.text(
            #     current_coord[0] + 0.3,
            #     current_coord[1] + 0.3,
            #     f"{cell_type}\n{tfs[0]}",
            #     fontsize=8,
            #     color="black",
            #     ha="center"
            # )

            # Iterate through the list of shifted coordinates and plot transitions
            for i, shifted_coord in enumerate(shifted_coords):
                next_tf = tfs[i] if i < len(tfs) else "Unknown TF"


                # Draw a line connecting the current coordinate to the shifted coordinate
                plt.plot(
                    [current_coord[0], shifted_coord[0]],
                    [current_coord[1], shifted_coord[1]],
                    c='blue',
                    linestyle='--',
                    linewidth=2,
                    label='Shift Vector' if 'Shift Vector' not in plt.gca().get_legend_handles_labels()[1] else None,
                )

                # Scatter plot of intermediate or final shifted coordinate (green triangle)
                plt.scatter(
                    shifted_coord[0],
                    shifted_coord[1],
                    s=50,
                    c='forestgreen',
                    marker='^',
                    label='Shifted Primary Cell' if 'Shifted Primary Cell' not in plt.gca().get_legend_handles_labels()[
                        1] else None,
                )

                # Annotate the intermediate or final point with its TF name
                # plt.text(
                #     shifted_coord[0] + 0.3,
                #     shifted_coord[1] + 0.3,
                #     f"{cell_type}\n{next_tf}",
                #     fontsize=8,
                #     color="black",
                #     ha="center"
                # )

                # Update the current coordinate to the new position
                current_coord = shifted_coord

        # 4. Add labels, legend, and title
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("UMAP Shifts with Multiple TF Perturbations")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")  # Place legend outside the plot
        plt.grid(False)
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal

        # 5. Show the plot
        plt.tight_layout()
        plt.show()

    def plot_umap_shifts_with_multiple_tfs(self, shift_dict, tf_dict):
        """
        Generates a scatter plot of UMAP embedding and visualizes shifts for multiple primary cells.
        Annotates initial cells with their cell type and the corresponding TF being perturbed.

        Args:
            shift_dict (dict): Dictionary where keys are indices of primary cells and values are their shifted coordinates (np.ndarray of shape (2,)).
            tf_dict (dict): Dictionary where keys are indices of primary cells and values are the TFs being perturbed.
        """
        # 1. Extract UMAP embedding and cell types
        adata = self.adata
        umap_embedding = adata.obsm["X_umap"]
        cell_types = adata.obs["celltype"]

        # 2. Create scatter plot for all cells without coloring by cell type
        plt.figure(figsize=(10, 8))
        plt.scatter(
            umap_embedding[:, 0],  # UMAP1 coordinates for all cells
            umap_embedding[:, 1],  # UMAP2 coordinates for all cells
            s=10,  # Marker size for background cells
            color="lightgray",  # Uniform color for all background cells
            alpha=0.5  # Transparency for background cells
        )

        # 3. Plot original and shifted primary cells with annotations
        for primary_index, shifted_coord in shift_dict.items():
            original_coord = umap_embedding[primary_index]

            # Scatter plot of original primary cell coordinate (red dot)
            plt.scatter(
                original_coord[0],
                original_coord[1],
                s=100,
                c='red',
                marker='o',
                label='Original Primary Cell' if 'Original Primary Cell' not in plt.gca().get_legend_handles_labels()[
                    1] else None,
            )

            # Annotate the original primary cell with its cell type and TF name
            cell_type = cell_types[primary_index]
            tf_name = tf_dict.get(primary_index, "Unknown TF")
            plt.text(
                original_coord[0] + 0.3,  # Slight offset to avoid overlap with the point
                original_coord[1] + 0.3,
                f"{cell_type}\n{tf_name}",
                fontsize=8,
                color="black",
                ha="center"
            )

            # Scatter plot of shifted primary cell coordinate (green triangle)
            plt.scatter(
                shifted_coord[0],
                shifted_coord[1],
                s=100,
                c='forestgreen',
                marker='^',
                label='Shifted Primary Cell' if 'Shifted Primary Cell' not in plt.gca().get_legend_handles_labels()[
                    1] else None,
            )

            # Draw a line connecting original and shifted primary cell coordinates
            plt.plot(
                [original_coord[0], shifted_coord[0]],
                [original_coord[1], shifted_coord[1]],
                c='blue',
                linestyle='--',
                linewidth=2,
                label='Shift Vector' if 'Shift Vector' not in plt.gca().get_legend_handles_labels()[1] else None,
            )

        # 4. Add labels, legend, and title
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("UMAP Shifts with Multiple TF Perturbations")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")  # Place legend outside the plot
        plt.grid(False)
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equa
        # 5. Show the plot
        plt.tight_layout()
        plt.show()

    def return_reg_genes(self) -> list:
        return self.all_regulatory_genes_in_TFdict

    def return_active_reg_genes(self) -> list:
        return self.active_regulatory_genes

    def _get_simulated_states_and_perturb_conditions_bulk(self) -> List:
        return self.save_simulated_counts
    ###END DEBUG FUNCTIONS###

    ####INIT FUNCTION NECESSARY TO SETUP CUSTOM CELLORACLE####
    def init(self, embedding_type:str = "X_umap", n_neighbors:int = 200, torch_approach:bool=False, cupy_approach:bool=False, batch_size:int =64):
        if torch_approach:
            self._init_custom_celloracle_tensor(embedding_type=embedding_type, n_neighbors=n_neighbors, gpu_opt=torch.cuda.is_available(), batch_size=batch_size)
        elif cupy_approach:
            self._init_custom_celloracle_cp(embedding_type=embedding_type, n_neighbors=n_neighbors, batch_size=batch_size)
        else:
            self._init_custom_celloracle_np(embedding_type=embedding_type, n_neighbors=n_neighbors, batch_size=batch_size)

    def _init_custom_celloracle_tensor(self, embedding_type:str = "X_umap", n_neighbors:int = 200,gpu_opt:bool = False, batch_size:int = 64):
        """
        Setup custom cellOracle, the custom embedding should already be precalculated and stored in adata.obsm["X_{name}"].
        1) This function calculates the neighbors in the custom embedding space
        2) Converts the layers in adata to torch tensor
        3) Compute gene to index dictionary for fast access
        4) Set torch tensor to GPU if gpu_opt is True

        """
        if not hasattr(self, "init_called"):
            self._create_class_variables()
        if self.init_called:
            raise ValueError("init_custom_celloracle can only be called once.")
        if embedding_type not in ["X_umap", "pca"]:
            raise ValueError("embedding_type should be either 'umap' or 'pca'")
        if f"X_{embedding_type}" not in self.adata.obsm:
            raise ValueError(f"{embedding_type} embedding is missing. Please compute {embedding_type} embedding first.")
        if embedding_type == "X_umap":
            self._init_neighbors_umap(n_neighbors=n_neighbors)
        else:
            self._init_pca_neighbors(n_neighbors=n_neighbors)
        self._init_torch()
        if gpu_opt:
            self._init_torch_gpu()
        #for the coef matrix and initial .X adata they result in the same gene to index dict so we share one complete dict
        self._init_prev_perturbed_list(batch_size=batch_size)
        self._init_gene_to_index_dict()
        self._init_var_reindex()
        self._init_obs_reindex()
        self._init_coef_matrix_tensors(gpu_opt=gpu_opt)
        self._init_cluster_label_to_idx_dict()
        self._normalize_imputed_counts_for_AI_input()

        #init done#
        self.init_called = True

    def _init_prev_perturbed_list(self, batch_size):
        """
        Initialize the list for storing previous perturbed values.
        """
        self.prev_perturbed_value_batch = []
        for i in range(batch_size):
            self.prev_perturbed_value_batch.append([])


    def _init_custom_celloracle_np(self, embedding_type:str = "X_umap", n_neighbors:int = 200, batch_size:int = 64):
        """
        Setup custom cellOracle, the custom embedding should already be precalculated and stored in adata.obsm["X_{name}"].
        1) This function calculates the neighbors in the custom embedding space
        2) Compute gene to index dictionary for fast access
        3) Set torch tensor to GPU if gpu_opt is True

        """
        if not hasattr(self, "init_called") or self.init_called is None or not self.init_called:
            self._create_class_variables()
        else:
            raise ValueError("init_custom_celloracle can only be called once.")


        if embedding_type not in ["X_umap", "pca"]:
            raise ValueError("embedding_type should be either 'umap' or 'pca'")
        if f"X_{embedding_type}" not in self.adata.obsm:
            raise ValueError(f"{embedding_type} embedding is missing. Please compute {embedding_type} embedding first.")
        if embedding_type == "X_umap":
            self._init_neighbors_umap(n_neighbors=n_neighbors)
        else:
            self._init_pca_neighbors(n_neighbors=n_neighbors)
        self._init_prev_perturbed_list(batch_size=batch_size)
        self._init_np()
        self._init_cluster_label_to_idx_dict()
        self._init_gene_to_index_dict()
        self._init_coef_matrix_np()
        self._normalize_imputed_counts_for_AI_input_np()
        # self._init_var_reindex()
        # self._init_obs_reindex()
        self.init_called = True

    def _init_custom_celloracle_cp(self, embedding_type:str = "X_umap", n_neighbors:int = 200, batch_size:int = 64):
        """
        Setup custom cellOracle, the custom embedding should already be precalculated and stored in adata.obsm["X_{name}"].
        1) This function calculates the neighbors in the custom embedding space
        2) Compute gene to index dictionary for fast access
        3) Set torch tensor to GPU if gpu_opt is True

        """
        if not hasattr(self, "init_called") or self.init_called is None or not self.init_called:
            self._create_class_variables()
        else:
            raise ValueError("init_custom_celloracle can only be called once.")

        if embedding_type not in ["X_umap", "pca"]:
            raise ValueError("embedding_type should be either 'umap' or 'pca'")
        if embedding_type not in self.adata.obsm:
            raise ValueError(f"{embedding_type} embedding is missing. Please compute {embedding_type} embedding first.")
        if embedding_type == "X_umap":
            self._init_neighbors_umap(n_neighbors=n_neighbors)
        else:
            self._init_pca_neighbors(n_neighbors=n_neighbors)
        self.is_cupy = True
        self._init_prev_perturbed_list(batch_size=batch_size)
        self._init_cp()
        self._init_cluster_label_to_idx_dict()
        self._convert_idx_dict_to_cp()
        self._init_gene_to_index_dict()
        self._init_coef_matrix_cp()
        self._normalize_imputed_counts_for_AI_input_np()
        self._init_embedding_cp()
        # self._init_var_reindex()
        # self._init_obs_reindex()
        self.init_called = True

    def _init_embedding_cp(self):
        """
        Initialize the embedding for cupy.
        """
        print(self.adata)
        if sp.issparse(self.adata.obsp["umap_neighbors_sparse"]):
            self.embedding_knn_cp = cp.array(self.adata.obsp["umap_neighbors_sparse"].todense())
        else:
            self.embedding_knn_cp  = cp.array(self.adata.obsp["umap_neighbors_sparse"])
        if sp.issparse(self.adata.obsm[self.embedding_name]):
            self.embedding_cp = cp.array(self.adata.obsm[self.embedding_name].todense())
        else:
            self.embedding_cp = cp.array(self.adata.obsm[self.embedding_name])
        if sp.issparse(self.adata.obsm["umap_neighbors"]):
            self.umap_neighbors_cp = cp.array(self.adata.obsm[self.embedding_neighbor_name].todense())
        else:
            self.umap_neighbors_cp = cp.array(self.adata.obsm[self.embedding_neighbor_name])

    def _normalize_imputed_counts_for_AI_input_np(self):
        self.AI_input = self.adata.layers["imputed_count"].copy()
        sc.pp.scale(self.AI_input, copy=False)



    def _init_obs_reindex(self):
        #check if necessary
        if self.adata.obs.index[0] == 0:
            return
        #reindex adata
        self.adata.obs['original_index'] = self.adata.obs.index
        # Create a new index from 0 to len(adata.obs)-1
        self.adata.obs.index = np.arange(len(self.adata.obs))

    def _create_class_variables(self, torch_approach:bool=False, cp_approach=False):
        self.gene_to_index_dict = {}
        self.embedding_neighbor_name = None
        self.embedding_neighbor_sparse_name = None
        self.init_called=False
        self.prev_perturbed_value_batch = []
        self.cluster_label_to_idx_dict = {}
        if torch_approach:
            self._create_class_variables_torch()
        elif cp_approach:
            self._create_class_variables_cp()
        else:
            self._create_class_variables_np()

    def _create_class_variables_np(self):
        """Declare all necessary things for custom numpy optimized celloracle"""
        self.X = None
        self.imputed_count = None
        self.coef_matrix_per_cluster_np_dict = {}
        self.AI_input = None

    def _create_class_variables_cp(self):
        """Declare all necessary things for custom cupy optimized celloracle"""
        self.X_CUPY = None
        self.imputed_count_CUPY = None
        self.coef_matrix_per_cluster_CUPY_dict = {}
        self.AI_input_CUPY = None
        self.embedding_knn_sparse_cp = None
        self.umap_neighbors_cp = Nones
        self.embedding_cp = None
        self.cluster_label_to_idx_dict_cp={}



    def _create_class_variables_torch(self):
        """Declare all necessary things for custom gpu optimized celloracle"""
        self.torch_X = None
        self.torch_imputed_count = None
        self.coef_matrix_per_cluster_tensor_dict = {}

    def _init_cluster_label_to_idx_dict(self):
        for cluster in np.unique(self.adata.obs[self.cluster_column_name]):
            self.cluster_label_to_idx_dict[cluster] = np.where(self.adata.obs[self.cluster_column_name] == cluster)[0]

    def _convert_idx_dict_to_cp(self):
        self.cluster_label_to_idx_dict_cp = { cluster: cp.asarray(indices_np)for cluster, indices_np in self.cluster_label_to_idx_dict.items()}

    def _init_coef_matrix_tensors(self, gpu_opt:bool = False):
        for cluster in self.coef_matrix_per_cluster.keys():
            tensor = torch.tensor(self.coef_matrix_per_cluster[cluster].values, dtype=torch.float32)
            if gpu_opt and torch.cuda.is_available():
                tensor = tensor.cuda()
            self.coef_matrix_per_cluster_tensor_dict[cluster] = tensor

    def _init_coef_matrix_np(self):
        """
        Convert coef_matrix_per_cluster to numpy array.
        """
        self.coef_matrix_per_cluster_np_dict = {cluster: self.coef_matrix_per_cluster[cluster].to_numpy() for cluster in self.coef_matrix_per_cluster.keys()}

    def _init_coef_matrix_cp(self):
        """
        Convert coef_matrix_per_cluster to numpy array.
        """
        self.coef_matrix_per_cluster_CUPY_dict = {cluster: cp.array(self.coef_matrix_per_cluster[cluster].to_numpy()) for cluster in self.coef_matrix_per_cluster.keys()}
    
    def _init_np(self):
        """
        Convert layers in adata to numpy array.
        """
        self.X = self.adata.X.toarray() if sp.issparse(self.adata.X) else self.adata.X

        # Convert imputed_count to NumPy array if it's a sparse matrix
        self.imputed_count = self.adata.layers["imputed_count"].toarray() if sp.issparse(
            self.adata.layers["imputed_count"]) else self.adata.layers["imputed_count"]

    def _init_cp(self):
        """
        Convert layers in adata to numpy array.
        """
        adata_X = self.adata.X
        if sp.issparse(adata_X):
            self.X_CUPY = cp.array(adata_X.todense())
        else:
            self.X_CUPY = cp.array(np.asarray(adata_X))
        adata_imputed_count = self.adata.layers["imputed_count"]
        if sp.issparse(adata_imputed_count):
            self.imputed_count_CUPY = cp.array(adata_imputed_count.todense())
        else:
            self.imputed_count_CUPY = cp.array(np.asarray(adata_imputed_count))


    def _init_torch(self):
        """
        Convert layers in adata to torch tensor.
        """
        self.torch_X = self._convert_to_torch(self.adata.X)
        self.torch_imputed_count = self._convert_to_torch(self.adata.layers["imputed_count"])

    def _convert_to_torch(self, data:np.ndarray) -> torch.Tensor:
        """
        Convert any object to Tensor, also possible for sparse matrices
        """
        if sp.issparse(data):
            data = data.todense()
        return torch.tensor(data, dtype=torch.float32)

    def _init_torch_gpu(self):
        """
        Set torch tensor to GPU.
        """
        if torch.cuda.is_available():
            self.torch_X = self.torch_X.cuda()
            self.torch_imputed_count = self.torch_imputed_count.cuda()

    def _init_gene_to_index_dict(self):
        """
        Compute gene to index dictionary for fast access.
        """
        self.gene_to_index_dict = {gene: i for i, gene in enumerate(self.adata.var.index)}

    def _init_var_reindex(self):
        """
        Compute gene to index dictionary for fast access.
        """
        self.adata.var['original_index'] = self.adata.var.index
        # Create sequential indices for genes/features
        self.adata.var.index = np.arange(len(self.adata.var))

    def _init_neighbors_umap(self, n_neighbors:int=200):
        """
        Compute neighbors in UMAP space.
        """
        #THIS NEAREST NEIGHBOR DOES NOT INCLUDE ITSELF
        self.knocked_out_TFs = set()
        if "X_umap" not in self.adata.obsm:
            raise ValueError("UMAP embedding is missing. Please compute UMAP embedding first.")
        #need to include one additional as nearestneighbors includes itself, so we do 200+1 to get 200 actual neighbors
        neighbors_umap = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1, algorithm="kd_tree", metric="euclidean")
        umap = np.array(self.adata.obsm["X_umap"])
        neighbors_umap.fit(umap)
        neighbor_indices= neighbors_umap.kneighbors(return_distance=False)
        self.adata.obsm["umap_neighbors"] = neighbor_indices
        self.adata.obsp["umap_neighbors_sparse"] = neighbors_umap.kneighbors_graph(mode="connectivity")
        self.adata.uns["umap_neighbors_sparse"] = self.adata.obsp["umap_neighbors_sparse"].copy()
        self._pca_kdtree = KDTree(self.adata.obsm["X_umap"])
        self.embedding_neighbor_name = "umap_neighbors"
        self.embedding_neighbor_sparse_name = "umap_neighbors_sparse"

    def _init_pca_neighbors(self, n_neighbors=200):
        """
        Precomputes and stores PCA neighbors for all cells in adata.
        Stores results in both list format (obsm) and sparse matrix format (uns/obsm).
        """
        import random
        random.seed(1)
        # 1. Make sure PCA is computed
        self._precompute_PCA_embedding(n_components=50)

        # 2. Compute KNN using scikit-learn (handles both neighbor indices and sparse matrix)
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1)
        knn.fit(self.adata.obsm["X_pca"])

        # Get neighbor indices (including self)
        neighbor_indices_raw = knn.kneighbors(return_distance=False)
        # Remove self-references from indices (first column if it's self, last column if not)
        neighbor_indices = np.array([
            row[1:] if row[0] == i else row[:-1]
            for i, row in enumerate(neighbor_indices_raw)
        ])

        # 3. Store neighbor indices as array in obsm
        self.adata.obsm["pca_neighbors"] = neighbor_indices

        # 4. Compute and store sparse connectivity matrix
        sparse_matrix = knn.kneighbors_graph(mode="connectivity")
        self.adata.uns["pca_neighbors_sparse"] = sparse_matrix
        self.adata.obsp["pca_neighbors_sparse"] = sparse_matrix

        # 6. Prepare KDTree for fast neighbor queries
        self._pca_kdtree = KDTree(self.adata.obsm["X_pca"])
        self.embedding_neighbor_name = "pca_neighbors"
        self.embedding_neighbor_sparse_name = "pca_neighbors_sparse"

    ####END OF INIT FUNCTIONALITY####


    ####Necessary get function to retrieve info, private functions, can be used to debug outside of this####
    def _get_neighbors(self, cell_ix):
        """
        Retrieve precomputed nearest neighbors for a given cell index.
        """
        if self.embedding_neighbor_name not in self.adata.obsm:
            raise ValueError(f"{self.embedding_neighbor_name} is missing. Cannot compute nearest neighbor.")
        return self.adata.obsm[self.embedding_neighbor_name][cell_ix]

    def _get_neighbors_cupy(self, cell_ix):
        """
        Retrieve precomputed nearest neighbors for a given cell index.
        """
        if self.embedding_neighbor_name not in self.adata.obsm:
            raise ValueError(f"{self.embedding_neighbor_name} is missing. Cannot compute nearest neighbor.")
        return self.umap_neighbors_cp[cell_ix]

    def _get_pca_neighbors(self, cell_ix):
        """
        Retrieve precomputed nearest neighbors for a given cell index.
        """
        if "pca_neighbors" not in self.adata.obsm:
            self.precompute_pca_neighbors()
        return self.adata.obsm["pca_neighbors"][cell_ix]

    def _get_umap_neighbors(self, cell_ix):
        """
        Retrieve precomputed nearest neighbors for a given cell index.
        """
        if "umap_neighbors" not in self.adata.obsm:
            self.precompute_umap_neighbors()
        return self.adata.obsm["umap_neighbors"][cell_ix]

    def _get_post_perturb_nn(self, embedding_state, original_cell_ix):
        """
        TODO do a check that you do not retreive the same cell
        Finds the closest existing cell in PCA space for a *new* cell state.
        Returns index of the nearest neighbor in adata.
        """
        if not hasattr(self, "_pca_kdtree") or self._pca_kdtree is None:
            raise ValueError("PCA KDTree is missing. Cannot compute nearest neighbor.")
        #check if reshaping is needed
        if len(embedding_state.shape) == 1:
            embedding_state = embedding_state.reshape(1, -1)
        # 4. Query the KDTree for the nearest neighbor (k=1)
        dist, ind = self._pca_kdtree.query(embedding_state, k=2)
        return ind[0][0]

        # #this is for debugging
        # if ind[0][0] == original_cell_ix:
        #     return ind[0][0]
        # if ind[0][1] == ind[0][0]:
        #     print("the nearest neighbor of the original cell is also the nearest neighbor of the shifted cell")
        # #return closest neighbor, currently not implemented to not check if we return the same cell (it is its own closest neighbor TODO)
        # return ind[0][0]

    def get_genes(self)->list:
        return self.gene_to_index_dict.keys()

    def get_AI_input_for_cell_indices(self, cell_indices:list, gene_indices:list=None)->np.ndarray:
        """
        Get the AI input for a list of cell indices and gene indices.
        If gene_indices is None, all genes are returned.
        """
        if gene_indices is None:
            return self.AI_input[cell_indices]
        return self.AI_input[cell_indices][:, gene_indices]

    def get_gene_index(self, gene:str)->int:
        return self.gene_to_index_dict[gene]

    def _convert_pca_to_umap(self, pca_embedding):
        """
        Convert pca embedding to umap embedding using the precomputed reducer.
        """
        # if not hasattr(self, "reducer"):
        #     raise ValueError("UMAP reducer is not computed. Please compute UMAP first.")
        #check if 2d, does the shape consists of 2 elements
        if not hasattr(self, "umap"):
            self.umap = UMAP(n_components=2, n_neighbors=15, a = self.adata.uns['umap']["params"]["a"], b = self.adata.uns['umap']["params"]["b"], random_state=1)
            self.umap.fit(self.adata.obsm["X_pca"])
        if len(pca_embedding.shape) == 1:
            pca_embedding = pca_embedding.reshape(1, -1)
        new_coords = self.umap.transform(pca_embedding)

        return new_coords
    ####END OF GET FUNCTIONS###


    ####THE MAIN INFERENCE FUNCTIONS CALLED BY THE AI, CHOOSE BETWEEN BATCH OR NON BATCH####
    def training_phase_inference(self, perturb_condition, idx, n_propagation, n_min=None,n_max=None,clip_delta_X=False, sigma_corr=0.05, threads = 1, calc_random=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function does everything required for inference during training
            Step 1: Get pca/umap neighbors (500) for the idx
            Step 2: Perform shift for the idx and its neighbors using simulate_shift_subset
            Step 3: Calculate the transition probabilities for the primary idx and its neighbors? or only for primary idx based on neighbors, not sure
            Step 4: Calculate the vector shift in the PCA embedding space (if possible) for the primary idx based on its neighbors
            Step 5: Calculate the new PCA embedding for the primary idx based on the shift vector
            Step 6: Return the cell in our data closest to the new PCA embedding of the primary idx

            RETURNS: The index of the cell in our data that is closest to the new PCA embedding of the primary idx"""
        #CURRENTLY PCA NEIGHBORS RETRIEVES THE TOP 200 NEIGHBORS!!!!
        #CURRENTLY FOR PCA, CAN BE CHANGE TO UMAP BY CHANGING THE GET NEIGHBOR FUNCTION AND THE VARIABLES IN estimate_transition_prob_subset FUNCTION BY RSETTING EMBEDDING CORRECTLY
        neighbor_idxs = self._get_umap_neighbors(idx)
        all_idx = np.concatenate(([idx], neighbor_idxs))
        simulated_states = self.simulate_shift_subset(perturb_condition= perturb_condition, GRN_unit="cluster", subset_idx=all_idx,n_propagation=n_propagation,n_min=n_min,n_max=n_max,clip_delta_X=clip_delta_X)
        simulated_states_np = simulated_states.to_numpy()
        self.estimate_transition_prob_subset(adata_subset=self.adata[all_idx], delta_X=simulated_states_np,indx =  all_idx,calculate_randomized=calc_random, threads=threads)
        shift_in_embedding = self.calculate_embedding_shift_sub(sigma_corr=sigma_corr)
        new_umap_coords = self.adata.obsm["X_umap"][idx] + shift_in_embedding[0]
        closest_neighbor_idx = self._get_post_perturb_nn(new_umap_coords, idx)
       # self.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shift_dict={idx: [new_umap_coords]}, tf_dict={idx: "Atf6"})
        return self.adata.obsm["X_umap"][idx], closest_neighbor_idx, self.adata.obsm["X_umap"][closest_neighbor_idx],new_umap_coords, shift_in_embedding[0]


    def training_phase_inference_batch(self, batch_size:int, idxs:list, perturb_condition:list, n_neighbors:int, knockout_prev_used:bool=False,n_propagation:int=3,
                                       n_min=None, n_max=None, clip_delta_X=False, sigma_corr=0.05, threads=4,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function does everything required for inference during training
            Step 1: Get pca/umap neighbors (500) for the idxs
            Step 2: Perform shift for the idx and its neighbors using simulate_shift_subset
            Step 3: Calculate the transition probabilities for the primary idx and its neighbors? or only for primary idx based on neighbors, not sure
            Step 4: Calculate the vector shift in the PCA embedding space (if possible) for the primary idx based on its neighbors
            Step 5: Calculate the new PCA embedding for the primary idx based on the shift vector
            Step 6: Return the cell in our data closest to the new PCA embedding of the primary idx

            RETURNS: (Original embedding coords, closest neighbor idx, closest neighbor coords, new coords, shift vector)
                        """

        # allow for faster batch computation within the AI agent training, so we want to mass perform a single matrix multiplicatio to speed up process
        all_indices = np.empty((batch_size, n_neighbors + 1), dtype=int)
        view_data = self.get_view_data_and_apply_prev_perturbed_values(idxs=idxs, batch_size=batch_size, n_neighbors=n_neighbors, all_indices=all_indices, knockout_prev_used=knockout_prev_used)
        starttime  = datetime.now()
        simulation_result_batch = self.simulate_shift_subset_batch_numpy(view_data = view_data,perturb_condition=perturb_condition, GRN_unit="cluster",batch_idxs=all_indices, n_propagation=n_propagation, n_min=n_min, n_max=n_max, clip_delta_X=clip_delta_X)
        print("batch numpy: " ,datetime.now()-starttime)
        starttime = datetime.now()
        self.estimate_transition_prob_opt_numpy_batch(X=view_data, delta_X=simulation_result_batch, all_original_idxs=all_indices, embedding_name=self.embedding_name, sparse_graph_name=self.embedding_neighbor_sparse_name, threads=threads)
        print("estimate transition took numpy: ", datetime.now()-starttime)
        starttime = datetime.now()
        shift_in_embedding = self.calculate_embedding_shift_numpy_batch(sigma_corr=sigma_corr)
        print("shift_in_embedding opt numpy took: ", datetime.now()-starttime)
        self._add_perturbs_to_prev_perturb_list(perturb_conditions=perturb_condition)
        closest_coords = np.empty((batch_size, 2))
        closest_neighbor_idxs = []
        shifts = []
        abs_shift = np.abs(shift_in_embedding)
        print("average shift for batch: ", np.mean(abs_shift))
        for i in range(batch_size):
            shifts.append(shift_in_embedding[i][0])
            new_coords_based_on_shift = self.adata.obsm[self.embedding_name][all_indices[i][0]] + shift_in_embedding[i][0]
            if np.any(np.isnan(new_coords_based_on_shift)):
                raise ValueError("New coordinates based on shift contain NaN values. Please check the input data and simulation parameters idxs: ", idxs, " with tfs: ", perturb_condition)
            closest_coords[i] = new_coords_based_on_shift
            closest_neighbor_idx = self._get_post_perturb_nn(new_coords_based_on_shift, all_indices[i][0])
            closest_neighbor_idxs.append(closest_neighbor_idx)
        return self.adata.obsm[self.embedding_name][idxs], closest_neighbor_idxs, self.adata.obsm[self.embedding_name][closest_neighbor_idxs], closest_coords, shifts

    def training_phase_inference_batch_cp(self, batch_size: int, idxs: list, perturb_condition: list, n_neighbors: int,knockout_prev_used: bool = False, n_propagation: int = 3,n_min=None, n_max=None, clip_delta_X=False, sigma_corr=0.05, threads=4, ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This function does everything required for inference during training
            Step 1: Get pca/umap neighbors (500) for the idxs
            Step 2: Perform shift for the idx and its neighbors using simulate_shift_subset
            Step 3: Calculate the transition probabilities for the primary idx and its neighbors? or only for primary idx based on neighbors, not sure
            Step 4: Calculate the vector shift in the PCA embedding space (if possible) for the primary idx based on its neighbors
            Step 5: Calculate the new PCA embedding for the primary idx based on the shift vector
            Step 6: Return the cell in our data closest to the new PCA embedding of the primary idx

            RETURNS: (Original embedding coords, closest neighbor idx, closest neighbor coords, new coords, shift vector)
                        """

        # allow for faster batch computation within the AI agent training, so we want to mass perform a single matrix multiplicatio to speed up process
        idxs_cpy = cp.array(np.array(idxs))
        all_indices = cp.empty((batch_size, n_neighbors + 1), dtype=int)

        view_data = self.get_view_data_and_apply_prev_perturbed_values_cp(idxs=idxs_cpy, batch_size=batch_size,n_neighbors=n_neighbors, all_indices_cp=all_indices,knockout_prev_used=knockout_prev_used)
        simulation_result_batch = self.simulate_shift_subset_batch_cp(view_data=view_data,perturb_condition=perturb_condition,GRN_unit="cluster", batch_idxs=all_indices,n_propagation=n_propagation, n_min=n_min,n_max=n_max, clip_delta_X=clip_delta_X)
        self.estimate_transition_prob_opt_cupy_batch(X_cp=view_data, delta_X_cp=simulation_result_batch,all_original_idxs=all_indices, embedding_name=self.embedding_name,sparse_graph_name=self.embedding_neighbor_sparse_name,threads=threads)
        shift_in_embedding_cp = self.calculate_embedding_shift_cupy_batch(sigma_corr=sigma_corr)
        self._add_perturbs_to_prev_perturb_list(perturb_conditions=perturb_condition)

        shift_in_embedding_np = shift_in_embedding_cp.get()
        #get absolute shift
        all_indices_np = all_indices.get()

        closest_coords = np.empty((batch_size, 2))
        closest_neighbor_idxs = []
        shifts = []

        for i in range(batch_size):
            primary_idx_np = all_indices_np[i, 0]
            shifts.append(shift_in_embedding_np[i][0])
            new_coords_based_on_shift = self.adata.obsm[self.embedding_name][primary_idx_np] + shift_in_embedding_np[i][0]
            if np.any(np.isnan(new_coords_based_on_shift)):
                print("New coordinates based on shift contain NaN values. Please check the input data and simulation parameters idxs: ",idxs, " with tfs: ", perturb_condition)

            closest_coords[i] = new_coords_based_on_shift
            closest_neighbor_idx = self._get_post_perturb_nn(new_coords_based_on_shift, primary_idx_np)
            closest_neighbor_idxs.append(closest_neighbor_idx)
        return self.adata.obsm[self.embedding_name][idxs], closest_neighbor_idxs, self.adata.obsm[self.embedding_name][closest_neighbor_idxs], closest_coords, shifts
    ###END NEW CODE FOR UMAP SHIFT INFERENCE###

    def _add_perturbs_to_prev_perturb_list(self, perturb_conditions):
        """
        Add perturbations to the previous perturbations list.
        """
        for i in range(len(self.prev_perturbed_value_batch)):
            self.prev_perturbed_value_batch[i].append(perturb_conditions[i])



    def get_view_data_and_apply_prev_perturbed_values(self, idxs:np.ndarray, batch_size:int, n_neighbors:int,all_indices:np.ndarray,knockout_prev_used:bool=False) -> np.ndarray:
        """
        Retrieve view data for the given batch indices and apply previous perturbations if specified. ALL_INDICES IS AN EMPTY ARRAY THAT WE ALSO FILL UP TO USE DOWNSTREAM (not the best practice but it works)
        """
        view_data = np.empty((batch_size, n_neighbors + 1, self.adata.n_vars))
        # Fill the array with each index and its neighbors
        for index, idx in enumerate(idxs):
            neighbor_idxs = self._get_neighbors(idx)
            all_indices[index, 0] = idx  # Place idx at first position
            all_indices[index, 1:] = neighbor_idxs  # Add neighbors afterward
            # COPYING is not necessary as we use fancy inexing in the simulate shift function which automatically copies, so here copying is unnecessary IF PREV PERTURBS IS NOT A THING SO WE NEED TO MOVE THAT AFTER INDEXING
            # IN THE SHIFT FUNCTION ITSELF AND NOT HERE!!!
            view_data_ = self.imputed_count[all_indices[index]]
            if not knockout_prev_used:
                # Fill the view_data array with the imputed counts for this batch
                view_data[index] = view_data_
                continue
            # apply prev perturbs
            prev_perturbs = self.prev_perturbed_value_batch[index]
            for tuple in prev_perturbs:
                condition, value = tuple
                if condition in self.gene_to_index_dict:
                    index_for_gene = self.gene_to_index_dict[condition]
                    # are we also adjusting the gem view for this? it should be as the standard view should look like this and we do not just do this major change, however, this will minimize effect again?
                    # futhermore, IF WE ADJUST IT GEM_VIEW NEEDS TO BE A COPY!
                    view_data_[:, index_for_gene] = value
            view_data[index] = view_data_
        return view_data

    def get_view_data_and_apply_prev_perturbed_values_cp(self,idxs: cp.ndarray,batch_size:int,n_neighbors:int,all_indices_cp: cp.ndarray,knockout_prev_used: bool = False) -> cp.ndarray:
        """Retrieve view data for the given batch indices and apply previous perturbations if specified. ALL_INDICES IS AN EMPTY ARRAY THAT WE ALSO FILL UP TO USE DOWNSTREAM (not the best practice but it works)"""
        view_data_cp = cp.empty((batch_size, n_neighbors + 1, self.adata.n_vars), dtype=cp.float32)
        for index in range(batch_size):
            idx_cp = idxs[index]
            neighbor_idxs_cp = self._get_neighbors_cupy(idx_cp)
            all_indices_cp[index, 0] = idx_cp
            all_indices_cp[index, 1:] = neighbor_idxs_cp
            current_indices_cp = all_indices_cp[index]
            view_data_slice_cp = self.imputed_count_CUPY[current_indices_cp].copy() #copy is necessary with cupy i think?
            if not knockout_prev_used:
                # Fill the view_data array with the imputed counts for this batch
                view_data_cp[index] = view_data_slice_cp
                continue
            if knockout_prev_used:
                prev_perturbs = self.prev_perturbed_value_batch[index]
                for condition, value in prev_perturbs:
                    if condition in self.gene_to_index_dict:
                        index_for_gene = self.gene_to_index_dict[condition]
                        view_data_slice_cp[0, index_for_gene] = value
            view_data_cp[index] = view_data_slice_cp

        return view_data_cp

    def reset_info_during_training_for_batch_instance(self, batch_instances):
        for i in batch_instances:
            self.prev_perturbed_value_batch[i] = []

    def simulate_shift_subset_batch_numpy(self,view_data:np.ndarray,perturb_condition: list, GRN_unit: str, batch_idxs: np.ndarray,
                                               n_propagation=3, n_min=None, n_max=None, clip_delta_X=False) -> np.ndarray:
        if GRN_unit != "cluster":
            raise ValueError("Currently only cluster GRN is supported for batch simulation")
        if n_min is None:
            n_min = CONFIG["N_PROP_MIN"]
        if n_max is None:
            n_max = CONFIG["N_PROP_MAX"]
        if GRN_unit != "cluster":
            raise ValueError("Currently only cluster GRN is supported for batch simulation")

        for tuple in perturb_condition:
            condition,value=tuple
            if condition not in self.all_regulatory_genes_in_TFdict:
                raise ValueError(
                    f"Gene {condition} is not included in the base GRN; It is not TF or TF motif information is not available. Cannot perform simulation.")

            if condition not in self.active_regulatory_genes:
                raise ValueError(
                    f"Gene {condition} does not have enough regulatory connection in the GRNs. Cannot perform simulation.")


        #TODO: DO WE NEED TO CLEAR THE SIMULATION RESULTS? I DO NOT HINK SO BUT KEEP IN MIND

        batch_size, neighbors_size, feature_size = batch_idxs.shape[0], batch_idxs.shape[1], self.adata.n_vars

        # Initialize final_output safely (e.g., with NaN to detect unwritten slots)
        final_output = np.empty((batch_size, neighbors_size, feature_size), dtype=np.float32)

        # Cluster Processing Loop
        for cluster_label, matrix in self.coef_matrix_per_cluster_np_dict.items():
            original_indices_for_cluster = self.cluster_label_to_idx_dict[cluster_label]

            # Store tuples: (batch_idx, original_pos_in_batch_array, num_cells, global_indices_array)
            batch_mapping_info = []
            total_cells_in_cluster =0
            #set all correct start and end indices to slice and set in correct place after, keep track how many cells in cluster to init array size, preventing list making and concatenation later, this is faster
            for b_idx, idxs_in_batch in enumerate(batch_idxs):
                batch_mask = np.isin(idxs_in_batch, original_indices_for_cluster, assume_unique=True)
                #global_indices_for_batch_cluster = idxs_in_batch[batch_mask]
                local_pos_in_batch_for_cluster = np.where(batch_mask)[0]
                num_cells_in_batch = len(local_pos_in_batch_for_cluster)
                if num_cells_in_batch == 0:
                    continue
                batch_mapping_info.append((b_idx, local_pos_in_batch_for_cluster, num_cells_in_batch, total_cells_in_cluster))
                total_cells_in_cluster += num_cells_in_batch


            full_data_for_cluster  = np.empty((total_cells_in_cluster, feature_size), dtype=np.float32)
            full_view_data_for_cluster  = np.empty((total_cells_in_cluster, feature_size), dtype=np.float32)

            
            for b_idx, local_pos, num_cells, total_cells_in_cluster in batch_mapping_info:
                # Find which cells in *this specific batch* belong to the *current cluster*
                end_idx = total_cells_in_cluster + num_cells

                #so this already creates a copy by fancy slicing, so we can slice from the same dataset as it should not adjust the original one
                input_data_slice = view_data[b_idx, local_pos]
                gem_view_slice = view_data[b_idx, local_pos]

                # Apply perturbation specific to this batch
                perturbation, value = perturb_condition[b_idx]
                if perturbation in self.gene_to_index_dict:  # Check if gene exists
                    index_for_gene = self.gene_to_index_dict[perturbation]
                    input_data_slice[:, index_for_gene] = value

                full_data_for_cluster[total_cells_in_cluster:end_idx, :] = input_data_slice
                full_view_data_for_cluster[total_cells_in_cluster:end_idx, :] = gem_view_slice


            # Perform sim
            simulated_result = _do_simulation_numpy(coef_matrix=matrix, simulation_input=full_data_for_cluster,
                                                    n_propagation=n_propagation, gem=full_view_data_for_cluster)

            # Map results back batch by batch using stored info
            for b_idx, local_pos, num_cells, start_idx in batch_mapping_info:
                end_idx = start_idx + num_cells
                result_slice = simulated_result[start_idx:end_idx, :]
                # Place results directly into the final output array
                final_output[b_idx, local_pos, :] = result_slice


        #we may not need to do this as wel can maybe let the sim function only return the delta so we do not need to add up and substract eacht ime saving time
        final_output_substracted = final_output - view_data

        if clip_delta_X:
            #TODO THIS NEEDS TO BE IMPLEMENTED MAY BE NEEDED/INTERESTING
            final_output_substracted = np.clip(final_output_substracted, n_min, n_max)

        return final_output_substracted

    def simulate_shift_subset_batch_cp(self,view_data: cp.ndarray, perturb_condition: list,GRN_unit: str,batch_idxs: cp.ndarray,  n_propagation=3,n_min=None,n_max=None,clip_delta_X=False) -> cp.ndarray:  # Return CuPy array
        """
        Simulates gene expression shifts in batches using CuPy for GPU acceleration.

        Args:
            view_data (cp.ndarray): Base gene expression data for cells (on GPU). Shape (batch_size, neighbors_size, n_features).
            perturb_condition (list): List of perturbation tuples [(gene, value)] corresponding to each batch.
            GRN_unit (str): Specifies GRN level ('cluster').
            batch_idxs (cp.ndarray): Indices mapping cells in view_data back to original adata indices (on GPU). Shape (batch_size, neighbors_size).
            n_propagation (int): Number of simulation steps.
            n_min (float, optional): Minimum clip value for delta_X. Defaults to None.
            n_max (float, optional): Maximum clip value for delta_X. Defaults to None.
            clip_delta_X (bool): Whether to clip the delta_X result. Defaults to False.

        Returns:
            cp.ndarray: The delta change in expression after simulation (on GPU). Shape (batch_size, neighbors_size, n_features).
        """
        """WE CURRENTLY DO NOT USE THESE CHECKS AS THE ENVIRONMENT MAKES SURE THIS IS CORRECT AND THIS SAVES COMPUTING TIME"""
        # if GRN_unit != "cluster":
        #     raise ValueError("Currently only cluster GRN is supported for batch simulation")
        # if n_min is None:
        #     n_min = CONFIG["N_PROP_MIN"]  # Assuming CONFIG is accessible
        # if n_max is None:
        #     n_max = CONFIG["N_PROP_MAX"]  # Assuming CONFIG is accessible
        #
        # for i, tup in enumerate(perturb_condition):
        #     condition, value = tup
        #     if condition not in self.all_regulatory_genes_in_TFdict:  # Assuming these dicts are CPU based
        #         raise ValueError(
        #             f"Gene {condition} (in batch {i}) is not included in the base GRN; "
        #             "It is not TF or TF motif information is not available. Cannot perform simulation.")
        #
        #     if condition not in self.active_regulatory_genes:  # Assuming this set is CPU based
        #         raise ValueError(
        #             f"Gene {condition} (in batch {i}) does not have enough regulatory connection "
        #             "in the GRNs. Cannot perform simulation.")

        batch_size, neighbors_size, feature_size = view_data.shape[0], view_data.shape[1], self.adata.n_vars


        final_output = cp.full((batch_size, neighbors_size, feature_size), cp.nan, dtype=cp.float32)

        for cluster_label, matrix_cp in self.coef_matrix_per_cluster_CUPY_dict.items():
            original_indices_for_cluster_cp = self.cluster_label_to_idx_dict_cp[cluster_label]

            batch_mapping_info = []
            total_cells_in_cluster = 0

            for b_idx in range(batch_size):
                idxs_in_batch_cp = batch_idxs[b_idx]

                batch_mask_cp = cp.isin(idxs_in_batch_cp, original_indices_for_cluster_cp, assume_unique=True)

                local_pos_in_batch_for_cluster_cp = cp.where(batch_mask_cp)[0]

                num_cells_in_batch = int(local_pos_in_batch_for_cluster_cp.size)

                if num_cells_in_batch == 0:
                    continue

                batch_mapping_info.append(
                    (b_idx, local_pos_in_batch_for_cluster_cp, num_cells_in_batch, total_cells_in_cluster)
                )
                total_cells_in_cluster += num_cells_in_batch

            if total_cells_in_cluster == 0:
                continue

            full_data_for_cluster_cp = cp.empty((total_cells_in_cluster, feature_size), dtype=cp.float32)
            full_view_data_for_cluster_cp = cp.empty((total_cells_in_cluster, feature_size), dtype=cp.float32)

            for b_idx, local_pos_cp, num_cells, start_idx in batch_mapping_info:
                end_idx = start_idx + num_cells

                input_data_slice_cp = view_data[b_idx, local_pos_cp].copy()
                gem_view_slice_cp = view_data[b_idx, local_pos_cp]

                perturbation, value = perturb_condition[b_idx]
                if perturbation in self.gene_to_index_dict:
                    index_for_gene = self.gene_to_index_dict[perturbation]
                    input_data_slice_cp[:, index_for_gene] = value

                full_data_for_cluster_cp[start_idx:end_idx, :] = input_data_slice_cp
                full_view_data_for_cluster_cp[start_idx:end_idx, :] = gem_view_slice_cp

            simulated_result_cp = _do_simulation_cupy(coef_matrix=matrix_cp,
                                                      simulation_input=full_data_for_cluster_cp,
                                                      n_propagation=n_propagation,
                                                      gem=full_view_data_for_cluster_cp)

            for b_idx, local_pos_cp, num_cells, start_idx in batch_mapping_info:
                end_idx = start_idx + num_cells  # CPU integer math
                result_slice_cp = simulated_result_cp[start_idx:end_idx, :]
                final_output[b_idx, local_pos_cp, :] = result_slice_cp

        final_output_substracted_cp = final_output - view_data

        if clip_delta_X:
            final_output_substracted_cp = cp.clip(final_output_substracted_cp, n_min, n_max)  # CuPy clip


        return final_output_substracted_cp

    def simulate_shift_subset_batch_tensor(self, perturb_condition:dict, GRN_unit:str, batch_idxs:np.ndarray, knockout_prev_used = False,
                              n_propagation=3, ignore_warning=False,use_randomized_GRN=False, n_min=None,n_max=None,clip_delta_X=False ) -> torch.Tensor:
        """
        Simulate signal propagation (that is, a future gene expression shift) on a specified subset of cells.

        Rather than using the entire self.adata, the simulation will be performed only on the primary cell
        (row index: primary_idx) and on its neighbors (row indices: neighbor_idxs). The method applies the standard
        perturbation procedure but only on the supplied subset.

        Arguments:
            perturb_condition: A dictionary where keys are gene names and values are the perturbation values.
            GRN_unit: The unit of the GRN to use for the simulation. Only allow for cluster in batch
            simulation_input: The imputed gene expression matrix for the subset of cell, shape: (batch, primary+neighbors, genes)
            batch_idxs: The indices of the primary cells in the batch, shape: (batch, primary+neighbors)
            n_propagation: The number of propagation steps to simulate.
            ignore_warning: If True, ignore the warning about the number of propagation steps.

        Side Effects:
            Writes the simulated expression values to a new layer "simulated_count_subset" in self.adata.
            The delta (shift) is stored in self.adata.layers["delta_X_subset"].
        """
        if GRN_unit != "cluster":
            raise ValueError("Currently only cluster GRN is supported for batch simulation")
        if n_min is None:
            n_min = CONFIG["N_PROP_MIN"]
        if n_max is None:
            n_max = CONFIG["N_PROP_MAX"]
        self._clear_simulation_results()

        if GRN_unit is not None:
            self.GRN_unit = GRN_unit
        elif hasattr(self, "GRN_unit"):
            GRN_unit = self.GRN_unit
            #print("Currently selected GRN_unit: ", self.GRN_unit)
        elif hasattr(self, "coef_matrix_per_cluster"):
            GRN_unit = "cluster"
            self.GRN_unit = GRN_unit
        elif hasattr(self, "coef_matrix"):
            GRN_unit = "whole"
            self.GRN_unit = GRN_unit
        else:
            raise ValueError("GRN is not ready. Please run 'fit_GRN_for_simulation' first.")

        if use_randomized_GRN:
            print("Attention: Using randomized GRN for the perturbation simulation.")

        # Prepare metadata before simulation
        if not hasattr(self, "active_regulatory_genes"):
            self.extract_active_gene_lists(verbose=False)

        if not hasattr(self, "all_regulatory_genes_in_TFdict"):
            self._process_TFdict_metadata()

        #get first key
        first_key = list(perturb_condition.keys())[0]
        if first_key not in self.all_regulatory_genes_in_TFdict:
            raise ValueError(
                f"Gene {first_key} is not included in the base GRN; It is not TF or TF motif information is not available. Cannot perform simulation.")

        if first_key not in self.active_regulatory_genes:
            raise ValueError(
                f"Gene {first_key} does not have enough regulatory connection in the GRNs. Cannot perform simulation.")

        # 2. Extract the imputed gene expression matrix for these cells.
        #    (Assume self.adata.layers["imputed_count"] is an array-like of shape (ncells, ngenes).)

        batch_size, num_instances_per_batch = batch_idxs.shape
        feature_dim = self.torch_imputed_count.shape[1]
        # Initialize tensor for selected instances
        simulation_input = torch.zeros(batch_size, num_instances_per_batch, feature_dim,dtype=self.torch_imputed_count.dtype,device=self.torch_imputed_count.device)

        # Fill with selected instances for each batch
        max_idx = np.max([np.max(batch_idxs) for batch_idxs in batch_idxs])
        batch_idx_maps = np.full((batch_size, max_idx + 1), -1, dtype=int)
        for batch_idx,data_idxs in enumerate(batch_idxs):
            simulation_input[batch_idx] = self.torch_imputed_count[data_idxs]
            batch_idx_maps[batch_idx, data_idxs] = np.arange(len(data_idxs))
        gem_imputed =simulation_input.clone()


        if batch_size != len(perturb_condition):
            raise ValueError("Batch size and perturb condition size do not match.")

        #this can be done as dictionary keep their original order when looping as its inserted
        for batch_idx, (gene, value) in enumerate(perturb_condition.items()):
            if gene not in self.gene_to_index_dict:
                print(f"Gene {gene} is not in the subset. Skipping perturbation.")
                continue
            index_of_gene = self.gene_to_index_dict[gene]
            simulation_input[batch_idx,: ,index_of_gene] = value# set perturbation on entire subset
            if not knockout_prev_used:
                continue
            already_perturbed_value = self.prev_perturbed_value_batch_dict[batch_idx]
            for gene,value in already_perturbed_value.items():
                if gene not in self.gene_to_index_dict:
                    print(f"Gene {gene} is not in the subset. Skipping perturbation.")
                    continue
                simulation_input[batch_idx, :, self.gene_to_index_dict[gene]] = value

        if knockout_prev_used:
            for batch_idx, (gene, value) in enumerate(perturb_condition.items()):
                #this also fixes the problem when we allow for activation I THINK, as we overwrite a previous knockout and then  activation )or other way around) and it must be final value so this overwrites it accordingly
                self.prev_perturbed_value_batch_dict[batch_idx][gene] = value

        coef_matrix_tensor = {}
        # here we extract the matrix for the cluster corresponding to the primary_idx.
        for batch_idx, subset_idxs in  enumerate(batch_idxs):
            cluster_labels_for_data = np.unique(self.adata.obs[self.cluster_column_name][subset_idxs])
            for cluster_label in cluster_labels_for_data:
                if cluster_label not in self.coef_matrix_per_cluster:
                    continue
                coef_matrix_tensor[cluster_label] = self.coef_matrix_per_cluster_tensor_dict[cluster_label].clone()

        #so we create a mapping of cluster labels to original tensor indices per batch
        #loop through all the possible clusters
        simulated_data = torch.zeros_like(simulation_input)
        for cluster_label, coef_matrix in coef_matrix_tensor.items():
            indices_for_cluster_label = self.cluster_label_to_idx_dict[cluster_label]
            simulated_data_batch = []
            original_data_batch = []
            original_indices_map = []
            batch_counts = []
            for batch_idx, subset_idxs in enumerate(batch_idxs):
                cluster_indices_in_batch  = np.intersect1d(subset_idxs, indices_for_cluster_label, assume_unique=True)
                if len(cluster_indices_in_batch) == 0:
                    continue
                tensor_indices_label = batch_idx_maps[batch_idx][cluster_indices_in_batch]
                simulated_data_label = simulation_input[batch_idx, tensor_indices_label]
                original_data_label = self.torch_imputed_count[cluster_indices_in_batch]
                simulated_data_batch.append(simulated_data_label)
                original_data_batch.append(original_data_label)
                original_indices_map.append((batch_idx, tensor_indices_label))
                batch_counts.append(len(cluster_indices_in_batch))

            #Construct entire data tensors
            start_indices = np.cumsum([0] + batch_counts[:-1])
            end_indices = np.cumsum(batch_counts)

            complete_tensor_simulated_data_for_label = torch.cat(simulated_data_batch, dim=0)
            complete_tensor_original_data_for_label = torch.cat(original_data_batch, dim=0)
            coef_matrix = coef_matrix_tensor[cluster_label]
            #do simulation
            result_of_sim_for_cluster_label = _do_simulation_torch(coef_matrix=coef_matrix,simulation_input=complete_tensor_simulated_data_for_label,gem=complete_tensor_original_data_for_label,n_propagation=n_propagation)
            #now we need to save the result back to the original tensor
            for i,(batch_idx, tensor_indices) in enumerate(original_indices_map):
                if len(tensor_indices) == 0:
                    continue
                start_idx = start_indices[i]
                end_idx = end_indices[i]
                simulated_data[batch_idx, tensor_indices] = result_of_sim_for_cluster_label[start_idx:end_idx]

        if clip_delta_X:
            # check if this works
            return self.clip_delta_X_subset(simulated_data, gem_imputed)
        else:
            return simulated_data - gem_imputed


    def simulate_shift_subset(self, perturb_condition, GRN_unit, subset_idx,
                              n_propagation=3, ignore_warning=False,use_randomized_GRN=False, n_min=None,n_max=None,clip_delta_X=False ) -> pd.DataFrame:
        """
        Simulate signal propagation (that is, a future gene expression shift) on a specified subset of cells.

        Rather than using the entire self.adata, the simulation will be performed only on the primary cell
        (row index: primary_idx) and on its neighbors (row indices: neighbor_idxs). The method applies the standard
        perturbation procedure but only on the supplied subset.

        Arguments:
            perturb_condition (dict): The desired perturbation. For example {"GeneX": 0.0}
            GRN_unit (str): Either "whole" or "cluster"; see fit_GRN_for_simulation for details.
            primary_idx (int): Row index in self.adata corresponding to the cell state of primary interest.
            neighbor_idxs (array-like): Array or list of row indices corresponding to neighbor cells.
            n_propagation (int): Number of iterations for GRN signal propagation (default: 3).
            use_randomized_GRN (bool): Whether to use the randomized GRN For negative control.
            clip_delta_X (bool): Whether to clip any simulated gene expression values outside the wild‐type range.

        Side Effects:
            Writes the simulated expression values to a new layer "simulated_count_subset" in self.adata.
            The delta (shift) is stored in self.adata.layers["delta_X_subset"].
        """
        if n_min is None:
            n_min = CONFIG["N_PROP_MIN"]
        if n_max is None:
            n_max = CONFIG["N_PROP_MAX"]
        self._clear_simulation_results()

        if GRN_unit is not None:
            self.GRN_unit = GRN_unit
        elif hasattr(self, "GRN_unit"):
            GRN_unit = self.GRN_unit
            #print("Currently selected GRN_unit: ", self.GRN_unit)
        elif hasattr(self, "coef_matrix_per_cluster"):
            GRN_unit = "cluster"
            self.GRN_unit = GRN_unit
        elif hasattr(self, "coef_matrix"):
            GRN_unit = "whole"
            self.GRN_unit = GRN_unit
        else:
            raise ValueError("GRN is not ready. Please run 'fit_GRN_for_simulation' first.")

        if use_randomized_GRN:
            print("Attention: Using randomized GRN for the perturbation simulation.")

        # Prepare metadata before simulation
        if not hasattr(self, "active_regulatory_genes"):
            self.extract_active_gene_lists(verbose=False)

        if not hasattr(self, "all_regulatory_genes_in_TFdict"):
            self._process_TFdict_metadata()

        #get first key
        first_key = list(perturb_condition.keys())[0]
        if first_key not in self.all_regulatory_genes_in_TFdict:
            raise ValueError(
                f"Gene {first_key} is not included in the base GRN; It is not TF or TF motif information is not available. Cannot perform simulation.")

        if first_key not in self.active_regulatory_genes:
            raise ValueError(
                f"Gene {first_key} does not have enough regulatory connection in the GRNs. Cannot perform simulation.")
        # 2. Extract the imputed gene expression matrix for these cells.
        #    (Assume self.adata.layers["imputed_count"] is an array-like of shape (ncells, ngenes).)
        subset_adata = self.adata[subset_idx]
        # _adata_to_df already makes copy of the data
        simulation_input_subset = _adata_to_df(subset_adata, 'imputed_count')
        # for gene in self.knocked_out_TFs:
        #     if gene not in simulation_input_subset.columns:
        #         print(f"Gene {gene} is not in the subset. Skipping perturbation.")
        #         continue
        #     simulation_input_subset[gene] = 0.0

        for gene, value in perturb_condition.items():
            if gene not in simulation_input_subset.columns:
                print(f"Gene {gene} is not in the subset. Skipping perturbation.")
                continue
            self.knocked_out_TFs.add(gene)
            simulation_input_subset[gene] = value  # set perturbation on entire subset
        gem_imputed_subset = _adata_to_df(subset_adata, "imputed_count")
        # 4. For each gene perturbed, set every cell in the subset to the specified perturbation value.
        #    (Typically, one perturbs one TF, but this code supports multiple keys.)
        #    We assume that the columns in simulation_input are labelled by gene names in self.adata.var.index.
        #    If simulation_input is a DataFrame use .loc; if it is a numpy array then assume a separate lookup.
        #    For demonstration, here we assume simulation_input is a DataFrame.

        # 5. Retrieve the GRN coefficient matrices for the subset.
        coef_matrix_list = {}
        if GRN_unit == "whole":
            if use_randomized_GRN:
                if not hasattr(self, "coef_matrix_randomized"):
                    self.calculate_randomized_coef_table()
                coef_matrix = self.coef_matrix_randomized.copy()
            else:
                coef_matrix = self.coef_matrix.copy()
            coef_matrix_list["whole"] = coef_matrix
        elif GRN_unit == "cluster":
            # For cluster-specific GRNs, assume self.coef_matrix_per_cluster is available;
            # here we extract the matrix for the cluster corresponding to the primary_idx.
            cluster_labels = self.adata.obs[self.cluster_column_name]
            cluster_labels = cluster_labels[subset_idx]
            cluster_labels = np.unique(cluster_labels)
            for cluster_label in cluster_labels:
                if use_randomized_GRN:
                    if not hasattr(self, "coef_matrix_per_cluster_randomized"):
                        self.calculate_randomized_coef_table()
                    coef_matrix = self.coef_matrix_per_cluster_randomized[cluster_label].copy()
                else:
                    coef_matrix = self.coef_matrix_per_cluster[cluster_label].copy()
                coef_matrix_list[cluster_label] = coef_matrix
        else:
            raise ValueError("GRN_unit should be either 'whole' or 'cluster'.")

        # 6. Call the simulation routine on the subset.
        #    _do_simulation is expected to accept:
        #         coef_matrix, simulation_input (DataFrame), and the original gem_imputed_subset (DataFrame) along with the propagation count.
        simulated_data = []
        for cluster_label, cluster_GRN in coef_matrix_list.items():
            cells_in_cluster_bool = subset_adata.obs[self.cluster_column_name] == cluster_label
            simulation_input_ = simulation_input_subset[cells_in_cluster_bool]
            gem_imputed_ = gem_imputed_subset[cells_in_cluster_bool]
            gem_simulated_subset = _do_simulation(coef_matrix=cluster_GRN,
                                                  simulation_input=simulation_input_,
                                                  gem=gem_imputed_,
                                                  n_propagation=n_propagation)
            #find idexes for false
            # idx = np.where(closearray == False)
            # row_indices, col_indices = idx[0], idx[1]
            # for r, c in zip(row_indices, col_indices):
            #     print(f"Position [{r}][{c}]:")
            #     print(f"  sim_input: {simulation_input_.to_numpy()[r][c]}")
            #     print(
            #         f"  other_input: {self.adata.layers['imputed_count'][subset_idx][cells_in_cluster_bool].copy()[r][c]}")
            #     print(
            #         f"  difference: {simulation_input_.to_numpy()[r][c] - self.adata.layers['imputed_count'][subset_idx][cells_in_cluster_bool].copy()[r][c]}")
            simulated_data.append(gem_simulated_subset)
        result_sim = pd.concat(simulated_data, axis=0)
        result_sim = result_sim.reindex(subset_adata.obs.index)
        # 7. Calculate the difference between the simulated and imputed gene expression values.
        delta_result_sim = result_sim - subset_adata.layers["imputed_count"]
        return delta_result_sim
        #print total zero count
        # 8. Optionally clip out-of-distribution predictions.
        if clip_delta_X:
            # check if this works
            self.clip_delta_X_subset(result_sim, gem_imputed_subset)

        return delta_result_sim

    ####END SIMULATE SHIFT, END OF CUSTOM CODE IN CURRENT CLASS####


    ####################################
    ### 2. Methods for GRN inference ###
    ####################################
    def fit_GRN_for_simulation(self, GRN_unit="cluster", alpha=1, use_cluster_specific_TFdict=False, verbose_level=1):
        """
        Do GRN inference.
        Please see the paper of CellOracle paper for details.

        GRN can be constructed for the entire population or each clusters.
        If you want to infer cluster-specific GRN, please set [GRN_unit="cluster"].
        You can select cluster information when you import data.

        If you set [GRN_unit="whole"], GRN will be made using all cells.

        Args:
            GRN_unit (str): Select "cluster" or "whole"

            alpha (float or int): The strength of regularization.
                If you set a lower value, the sensitivity increases, and you can detect weaker network connections. However, there may be more noise.
                If you select a higher value, it will reduce the chance of overfitting.

            verbose_level (int): if [verbose_level>1], most detailed progress information will be shown.
                if [1 >= verbose_level > 0], one progress bar will be shown.
                if [verbose_level == 0], no progress bar will be shown.

        """

        if verbose_level > 1:
            verbose_cluster = True
            verbose_gene = True
        elif 0 < verbose_level <= 1:
            verbose_cluster = True
            verbose_gene = False
        else:
            verbose_cluster = False
            verbose_gene = False

        # prepare data for GRN calculation
        gem_imputed = _adata_to_df(self.adata, "imputed_count")
        self.adata.layers["simulation_input"] = self.adata.layers["imputed_count"].copy()
        self.alpha_for_trajectory_GRN = alpha
        self.GRN_unit = GRN_unit

        if use_cluster_specific_TFdict & (self.cluster_specific_TFdict is not None):
            self.coef_matrix_per_cluster = {}
            cluster_info = self.adata.obs[self.cluster_column_name]
            with tqdm(np.unique(cluster_info), disable=(verbose_cluster==False)) as pbar:
                for cluster in pbar:
                    pbar.set_postfix(cluster=f"{cluster}")
                    cells_in_the_cluster_bool = (cluster_info == cluster)
                    gem_ = gem_imputed[cells_in_the_cluster_bool]
                    self.coef_matrix_per_cluster[cluster] = _getCoefMatrix(gem=gem_,
                                                                           TFdict=self.cluster_specific_TFdict[cluster],
                                                                           alpha=alpha,
                                                                           verbose=verbose_gene)

        else:
            if GRN_unit == "whole":
                self.coef_matrix = _getCoefMatrix(gem=gem_imputed, TFdict=self.TFdict, alpha=alpha, verbose=verbose_gene)
            if GRN_unit == "cluster":
                self.coef_matrix_per_cluster = {}
                cluster_info = self.adata.obs[self.cluster_column_name]
                with tqdm(np.unique(cluster_info), disable=(verbose_cluster==False)) as pbar:
                    for cluster in pbar:
                        pbar.set_postfix(cluster=f"{cluster}")
                        cells_in_the_cluster_bool = (cluster_info == cluster)
                        gem_ = gem_imputed[cells_in_the_cluster_bool]
                        self.coef_matrix_per_cluster[cluster] = _getCoefMatrix(gem=gem_,
                                                                               TFdict=self.TFdict,
                                                                               alpha=alpha,
                                                                               verbose=verbose_gene)

        self.extract_active_gene_lists(verbose=False)


    def extract_active_gene_lists(self, return_as=None, verbose=False):
        """
        Args:
            return_as (str): If not None, it returns dictionary or list. Chose either "indivisual_dict" or "unified_list".
            verbose (bool): Whether to show progress bar.

        Returns:
            dictionary or list: The format depends on the argument, "return_as".

        """
        if return_as not in ["indivisual_dict", "unified_list", None]:
            raise ValueError("return_as should be either 'indivisual_dict' or 'unified_list'.")

        if not hasattr(self, "GRN_unit"):
            try:
                loop = self.coef_matrix_per_cluster.items()
                self.GRN_unit = "cluster"
                print("Currently selected GRN_unit: ", self.GRN_unit)

            except:
                try:
                    loop = {"whole_cell": self.coef_matrix}.items()
                    self.GRN_unit = "whole"
                    print("Currently selected GRN_unit: ", self.GRN_unit)
                except:
                    raise ValueError("GRN is not ready. Please run 'fit_GRN_for_simulation' first.")

        elif self.GRN_unit == "cluster":
            loop = self.coef_matrix_per_cluster.items()
        elif self.GRN_unit == "whole":
            loop = {"whole_cell": self.coef_matrix}.items()

        if verbose:
            loop = tqdm(loop)

        unified_list = []
        indivisual_dict = {}
        for cluster, coef_matrix in loop:
            active_genes = _coef_to_active_gene_list(coef_matrix)
            unified_list += active_genes
            indivisual_dict[cluster] = active_genes

        unified_list = list(np.unique(unified_list))

        # Store data
        self.active_regulatory_genes = unified_list.copy()
        self.adata.var["symbol"] = self.adata.var.index.values
        if "isin_top1000_var_mean_genes" not in self.adata.var.columns:
            self.adata.var["isin_top1000_var_mean_genes"] = self.adata.var.symbol.isin(self.high_var_genes)
        self.adata.var["isin_actve_regulators"] = self.adata.var.symbol.isin(unified_list)

        if return_as == "indivisual_dict":
            return indivisual_dict

        elif return_as == "unified_list":
            return unified_list




    #######################################################
    ### 3. Methods for simulation of signal propagation ###
    #######################################################

    def simulate_shift(self, perturb_condition=None, GRN_unit=None,
                       n_propagation=3, ignore_warning=False, use_randomized_GRN=False, clip_delta_X=False):
        """
        Simulate signal propagation with GRNs. Please see the CellOracle paper for details.
        This function simulates a gene expression pattern in the near future.
        Simulated values will be stored in anndata.layers: ["simulated_count"]


        The simulation use three types of data.
        (1) GRN inference results (coef_matrix).
        (2) Perturb_condition: You can set arbitrary perturbation condition.
        (3) Gene expression matrix: The simulation starts from imputed gene expression data.

        Args:
            perturb_condition (dictionary): condition for perturbation.
               if you want to simulate knockout for GeneX, please set [perturb_condition={"GeneX": 0.0}]
               Although you can set any non-negative values for the gene condition, avoid setting biologically infeasible values for the perturb condition.
               It is strongly recommended to check gene expression values in your data before selecting the perturb condition.

            GRN_unit (str): GRN type. Please select either "whole" or "cluster". See the documentation of "fit_GRN_for_simulation" for the detailed explanation.

            n_propagation (int): Calculation will be performed iteratively to simulate signal propagation in GRN.
                You can set the number of steps for this calculation.
                With a higher number, the results may recapitulate signal propagation for many genes.
                However, a higher number of propagation may cause more error/noise.

            clip_delta_X (bool): If simulated gene expression shift can lead to gene expression value that is outside of WT distribution, such gene expression is clipped to WT range.
        """
        self.__simulate_shift(perturb_condition=perturb_condition,
                              GRN_unit=GRN_unit,
                              n_propagation=n_propagation,
                              ignore_warning=ignore_warning,
                              use_randomized_GRN=use_randomized_GRN,
                              clip_delta_X=clip_delta_X)

    def __simulate_shift(self, perturb_condition=None, GRN_unit=None,
                         n_propagation=3, ignore_warning=False, use_randomized_GRN=False, n_min=None, n_max=None,
                         clip_delta_X=False):
        """
        Simulate signal propagation with GRNs. Please see the CellOracle paper for details.
        This function simulates a gene expression pattern in the near future.
        Simulated values will be stored in anndata.layers: ["simulated_count"]


        The simulation use three types of data.
        (1) GRN inference results (coef_matrix).
        (2) Perturb_condition: You can set arbitrary perturbation condition.
        (3) Gene expression matrix: The simulation starts from imputed gene expression data.

        Args:
            perturb_condition (dictionary): condition for perturbation.
               if you want to simulate knockout for GeneX, please set [perturb_condition={"GeneX": 0.0}]
               Although you can set any non-negative values for the gene condition, avoid setting biologically infeasible values for the perturb condition.
               It is strongly recommended to check gene expression values in your data before selecting the perturb condition.

            GRN_unit (str): GRN type. Please select either "whole" or "cluster". See the documentation of "fit_GRN_for_simulation" for the detailed explanation.

            n_propagation (int): Calculation will be performed iteratively to simulate signal propagation in GRN.
                You can set the number of steps for this calculation.
                With a higher number, the results may recapitulate signal propagation for many genes.
                However, a higher number of propagation may cause more error/noise.
        """

        # 0. Reset previous simulation results if it exist
        # self.ixs_markvov_simulation = None
        # self.markvov_transition_id = None
        # self.corrcoef = None
        # self.transition_prob = None
        # self.tr = None
        if n_min is None:
            n_min = CONFIG["N_PROP_MIN"]
        if n_max is None:
            n_max = CONFIG["N_PROP_MAX"]
        self._clear_simulation_results()

        if GRN_unit is not None:
            self.GRN_unit = GRN_unit
        elif hasattr(self, "GRN_unit"):
            GRN_unit = self.GRN_unit
            # print("Currently selected GRN_unit: ", self.GRN_unit)
        elif hasattr(self, "coef_matrix_per_cluster"):
            GRN_unit = "cluster"
            self.GRN_unit = GRN_unit
        elif hasattr(self, "coef_matrix"):
            GRN_unit = "whole"
            self.GRN_unit = GRN_unit
        else:
            raise ValueError("GRN is not ready. Please run 'fit_GRN_for_simulation' first.")

        if use_randomized_GRN:
            print("Attention: Using randomized GRN for the perturbation simulation.")

        # 1. prepare perturb information

        self.perturb_condition = perturb_condition.copy()

        # Prepare metadata before simulation
        if not hasattr(self, "active_regulatory_genes"):
            self.extract_active_gene_lists(verbose=False)

        if not hasattr(self, "all_regulatory_genes_in_TFdict"):
            self._process_TFdict_metadata()

        for i, value in perturb_condition.items():
            # 1st Sanity check
            if not i in self.adata.var.index:
                raise ValueError(f"Gene {i} is not included in the Gene expression matrix.")

            # 2nd Sanity check
            if i not in self.all_regulatory_genes_in_TFdict:
                raise ValueError(
                    f"Gene {i} is not included in the base GRN; It is not TF or TF motif information is not available. Cannot perform simulation.")

            # 3rd Sanity check
            if i not in self.active_regulatory_genes:
                raise ValueError(
                    f"Gene {i} does not have enough regulatory connection in the GRNs. Cannot perform simulation.")

            # 4th Sanity check
            if i not in self.high_var_genes:
                if ignore_warning:
                    pass
                    # print(f"Variability score of Gene {i} is too low. Simulation accuracy may be poor with this gene.")
                else:
                    pass
                    # print(f"Variability score of Gene {i} is too low. Simulation accuracy may be poor with this gene.")
                    # raise ValueError(f"Variability score of Gene {i} is too low. Cannot perform simulation.")

            # 5th Sanity check
            if value < 0:
                raise ValueError(f"Negative gene expression value is not allowed.")

            # 6th Sanity check
            safe = _is_perturb_condition_valid(adata=self.adata,
                                               goi=i, value=value, safe_range_fold=2)
            if not safe:
                if ignore_warning:
                    pass
                else:
                    raise ValueError(
                        f"Input perturbation condition is far from actural gene expression value. Please follow the recommended usage. ")
            # 7th QC
            if n_min <= n_propagation <= n_max:
                pass
            else:
                raise ValueError(f'n_propagation value error. It should be an integer from {n_min} to {n_max}.')

        # reset simulation initiation point
        self.adata.layers["simulation_input"] = self.adata.layers["imputed_count"].copy()
        simulation_input = _adata_to_df(self.adata, "simulation_input")
        for i in perturb_condition.keys():
            simulation_input[i] = perturb_condition[i]

        # 2. load gene expression matrix (initiation information for the simulation)
        gem_imputed = _adata_to_df(self.adata, "imputed_count")
        print("shape of gem imputed: ", gem_imputed.shape)
        # 3. do simulation for signal propagation within GRNs
        if GRN_unit == "whole":
            if use_randomized_GRN == False:
                coef_matrix = self.coef_matrix.copy()
            else:
                if hasattr(self, "coef_matrix_randomized") == False:
                    print("The random coef matrix was calculated.")
                    self.calculate_randomized_coef_table()
                coef_matrix = self.coef_matrix_randomized.copy()
            gem_simulated = _do_simulation(coef_matrix=coef_matrix,
                                           simulation_input=simulation_input,
                                           gem=gem_imputed,
                                           n_propagation=n_propagation)

        elif GRN_unit == "cluster":
            simulated = []
            cluster_info = self.adata.obs[self.cluster_column_name]
            for cluster in np.unique(cluster_info):

                if use_randomized_GRN == False:
                    coef_matrix = self.coef_matrix_per_cluster[cluster].copy()
                else:
                    if hasattr(self, "coef_matrix_per_cluster_randomized") == False:
                        print("The random coef matrix was calculated.")
                        self.calculate_randomized_coef_table()
                    coef_matrix = self.coef_matrix_per_cluster_randomized[cluster].copy()
                cells_in_the_cluster_bool = (cluster_info == cluster)
                simulation_input_ = simulation_input[cells_in_the_cluster_bool]
                gem_ = gem_imputed[cells_in_the_cluster_bool]

                simulated_in_the_cluster = _do_simulation(
                    coef_matrix=coef_matrix,
                    simulation_input=simulation_input_,
                    gem=gem_,
                    n_propagation=n_propagation)

                simulated.append(simulated_in_the_cluster)
            gem_simulated = pd.concat(simulated, axis=0)
            gem_simulated = gem_simulated.reindex(gem_imputed.index)

        else:
            raise ValueError("GRN_unit shold be either of 'whole' or 'cluster'")

        # 4. store simulation results
        #  simulated future gene expression matrix
        self.adata.layers["simulated_count"] = gem_simulated.values

        #  difference between simulated values and original values
        self.adata.layers["delta_X"] = self.adata.layers["simulated_count"] - self.adata.layers["imputed_count"]
        #save certain entry to be used for future simulations


        # Clip simulated gene expression to avoid out of distribution prediction.
        if clip_delta_X:
            self.clip_delta_X()

        # Sanity check; check distribution of simulated values. If the value is far from original gene expression range, it will give warning.
        if ignore_warning:
            pass
        else:
            ood_stat = self.evaluate_simulated_gene_distribution_range()
            ood_stat = ood_stat[ood_stat.Max_exceeding_ratio > CONFIG["OOD_WARNING_EXCEEDING_PERCENTAGE"] / 100]
            if len(ood_stat) > 0:
                message = f"There may be out of distribution prediction in {len(ood_stat)} genes. It is recommended to set `clip_delta_X=True` to avoid the out of distribution prediction."
                message += "\n To see the detail, please run `oracle.evaluate_simulated_gene_distribution_range()`"
                warnings.warn(message, UserWarning, stacklevel=2)


    def _clear_simulation_results(self):
        att_list = ["flow_embedding", "flow_grid", "flow", "flow_norm_magnitude",
                    "flow_rndm", "flow_norm_rndm", "flow_norm_magnitude_rndm",
                    "corrcoef","corrcoef_random", "transition_prob", "transition_prob_random",
                    "delta_embedding", "delta_embedding_random",
                    "ixs_markvov_simulation", "markvov_transition_id", "tr"]

        for i in att_list:
            if hasattr(self, i):
                setattr(self, i, None)

    def evaluate_simulated_gene_distribution_range(self):
        """
        CellOracle does not intend to simulate out-of-distribution simulation.
        This function evaluates how the simulated gene expression values differ from the undisturbed gene expression distribution range.
        """

        exceedance = self._calculate_potential_OOD_exceedance_ratio()
        statistics = pd.concat([exceedance.max(),
                                (exceedance != 0).mean(axis=0)], axis=1)
        statistics.columns = ["Max_exceeding_ratio", "OOD_cell_ratio"]

        statistics = statistics.sort_values(by="Max_exceeding_ratio", ascending=False)

        return statistics

    def _calculate_potential_OOD_exceedance_ratio(self):

        """
        CellOracle does not intend to simulate out-of-distribution simulation.
        This function evaluates how the simulated gene expression values differ from the undisturbed gene expression distribution range.

        Args:

            pandas.DataFrame: The value is exceeding ratio.
        """
        # Sanity check
        if "simulated_count" in self.adata.layers.keys():
            pass
        else:
            raise ValueError("Simulation results not found. Run simulation first.")

        simulated_count = self.adata.to_df(layer="simulated_count")
        imputed_count = self.adata.to_df(layer="imputed_count")


        relative_ratio = _calculate_relative_ratio_of_simulated_value(simulated_count=simulated_count,
                                                                      reference_count=imputed_count)

        lower_exceedance = np.clip(relative_ratio, -np.inf, 0).abs()
        higer_exceedance = np.clip(relative_ratio-1, 0, np.inf)
        exceedance = pd.DataFrame(np.maximum(lower_exceedance.values, higer_exceedance.values),
                                index=relative_ratio.index,
                                columns=relative_ratio.columns)
        return exceedance

    def evaluate_and_plot_simulation_value_distribution(self, n_genes=4, n_bins=10, alpha=0.5, figsize=[5, 3], save=None):

        """
        This function will visualize distribution of original gene expression value and simulated values.
        This cunction is built to confirm there is no significant out-of-distribution in the simulation results.

        Args:
            n_genes (int): Number of genes to show. This functin will show the results of top n_genes with large difference between original and simulation values.
            n_bins (int): Number of bins.
            alpha (float): Transparency.
            figsize ([float, float]): Figure size.
            save (str): Folder path to save your plots. If it is not specified, no figure is saved.
        Return:
            None
        """

        simulated_count = self.adata.to_df(layer="simulated_count")
        imputed_count = self.adata.to_df(layer="imputed_count")

        ood_stats = self.evaluate_simulated_gene_distribution_range()

        if save is not None:
            os.makedirs(save, exist_ok=True)

        for goi, val in ood_stats[:n_genes].iterrows():
            fig, ax = plt.subplots(figsize=figsize)
            in_range_cell_ratio = 1 - val["OOD_cell_ratio"]
            ax.hist(imputed_count[goi], label="Original value", alpha=alpha, bins=n_bins)
            ax.hist(simulated_count[goi], label="Simulation value", alpha=alpha,  bins=n_bins)
            message = f"Gene: ${goi}$, "
            message += f"Cells in original gene range: {in_range_cell_ratio*100:.5g}%, "
            message += f"\nMax exceedance: {val['Max_exceeding_ratio']*100:.3g}%"
            plt.title(message)
            plt.legend()
            plt.xlabel("Gene expression")
            plt.ylabel("Count")
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.15)
            if save is not None:
                fig.savefig(os.path.join(save, f"gene_expression_histogram_Spi1_KO_{goi}.png"), transparent=True)
            plt.show()


    def clip_delta_X(self):
        """
        To avoid issue caused by out-of-distribution prediction, this function clip simulated gene expression value to the unperturbed gene expression range.
        """
        # Sanity check
        if "simulated_count" in self.adata.layers.keys():
            pass
        else:
            raise ValueError("Simulation results not found. Run simulation first.")

        simulated_count = self.adata.to_df(layer="simulated_count").copy()
        imputed_count = self.adata.to_df(layer="imputed_count").copy()

        min_ = imputed_count.min(axis=0)
        max_ = imputed_count.max(axis=0)

        for goi in simulated_count.columns:
            simulated_count[goi] = np.clip(simulated_count[goi], min_[goi], max_[goi])

        self.adata.layers["simulated_count"] = simulated_count.values
        self.adata.layers["delta_X"] = self.adata.layers["simulated_count"] - self.adata.layers["imputed_count"]


    def clip_delta_X_subset(self, simulated_count, imputed_count):
        """
        To avoid issue caused by out-of-distribution prediction, this function clip simulated gene expression value to the unperturbed gene expression range.
        """
        min_ = imputed_count.min(axis=0)
        max_ = imputed_count.max(axis=0)

        for goi in simulated_count.columns:
            simulated_count[goi] = np.clip(simulated_count[goi], min_[goi], max_[goi])

        return simulated_count


    def estimate_impact_of_perturbations_under_various_ns(self, perturb_condition, order=1, n_prop_max=5, GRN_unit=None, figsize=[7, 3]):
        """
        This function is designed to help user to estimate appropriate n for signal propagation.
        The function will do the following calculation for each n and plot results as line plot.
        1. Calculate signal propagation.
        2. Calculate the vector length of delta_X, which represents the simulated shift vector for each cell in the multi dimensional gene expression space.
        3. Calculate mean of delta_X for each cluster.
        Repeat step 1~3 for each n and plot results as a line plot.

        Args:
            perturb_condition (dictionary): Please refer to the function 'simulate_shift' for detail of this.
            order (int): If order=1, this function calculate l1 norm. If order=2, it calculate l2 norm.
            n_prop_max (int): Max of n to try.
        Return
            matplotlib figure
        """
        lengths = []
        for i in tqdm(range(0, n_prop_max+1)):
            self.__simulate_shift(perturb_condition=perturb_condition,
                                  GRN_unit=None,
                                  n_propagation=i,
                                  ignore_warning=False,
                                  use_randomized_GRN=False,
                                  n_min=0, n_max=n_prop_max+1)

            delta = self.adata.to_df(layer="delta_X")
            length = np.linalg.norm(delta, ord=order, axis=1)
            lengths.append(length)

        lengths = pd.DataFrame(lengths).transpose()
        lengths.columns = [f"{i}" for i in range(0, n_prop_max+1)]
        lengths["group"] = self.adata.obs[self.cluster_column_name].values

        # Plot results
        fig, ax = plt.subplots(figsize=figsize)
        lengths.groupby("group").mean().transpose().plot(ax=ax)
        plt.xlabel("n_propagation")
        plt.ylabel(f"Mean delta_X length")
        plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0, fontsize=13)
        plt.subplots_adjust(right=0.5, bottom=0.15)

        return fig


    def calculate_p_mass(self, smooth=0.8, n_grid=40, n_neighbors=200, n_jobs=-1):

        self.calculate_grid_arrows(smooth=0.8, steps=(n_grid, n_grid), n_neighbors=n_neighbors, n_jobs=-1)


    def suggest_mass_thresholds(self, n_suggestion=12, s=1, n_col=4):

        min_ = self.total_p_mass.min()
        max_ = self.total_p_mass.max()
        suggestions = np.linspace(min_, max_/2, n_suggestion)

        n_rows = math.ceil(n_suggestion / n_col)

        fig, ax = plt.subplots(n_rows, n_col, figsize=[5*n_col, 5*n_rows])
        if n_rows == 1:
            ax = ax.reshape(1, -1)

        row = 0
        col = 0
        for i in range(n_suggestion):

            ax_ = ax[row, col]

            col += 1
            if col == n_col:
                col = 0
                row += 1

            idx = self.total_p_mass > suggestions[i]

                #ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
            ax_.scatter(self.embedding[:, 0], self.embedding[:, 1], c="lightgray", s=s)
            ax_.scatter(self.flow_grid[idx, 0],
                       self.flow_grid[idx, 1],
                       c="black", s=s)
            ax_.set_title(f"min_mass: {suggestions[i]: .2g}")
            ax_.axis("off")


    def calculate_mass_filter(self, min_mass=0.01, plot=False):

        self.min_mass = min_mass
        self.mass_filter = (self.total_p_mass < min_mass)

        if plot:
            fig, ax = plt.subplots(figsize=[5,5])

            #ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
            ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c="lightgray", s=10)
            ax.scatter(self.flow_grid[~self.mass_filter, 0],
                       self.flow_grid[~self.mass_filter, 1],
                       c="black", s=0.5)
            ax.set_title("Grid points selected")
            ax.axis("off")

    ## Get randomized GRN coef to do randomized perturbation simulation
    def calculate_randomized_coef_table(self, random_seed=123):
        "Calculate randomized GRN coef table."

        if hasattr(self, "coef_matrix_per_cluster"):
            coef_matrix_per_cluster_randomized = {}
            for key, val in self.coef_matrix_per_cluster.items():
                coef_matrix_per_cluster_randomized[key] = _shuffle_celloracle_GRN_coef_table(coef_dataframe=val, random_seed=random_seed)
            self.coef_matrix_per_cluster_randomized = coef_matrix_per_cluster_randomized

        if hasattr(self, "coef_matrix"):
            self.coef_matrix_randomized = _shuffle_celloracle_GRN_coef_table(coef_dataframe=self.coef_matrix, random_seed=random_seed)

        if (hasattr(self, "coef_matrix_per_cluster") == False) and (hasattr(self, "coef_matrix") == False):
            print("GRN calculation for simulation is not finished. Run fit_GRN_for_simulation() first.")

    ########################################
    ### 4. Methods for Markov simulation ###
    ########################################
    def prepare_markov_simulation(self, verbose=False):
        """
        Pick up cells for Markov simulation.

        Args:
            verbose (bool): If True, it plots selected cells.

        """
        # Sample uniformly the points to avoid density driven effects - Should reimplement as a method
        steps = 100, 100
        grs = []
        for dim_i in range(self.embedding.shape[1]):
            m, M = np.min(self.embedding[:, dim_i]), np.max(self.embedding[:, dim_i])
            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps[dim_i])
            grs.append(gr)

        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

        nn = NearestNeighbors()
        nn.fit(self.embedding)
        dist, ixs = nn.kneighbors(gridpoints_coordinates, 1)

        diag_step_dist = np.sqrt((meshes_tuple[0][0,0] - meshes_tuple[0][0,1])**2 + (meshes_tuple[1][0,0] - meshes_tuple[1][1,0])**2)
        min_dist = diag_step_dist / 2
        ixs = ixs[dist < min_dist]
        gridpoints_coordinates = gridpoints_coordinates[dist.flat[:]<min_dist,:]
        dist = dist[dist < min_dist]

        ixs = np.unique(ixs)
        self.ixs_markvov_simulation = ixs

        if verbose:
            plt.scatter(self.embedding[ixs, 0], self.embedding[ixs, 1],
                        c=self.colorandum[ixs], alpha=1, s=30, lw=0.4,
                        edgecolor="0.4")

        self.prepare_markov(sigma_D=diag_step_dist, sigma_W=diag_step_dist/2.,
                       direction='forward', cells_ixs=ixs)


    def run_markov_chain_simulation(self, n_steps=500, n_duplication=5, seed=123, calculate_randomized=True):
        """
        Do Markov simlations to predict cell transition after perturbation.
        The transition probability between cells has been calculated
        based on simulated gene expression values in the signal propagation process.
        The cell state transition will be simulated based on the probability.
        You can simulate the process multiple times to get a robust outcome.

        Args:
            n_steps (int): steps for Markov simulation. This value is equivalent to the amount of time after perturbation.

            n_duplication (int): the number for multiple calculations.

        """
        warnings.warn(
            "Functions for Markov simulation are deprecated. They may be retired in the future version. Use Perturbation score analysis instead.",
            DeprecationWarning,
            stacklevel=2)

        np.random.seed(seed)
        #_numba_random_seed(seed)

        self.prepare_markov_simulation()

        transition_prob = self.tr.toarray()

        #
        transition_prob = _deal_with_na(transition_prob) # added 20200607

        n_cells = transition_prob.shape[0]

        start_cell_id_array = np.repeat(np.arange(n_cells), n_duplication)

        transition = _walk(start_cell_id_array, transition_prob, n_steps)
        transition = self.ixs_markvov_simulation[transition]

        li = None

        ind = np.repeat(self.ixs_markvov_simulation, n_duplication)
        self.markvov_transition_id = pd.DataFrame(transition, ind)

        if calculate_randomized:
            transition_prob_random = self.tr_random.toarray()
            #
            transition_prob_random = _deal_with_na(transition_prob_random) # added 20200607

            n_cells = transition_prob_random.shape[0]

            start_cell_id_array = np.repeat(np.arange(n_cells), n_duplication)

            transition_random = _walk(start_cell_id_array, transition_prob_random, n_steps)
            transition_random = self.ixs_markvov_simulation[transition_random]

            li = None

            ind = np.repeat(self.ixs_markvov_simulation, n_duplication)
            self.markvov_transition_random_id = pd.DataFrame(transition_random, ind)


    def summarize_mc_results_by_cluster(self, cluster_use, random=False):
        """
        This function summarizes the simulated cell state-transition by groping the results into each cluster.
        It returns sumarized results as a pandas.DataFrame.

        Args:
            cluster_use (str): cluster information name in anndata.obs.
               You can use any arbitrary cluster information in anndata.obs.
        """
        if random:
            transition = self.markvov_transition_random_id.values
        else:
            transition = self.markvov_transition_id.values

        markvov_transition_cluster = np.array(self.adata.obs[cluster_use])[transition]
        markvov_transition_cluster = pd.DataFrame(markvov_transition_cluster,
                                               index=self.markvov_transition_id.index)
        return markvov_transition_cluster


    def plot_mc_results_as_sankey(self, cluster_use, start=0, end=-1, order=None, font_size=10):
        """
        Plot the simulated cell state-transition as a Sankey-diagram after groping by the cluster.

        Args:
            cluster_use (str): cluster information name in anndata.obs.
               You can use any cluster information in anndata.obs.

            start (int): The starting point of Sankey-diagram. Please select a  step in the Markov simulation.

            end (int): The end point of Sankey-diagram. Please select a  step in the Markov simulation.
                if you set [end=-1], the final step of Markov simulation will be used.

            order (list of str): The order of cluster name in the Sankey-diagram.

            font_size (int): Font size for cluster name label in the Sankey diagram.

        """
        warnings.warn(
            "Functions for Markov simulation are deprecated. They may be retired in the future version. Use Perturbation score analysis instead.",
            DeprecationWarning,
            stacklevel=2)

        markvov_transition_cluster = self.summarize_mc_results_by_cluster(cluster_use)
        markvov_simulation_color_dict =  _adata_to_color_dict(self.adata, cluster_use)

        df = markvov_transition_cluster.iloc[:, [start, end]]
        df.columns = ["start", "end"]

        if not order is None:
            order_ = order.copy()
            order_.reverse()
            order_left = [i for i in order_ if i in df.start.unique()]
            order_right = [i for i in order_ if i in df.end.unique()]
        else:
            order_left = list(df.start.unique())
            order_right = list(df.end.unique())

        sankey(left=df['start'], right=df['end'],
               aspect=2, fontsize=font_size,
               colorDict=markvov_simulation_color_dict,
               leftLabels=order_left, rightLabels=order_right)

    def plot_mc_results_as_kde(self, n_time, args={}):
        """
        Pick up one timepoint in the cell state-transition simulation and plot as a kde plot.

        Args:
            n_time (int): the number in Markov simulation

            args (dictionary): An argument for seaborn.kdeplot.
                See seaborn documentation for details (https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot).

        """
        warnings.warn(
            "Functions for Markov simulation are deprecated. They may be retired in the future version. Use Perturbation score analysis instead.",
            DeprecationWarning,
            stacklevel=2)

        cell_ix = self.markvov_transition_id.iloc[:, n_time].values

        x = self.embedding[cell_ix, 0]
        y = self.embedding[cell_ix, 1]

        try:
            sns.kdeplot(x=x, y=y, **args)
        except:
            sns.kdeplot(x, y, **args)

    def plot_mc_results_as_trajectory(self, cell_name, time_range, args={}):
        """
        Pick up several timepoints in the cell state-transition simulation and plot as a line plot.
        This function can be used to visualize how cell-state changes after perturbation focusing on a specific cell.

        Args:
            cell_name (str): cell name. chose from adata.obs.index

            time_range (list of int): the list of index in Markov simulation

            args (dictionary): dictionary for the arguments for matplotlib.pyplit.plot.
                See matplotlib documentation for details (https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot).

        """
        warnings.warn(
            "Functions for Markov simulation are deprecated. They may be retired in the future version. Use Perturbation score analysis instead.",
            DeprecationWarning,
            stacklevel=2)

        cell_ix = np.where(self.adata.obs.index == cell_name)[0][0]
        cell_ix_in_markvov_simulation_tid = np.where(self.markvov_transition_id.index == cell_ix)[0]

        # plot all cells in gray color
        plt.scatter(self.embedding[:,0], self.embedding[:,1], s=1, c="lightgray")


        for i in cell_ix_in_markvov_simulation_tid:
            self._plot_one_trajectory(i, time_range, args)

        # plot cell of interest (initiation point of simulation) in red color
        plt.scatter(self.embedding[cell_ix,0], self.embedding[cell_ix,1], s=50, c="red")

    def _plot_one_trajectory(self, cell_ix_in_markvov_simulation_tid, time_range, args={}):
        tt = self.markvov_transition_id.iloc[cell_ix_in_markvov_simulation_tid,:].values[time_range]
        plt.plot(self.embedding[:,0][tt], self.embedding[:,1][tt], **args)


    def count_cells_in_mc_resutls(self, cluster_use, end=-1, order=None):
        """
        Count the simulated cell by the cluster.

        Args:
            cluster_use (str): cluster information name in anndata.obs.
               You can use any cluster information in anndata.obs.

            end (int): The end point of Sankey-diagram. Please select a  step in the Markov simulation.
                if you set [end=-1], the final step of Markov simulation will be used.
        Returns:
            pandas.DataFrame : Number of cells before / after simulation

        """
        warnings.warn(
            "Functions for Markov simulation are deprecated. They may be retired in the future version. Use Perturbation score analysis instead.",
            DeprecationWarning,
            stacklevel=2)

        markvov_transition_cluster = self.summarize_mc_results_by_cluster(cluster_use, random=False)

        if hasattr(self, "markvov_transition_random_id"):
            markvov_transition_cluster_random = self.summarize_mc_results_by_cluster(cluster_use, random=True)

            df = pd.DataFrame({"original": markvov_transition_cluster.iloc[:, 0],
                               "simulated": markvov_transition_cluster.iloc[:, end],
                               "randomized": markvov_transition_cluster_random.iloc[:, end]})
        else:
            df = pd.DataFrame({"original": markvov_transition_cluster.iloc[:, 0],
                               "simulated": markvov_transition_cluster.iloc[:, end]})

        # Post processing
        n_duplicated = df.index.value_counts().values[0]
        df["simulation_batch"] = [i%n_duplicated for i in np.arange(len(df))]

        df = df.melt(id_vars="simulation_batch")
        df["count"] = 1
        df = df.groupby(["value", "variable", "simulation_batch"]).count()
        df = df.reset_index(drop=False)

        df = df.rename(columns={"value": "cluster", "variable": "data"})
        df["simulation_batch"] = df["simulation_batch"].astype(object)

        return df

    def get_markov_simulation_cell_transition_table(self, cluster_column_name=None, end=-1, return_df=True):

        """
        Calculate cell count in the initial state and final state after Markov simulation.
        Cell counts are grouped by the cluster of interest.
        Result will be stored as 2D matrix.
        """

        if cluster_column_name is None:
            cluster_column_name = self.cluster_column_name

        start = 0

        markvov_transition = self.summarize_mc_results_by_cluster(cluster_column_name, random=False)
        markvov_transition = markvov_transition.iloc[:, [start, end]]
        markvov_transition.columns = ["start", "end"]
        markvov_transition["count"] = 1
        markvov_transition = pd.pivot_table(markvov_transition, values='count', index=['start'],
                               columns=['end'], aggfunc=np.sum, fill_value=0)


        markvov_transition_random = self.summarize_mc_results_by_cluster(cluster_column_name, random=True)
        markvov_transition_random = markvov_transition_random.iloc[:, [start, end]]
        markvov_transition_random.columns = ["start", "end"]
        markvov_transition_random["count"] = 1
        markvov_transition_random = pd.pivot_table(markvov_transition_random, values='count', index=['start'],
                               columns=['end'], aggfunc=np.sum, fill_value=0)

        # store data
        markvov_transition_random.index.name = None
        markvov_transition_random.columns.name = None
        markvov_transition.index.name = None
        markvov_transition.columns.name = None

        self.markvov_transition = markvov_transition
        self.markvov_transition_random = markvov_transition_random

        if return_df:
            return self.markov_transition, self.markov_transition_random

    def get_markvov_simulation_cell_transition_table(self, cluster_column_name=None, end=-1, return_df=True):
        self.get_markov_simulation_cell_transition_table(cluster_column_name=cluster_column_name, end=end, return_df=False)




    ###################################################
    ### 5. GRN inference for Network score analysis ###
    ###################################################
    def get_links(self, cluster_name_for_GRN_unit=None, alpha=10, bagging_number=20, verbose_level=1, test_mode=False, model_method="bagging_ridge", ignore_warning=False, n_jobs=-1):
        """
        Makes GRN for each cluster and returns results as a Links object.
        Several preprocessing should be done before using this function.

        Args:
            cluster_name_for_GRN_unit (str): Cluster name for GRN calculation. The cluster information should be stored in Oracle.adata.obs.

            alpha (float or int): The strength of regularization.
                If you set a lower value, the sensitivity increases, and you can detect weaker network connections. However, there may be more noise.
                If you select a higher value, it will reduce the chance of overfitting.

            bagging_number (int): The number used in bagging calculation.


            verbose_level (int): if [verbose_level>1], most detailed progress information will be shown.
                if [verbose_level > 0], one progress bar will be shown.
                if [verbose_level == 0], no progress bar will be shown.

            test_mode (bool): If test_mode is True, GRN calculation will be done for only one cluster rather than all clusters.

            model_method (str): Chose modeling algorithm. "bagging_ridge" or "bayesian_ridge"

            n_jobs (int): Number of cpu cores for parallel calculation.  -1 means using all available cores. Default is -1.


        """

        ## Check data
        info = self._generate_meta_data()

        if ignore_warning:
            pass
        else:
            if info["status - Gene expression matrix"] != "Ready":
                raise ValueError("scRNA-seq data is not imported.")

            if info["status - PCA calculation"] != "Done":
                raise ValueError("Preprocessing is not done. Do PCA and Knn imputation.")

            if info["status - Knn imputation"] != "Done":
                raise ValueError("Preprocessing is not done. Do Knn imputation.")

            if info["status - BaseGRN"] != "Ready":
                raise ValueError("Found No TF information. Please import TF data (base-GRN) first.")

            if info["n_regulatory_in_both_TFdict_and_scRNA-seq"] == '0 genes':
                raise ValueError("Found No overlap between TF info (base GRN) and your scRNA-seq data. Please check your data format and species.")

            if info["n_target_genes_both_TFdict_and_scRNA-seq"] == '0 genes':
                raise ValueError("Found No overlap between TF info (base GRN) and your scRNA-seq data. Please check your data format and species.")


        links = get_links(oracle_object=self,
                          cluster_name_for_GRN_unit=cluster_name_for_GRN_unit,
                          alpha=alpha, bagging_number=bagging_number,
                          verbose_level=verbose_level, test_mode=test_mode,
                          model_method=model_method,
                          n_jobs=n_jobs)
        return links

    def get_adata(self):
        return self.adata




def _deal_with_na(transition_prob):
    tr = transition_prob.copy()

    # remove nan
    tr = np.nan_to_num(tr, copy=True, nan=0)

    # if transition prob is 0 in all row, assign transitionprob = 1 to self row.
    no_transition_ids = (tr.sum(axis=1) == 0)
    tr[no_transition_ids, no_transition_ids] = 1

    return tr

#     def simulate_shift_subset_batch_numpy_test(self, perturb_condition: list, GRN_unit: str, batch_idxs: np.ndarray, view_data:np.ndarray,
#                                           knockout_prev_used=False,
#                                           n_propagation=3, ignore_warning=False, use_randomized_GRN=False, n_min=None,
#                                           n_max=None, clip_delta_X=False, compare_results = None) -> np.ndarray:
#         """
#         Simulate signal propagation (that is, a future gene expression shift) on a specified subset of cells.
#         NumPy implementation.
#
#         Arguments:
#             perturb_condition: A dictionary where keys are gene names and values are the perturbation values.
#             GRN_unit: The unit of the GRN to use for the simulation. Only allow for cluster in batch
#             batch_idxs: The indices of the primary cells in the batch, shape: (batch, primary+neighbors)
#             knockout_prev_used: Whether to use previous knockout values.
#             n_propagation: The number of propagation steps to simulate.
#             ignore_warning: If True, ignore the warning about the number of propagation steps.
#             use_randomized_GRN: Whether to use randomized GRN.
#             n_min: Minimum number of propagation steps.
#             n_max: Maximum number of propagation steps.
#             clip_delta_X: Whether to clip delta_X values.
#
#         Returns:
#             Delta X values (differences between simulated and original data).
#         """
#         if GRN_unit != "cluster":
#             raise ValueError("Currently only cluster GRN is supported for batch simulation")
#         if n_min is None:
#             n_min = CONFIG["N_PROP_MIN"]
#         if n_max is None:
#             n_max = CONFIG["N_PROP_MAX"]
#         self._clear_simulation_results()
#
#         if GRN_unit is not None:
#             self.GRN_unit = GRN_unit
#         elif hasattr(self, "GRN_unit"):
#             GRN_unit = self.GRN_unit
#         elif hasattr(self, "coef_matrix_per_cluster"):
#             GRN_unit = "cluster"
#             self.GRN_unit = GRN_unit
#         elif hasattr(self, "coef_matrix"):
#             GRN_unit = "whole"
#             self.GRN_unit = GRN_unit
#         else:
#             raise ValueError("GRN is not ready. Please run 'fit_GRN_for_simulation' first.")
#
#         if use_randomized_GRN:
#             print("Attention: Using randomized GRN for the perturbation simulation.")
#
#         # Prepare metadata before simulation
#         if not hasattr(self, "active_regulatory_genes"):
#             self.extract_active_gene_lists(verbose=False)
#
#         if not hasattr(self, "all_regulatory_genes_in_TFdict"):
#             self._process_TFdict_metadata()
#
#         # Get first key
#         # first_key = list(perturb_condition.keys())[0]
#         # if first_key not in self.all_regulatory_genes_in_TFdict:
#         #     raise ValueError(
#         #         f"Gene {first_key} is not included in the base GRN; It is not TF or TF motif information is not available. Cannot perform simulation.")
#         #
#         # if first_key not in self.active_regulatory_genes:
#         #     raise ValueError(
#         #         f"Gene {first_key} does not have enough regulatory connection in the GRNs. Cannot perform simulation.")
#
#         global_idx_to_batch_pos = {}
#         for b_idx, idxs_in_batch in enumerate(batch_idxs):
#             for pos_in_batch, global_idx in enumerate(idxs_in_batch):
#                 global_idx_to_batch_pos[global_idx] = (b_idx, pos_in_batch)
#
#         batch_size, neighbors_size, feature_size = batch_idxs.shape[0], batch_idxs.shape[1], self.adata.n_vars
#         # Declare output
#         final_output = np.empty((batch_size, neighbors_size, feature_size), dtype=np.float32)
#         for cluster_label, matrix in self.coef_matrix_per_cluster_np_dict.items():
#             input_data_for_cluster = []
#             view_data_for_cluster = []
#             global_indices_in_simulation_order = []
#             original_indices_for_cluster = self.cluster_label_to_idx_dict[cluster_label]
#
#             for idx,idxs in enumerate(batch_idxs):
#                 #get intersection of batch indexes and all the indexes for the labels
#                 batch_indices_for_cluster = np.intersect1d(original_indices_for_cluster, idxs)
#
#                 #skip if necessary
#                 if len(batch_indices_for_cluster) == 0:
#                     continue
#
#                 #get the data for the cluster
#                 input_data_for_sim = self.imputed_count[batch_indices_for_cluster].copy()
#                 #perform perturb
#                 perturbation,value = perturb_condition[idx]
#
#                 index_for_gene = self.gene_to_index_dict[perturbation]
#                 input_data_for_sim[:,index_for_gene] = value
#                 #pertub done
#
#                 input_data_for_cluster.append(input_data_for_sim)
#                 view_data_for_cluster.append(self.imputed_count[batch_indices_for_cluster])
#                 global_indices_in_simulation_order.append(batch_indices_for_cluster)
#
#             #create full datasets for this cluster to perform operation
#             if len(input_data_for_cluster) == 0:
#                 continue
#             full_data_for_cluster = np.concatenate(input_data_for_cluster, axis=0)
#             full_view_data_for_cluster = np.concatenate(view_data_for_cluster, axis=0)
#             ordered_global_indices = np.concatenate(global_indices_in_simulation_order, axis=0)
#             simulated_result = _do_simulation_numpy(coef_matrix=matrix, simulation_input=full_data_for_cluster, n_propagation=n_propagation, gem=full_view_data_for_cluster)
#             print(f"cluster_label: {cluster_label},  does it contain nans: ", np.isnan(simulated_result).any())
#
#             #resort the data back to the correct batches and order with the data being in the correct batch and order as it was
#             for i, global_idx in enumerate(ordered_global_indices):
#                 if global_idx in global_idx_to_batch_pos:
#                     b_idx, pos_in_batch = global_idx_to_batch_pos[global_idx]
#                     # Ensure the target position is within the bounds for that batch
#                     if pos_in_batch < final_output.shape[1]:
#                         final_output[b_idx, pos_in_batch, :] = simulated_result[i, :]
#             print(f"does it contain nans now after cluster label: {cluster_label},  does it contain nans: ", np.isnan(final_output).any())
#
#         final_output_substracted = final_output - view_data
#         return final_output_substracted
#
#
#
#         # simulated_data_batch = []
#         #
#         # for batch_idx, subset_idxs in enumerate(batch_idxs):
#         #     simulated_data = []
#         #     original_positions = []
#         #     cluster_labels_unique = np.unique(self.adata.obs[self.cluster_column_name][subset_idxs])
#         #     for cluster_label in cluster_labels_unique:
#         #         cells_in_batch_bool = self.adata.obs[self.cluster_column_name][subset_idxs] == cluster_label
#         #         global_indices = subset_idxs[cells_in_batch_bool]
#         #         original_positions.extend(global_indices)
#         #         coef_matrix = self.coef_matrix_per_cluster_np_dict[cluster_label]
#         #         simulation_input_single = self.adata.layers["imputed_count"][subset_idxs][cells_in_batch_bool].copy()
#         #         condition_to_perturb,value_to_perturb = perturb_condition[batch_idx]
#         #         index_for_gene = self.gene_to_index_dict[condition_to_perturb]
#         #         simulation_input_single[:,index_for_gene] = value_to_perturb
#         #         simulated_result= _do_simulation_numpy(coef_matrix=coef_matrix, simulation_input=simulation_input_single,  n_propagation=n_propagation,gem=self.adata.layers["imputed_count"][subset_idxs][cells_in_batch_bool])
#         #         # Print number of differences found
#         #         simulated_data.append(simulated_result)
#         #
#         #
#         #     clustered_ordered_data = np.concatenate(simulated_data, axis=0)
#         #     # Get the indices needed to match pandas' reindexing order
#         #     # (self.adata[subset_idxs].obs.index contains the desired order)
#         #     position_to_index = {pos: idx for idx, pos in enumerate(original_positions)}
#         #
#         #     # Create reorder indices based on the row order of subset_idxs
#         #     # This will match how pandas.concat followed by reindex behaves
#         #     reorder_indices = np.array([position_to_index[idx] for idx in subset_idxs])
#         #
#         #     # Apply the reordering in one vectorized operation
#         #     reordered_data = clustered_ordered_data[reorder_indices]
#         #
#         #     # Now use these integer indices
#         #     #check if any a
#         #     simulated_data_batch.append(reordered_data)
#         # result_sim = np.concatenate(simulated_data_batch, axis=0)
#         #
#         # #TODO THIS MUST BE CHANGED
#         # final_result = result_sim - self.adata.layers["imputed_count"][batch_idxs[0]]
#         # #if shape 2 then add a new dimension
#         # if len(final_result.shape) == 2:
#         #     final_result = final_result[np.newaxis, :]
#         # print("final result shape: ", final_result.shape)
#         # print("final output shape: ", final_output_substracted.shape)
#         # print("are both similiar: ", np.allclose(final_result, final_output_substracted, atol=1e-6))
#         # return final_result
#         # # print(result_sim)
#         #
#         # if clip_delta_X:
#         #     # Assuming clip_delta_X_subset can handle NumPy arrays
#         #     return self.clip_delta_X_subset(simulated_data, gem_imputed)
#         # else:
#         #     return result_sim
#
#     ####SIMULATE SHIFT FUNCTIONS####
#     def simulate_shift_subset_batch_numpy(self, perturb_condition: dict, GRN_unit: str, batch_idxs: np.ndarray,
#                                           knockout_prev_used=False,
#                                           n_propagation=3, ignore_warning=False, use_randomized_GRN=False, n_min=None,
#                                           n_max=None, clip_delta_X=False, compare_results = None) -> np.ndarray:
#         """
#         Simulate signal propagation (that is, a future gene expression shift) on a specified subset of cells.
#         NumPy implementation.
#
#         Arguments:
#             perturb_condition: A dictionary where keys are gene names and values are the perturbation values.
#             GRN_unit: The unit of the GRN to use for the simulation. Only allow for cluster in batch
#             batch_idxs: The indices of the primary cells in the batch, shape: (batch, primary+neighbors)
#             knockout_prev_used: Whether to use previous knockout values.
#             n_propagation: The number of propagation steps to simulate.
#             ignore_warning: If True, ignore the warning about the number of propagation steps.
#             use_randomized_GRN: Whether to use randomized GRN.
#             n_min: Minimum number of propagation steps.
#             n_max: Maximum number of propagation steps.
#             clip_delta_X: Whether to clip delta_X values.
#
#         Returns:
#             Delta X values (differences between simulated and original data).
#         """
#         if GRN_unit != "cluster":
#             raise ValueError("Currently only cluster GRN is supported for batch simulation")
#         if n_min is None:
#             n_min = CONFIG["N_PROP_MIN"]
#         if n_max is None:
#             n_max = CONFIG["N_PROP_MAX"]
#         self._clear_simulation_results()
#
#         if GRN_unit is not None:
#             self.GRN_unit = GRN_unit
#         elif hasattr(self, "GRN_unit"):
#             GRN_unit = self.GRN_unit
#         elif hasattr(self, "coef_matrix_per_cluster"):
#             GRN_unit = "cluster"
#             self.GRN_unit = GRN_unit
#         elif hasattr(self, "coef_matrix"):
#             GRN_unit = "whole"
#             self.GRN_unit = GRN_unit
#         else:
#             raise ValueError("GRN is not ready. Please run 'fit_GRN_for_simulation' first.")
#
#         if use_randomized_GRN:
#             print("Attention: Using randomized GRN for the perturbation simulation.")
#
#         # Prepare metadata before simulation
#         if not hasattr(self, "active_regulatory_genes"):
#             self.extract_active_gene_lists(verbose=False)
#
#         if not hasattr(self, "all_regulatory_genes_in_TFdict"):
#             self._process_TFdict_metadata()
#
#         # Get first key
#         first_key = list(perturb_condition.keys())[0]
#         if first_key not in self.all_regulatory_genes_in_TFdict:
#             raise ValueError(
#                 f"Gene {first_key} is not included in the base GRN; It is not TF or TF motif information is not available. Cannot perform simulation.")
#
#         if first_key not in self.active_regulatory_genes:
#             raise ValueError(
#                 f"Gene {first_key} does not have enough regulatory connection in the GRNs. Cannot perform simulation.")
#
#
#
#         # Get dimensions
#         batch_size, num_instances_per_batch = batch_idxs.shape
#         feature_dim = self.adata.n_vars
#
#         # Fill with selected instances for each batch
#         max_idx = np.max([np.max(batch_idxs) for batch_idxs in batch_idxs])
#         batch_idx_maps = np.full((batch_size, max_idx + 1), -1, dtype=int)
#         simulation_input = np.zeros((batch_size, num_instances_per_batch, feature_dim))
#         cluster_labels_present_in_all_batch = set()
#         for batch_idx, data_idxs in enumerate(batch_idxs):
#             simulation_input[batch_idx] = self.adata.X[data_idxs].toarray() if sp.issparse(self.adata.X) else self.adata.X[data_idxs]
#             cluster_labels_present_in_all_batch.update(self.adata.obs[self.cluster_column_name][data_idxs])
#             batch_idx_maps[batch_idx, data_idxs] = np.arange(len(data_idxs))
#
#         gem_imputed = simulation_input.copy()
#
#         if batch_size != len(perturb_condition):
#             raise ValueError("Batch size and perturb condition size do not match.")
#
#         # Apply perturbations
#         for batch_idx, (gene, value) in enumerate(perturb_condition.items()):
#             if gene not in self.gene_to_index_dict:
#                 print(f"Gene {gene} is not in the subset. Skipping perturbation.")
#                 continue
#             index_of_gene = self.gene_to_index_dict[gene]
#             simulation_input[batch_idx, :, index_of_gene] = value  # set perturbation on entire subset
#             if not knockout_prev_used:
#                 continue
#             already_perturbed_value = self.prev_perturbed_value_batch_dict[batch_idx]
#             for gene, value in already_perturbed_value.items():
#                 if gene not in self.gene_to_index_dict:
#                     print(f"Gene {gene} is not in the subset. Skipping perturbation.")
#                     continue
#                 simulation_input[batch_idx, :, self.gene_to_index_dict[gene]] = value
#
#         if knockout_prev_used:
#             for batch_idx, (gene, value) in enumerate(perturb_condition.items()):
#                 # Update perturbation history
#                 self.prev_perturbed_value_batch_dict[batch_idx][gene] = value
#
#
#
#         # Process by cluster
#         simulated_data = np.zeros_like(simulation_input)
#         for cluster_label in cluster_labels_present_in_all_batch:
#             indices_for_cluster_label = self.cluster_label_to_idx_dict[cluster_label]
#             coef_matrix = self.coef_matrix_per_cluster_np_dict[cluster_label]
#
#             simulated_data_batch = []
#             original_data_batch = []
#             original_indices_map = []
#             batch_counts = []
#
#             for batch_idx, subset_idxs in enumerate(batch_idxs):
#                 cluster_indices_in_batch = np.intersect1d(subset_idxs, indices_for_cluster_label, assume_unique=True)
#                 if len(cluster_indices_in_batch) == 0:
#                     continue
#
#                 tensor_indices_label = batch_idx_maps[batch_idx][cluster_indices_in_batch]
#                 simulated_data_label = simulation_input[batch_idx, tensor_indices_label]
#                 original_data_label = gem_imputed[batch_idx, tensor_indices_label]
#
#                 simulated_data_batch.append(simulated_data_label)
#                 original_data_batch.append(original_data_label)
#                 original_indices_map.append((batch_idx, tensor_indices_label))
#                 batch_counts.append(len(cluster_indices_in_batch))
#
#             if not simulated_data_batch:
#                 continue
#
#             # Construct entire data arrays
#             start_indices = np.cumsum([0] + batch_counts[:-1])
#             end_indices = np.cumsum(batch_counts)
#
#             complete_simulated_data_for_label = np.concatenate(simulated_data_batch, axis=0)
#             complete_original_data_for_label = np.concatenate(original_data_batch, axis=0)
#
#             # Do simulation - using NumPy version, NumPy version of do_simulation ALREADY RETURNS THE DELTA FOR OPTIMIZATION PURPOSES
#             result_of_sim_for_cluster_label = _do_simulation_numpy(
#                 coef_matrix=coef_matrix,
#                 simulation_input=complete_simulated_data_for_label,
#                 gem=complete_original_data_for_label,
#                 n_propagation=n_propagation
#             )
#
#             # Save the result back to the original array
#             for i, (batch_idx, tensor_indices) in enumerate(original_indices_map):
#                 if len(tensor_indices) == 0:
#                     continue
#                 start_idx = start_indices[i]
#                 end_idx = end_indices[i]
#                 simulated_data[batch_idx, tensor_indices] = result_of_sim_for_cluster_label[start_idx:end_idx] - complete_original_data_for_label[start_idx:end_idx]
#
#             #compare results
#         if compare_results is not None:
#             # Convert pandas DataFrame to numpy array if needed
#             compare_array = compare_results.to_numpy() if hasattr(compare_results, 'to_numpy') else np.array(
#                 compare_results)
#
#             # Flatten both arrays for set comparison
#             simulated_flat = simulated_data.flatten()
#             compare_flat = compare_array.flatten()
#
#             # Check if both arrays contain the same values (ignoring position)
#             simulated_set = set(np.round(simulated_flat, decimals=6))
#             compare_set = set(np.round(compare_flat, decimals=6))
#
#             if simulated_set == compare_set:
#                 print("Both arrays contain the same values (ignoring position)")
#             else:
#                 # Show differences in values
#                 only_in_simulated = simulated_set - compare_set
#                 only_in_compare = compare_set - simulated_set
#
#                 print(f"Values only in simulated: {len(only_in_simulated)} unique values")
#                 print(f"Values only in reference: {len(only_in_compare)} unique values")
#
#                 # Show sample differences
#                 if len(only_in_simulated) > 0:
#                     print("Sample values only in simulated:", list(only_in_simulated)[:5])
#                 if len(only_in_compare) > 0:
#                     print("Sample values only in reference:", list(only_in_compare)[:5])
#         if clip_delta_X:
#             # Assuming clip_delta_X_subset can handle NumPy arrays
#             return self.clip_delta_X_subset(simulated_data, gem_imputed)
#         else:
#             return simulated_data