import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import setuptools
import site
import logging

class ReadyData:

    def __init__(self, raw_count_data: sc.AnnData, normalized_data: sc.AnnData, output_dir: str):
        self.raw_count_data = raw_count_data
        self.normalized_data = normalized_data
        self.output_dir = output_dir


    def get_raw_count_data(self):
        return self.raw_count_data

    def get_normalized_data(self):
        return self.normalized_data

    def get_output_dir(self):
        return self.output_dir

    def add_layers(self):
        logging.info("shape of both sets, raw:norm" + str(self.raw_count_data.shape) + ":" + str(self.normalized_data.shape))
        logging.info(self.raw_count_data)
        logging.info(self.normalized_data)
        logging.info("adding raw count data to normalized data")
        self.normalized_data.layers['raw_counts_unspliced'] = self.raw_count_data.layers['unspliced'].copy()
        self.normalized_data.layers['raw_counts_chic'] = self.raw_count_data.layers['chic_counts'].copy()
        logging.info(self.raw_count_data)
        logging.info(self.normalized_data)
        # logging.info("raw obs columns: "+self.raw_count_data.obs.columns)
        # logging.info("raw obms keys"+self.raw_count_data.obms.keys)
        # logging.info("norm obs columns: "+self.normalized_data.obs.columns)
        # logging.info("norm obms keys"+self.normalized_data.obms.keys)

    def add_umap_data(self, umap_data:pd.DataFrame, output_dir_umap:str):
        logging.info("adding umap data to normalized data")
        logging.info(umap_data)
        umap_data.index = umap_data['Cell_ID'].str.replace("::", ":")  # slight modification to cell IDs
        self.normalized_data.obs = self.normalized_data.obs.join(umap_data[['celltype', 'celltype_general', 'lineage', 'X_umap-0', 'X_umap-1']])  # copy information
        self.normalized_data.obsm['X_umap'] = np.array(self.normalized_data.obs[['X_umap-0', 'X_umap-1']])  # move umap to `obsm
        logging.info("added umap data to normalized")
        logging.info(self.normalized_data)
        umap_data.celltype_general.unique()  # this gives major cell types
        umap_data.celltype.unique()  # this gives sub-types of cells (can be considered "cell states" for this analysis)
        self.normalized_data.write(output_dir_umap + "/normalized_data_with_umap.h5ad")
        ## plot to double-check
