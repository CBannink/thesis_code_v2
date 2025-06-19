import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import logging
import natsort



class Converter:
    """Convert the promoter to TF list to base GRN"""

    def __init__(self, initial_promoter_gene_list:pd.DataFrame, pearson_values:sc.AnnData, output_dir:str):

        self.logger = logging.getLogger(__name__)
        initial_promoter_gene_list = initial_promoter_gene_list.dropna(subset=['TF_inSwissRegulon'])
        self.promoter_to_tf_list = initial_promoter_gene_list
        self.promoter_to_tf_list_copy = initial_promoter_gene_list.copy()

        # self.pearson_values = pearson_values
        # self.pearson_values_copy = pearson_values.copy()
        self.output_dir = output_dir
        self.initial_promoter_gene_parquet = None

    def create_base_grn(self) -> pd.DataFrame:
        """Create the base GRN and output in a dataframe format, file is located as a .parquet at output dir"""
        # there are 2 distinct colums: peak_id and gene_short_name, the other columns are transcription factors
        # we will create a .parquet file with peak_id and gene_short_name and all the tf's as columns, the rows constist of the genes
        # the values are binary values, 1 if the gene is a target of the tf, 0 otherwise
        #subset_genes_set = self._create_dict_of_subset_genes()
        self._create_gene_column_from_promoter_column()
        self._subset_promoter_dataset_with_gene_list()
        self._create_peak_id_column_in_promoter_dataset()
        #currently, the promoter dataset has now a gene_short_name and peak_id column
        tf_set = self._create_set_of_tfs_from_promoter_dataset()
        peak_id_to_tf_dict = self._create_peak_id_to_tf_dict()
        #create the base grn with all 0 for the tfs
        base_grn_unfilled = self._create_base_grn_with_emtpy_tfs(tf_set=tf_set)
        #fill the base grn with 1 if the tf is present for the gene
        base_grn_filled = self._fill_tf_base_grn(base_grn_unfilled, peak_id_to_tf_dict)
        self._save_base_grn(base_grn_filled)

    def _create_dict_of_subset_genes(self) -> set:
        """create the dict of the subset of the genes"""
        self.logger.info('Creating dict of genes with size: ' + str(self.promoter_to_tf_list_copy.shape))
        top_gene_dict = set()
        for gene in self.pearson_values_copy.var["Symbol"]:
            top_gene_dict.add(gene)
        self.logger.info('Created dict of genes with size: ' + str(len(top_gene_dict)))
        return top_gene_dict

    def _create_gene_column_from_promoter_column(self) ->pd.DataFrame:
        """Create a gene column from the promoter column, e.g: P_1:gene_name:count -> gene_name"""
        self.logger.info('Creating gene column from promoter column')
        print(self.promoter_to_tf_list_copy["name"])
        self.promoter_to_tf_list_copy['gene_short_name'] = self.promoter_to_tf_list_copy['name'].str.split(':').str[1]
        self.logger.info('Created gene column from promoter column')
        return self.promoter_to_tf_list_copy

    def _subset_promoter_dataset_with_gene_list(self) ->pd.DataFrame:
        """Subset the initial promoter gene list to only include the genes in the gene list"""
        self.logger.info('subsetting the initial promoter gene list')
        #check if name column exists
        if 'gene_short_name' not in self.promoter_to_tf_list_copy.columns:
            self.logger.info('The name column does not exist in the initial promoter gene list')
            self._create_gene_column_from_promoter_column()

        #subset the initial gene list from set with the column 'name'in the self.itial list
        self.logger.info('Subsetting the initial promoter gene list with original length: ' + str(self.promoter_to_tf_list_copy.shape))
        #self.promoter_to_tf_list_copy = self.promoter_to_tf_list_copy[self.promoter_to_tf_list_copy['gene_short_name'].isin(gene_list)]
        self.logger.info('Subsetting the initial promoter gene list with new length: ' + str(self.promoter_to_tf_list_copy.shape))
        return self.promoter_to_tf_list_copy

    def _create_peak_id_column_in_promoter_dataset(self) -> pd.DataFrame:
        """Create a peak_id column in the promoter dataset"""
        self.logger.info('Creating peak id column in promoter dataset')
        self.promoter_to_tf_list_copy['peak_id'] = 'chr'+ self.promoter_to_tf_list_copy['seqnames'].astype(str) + '_' + self.promoter_to_tf_list_copy['start'].astype(str) + '_' + self.promoter_to_tf_list_copy['end'].astype(str)
        self.logger.info('Created peak id column in promoter dataset')
        return self.promoter_to_tf_list_copy

    def _create_set_of_tfs_from_promoter_dataset(self) -> set:
        """Create a set of TFs from the promoter column"""
        self.logger.info('Creating a set of TFs from the promoter column')
        tf_set = set()
        i=0
        for tf in self.promoter_to_tf_list_copy['TF_inSwissRegulon']:
            capitalized_tfs = [tf_name.capitalize() for tf_name in tf.split(',')]
            tf_set.update(capitalized_tfs)
        self.logger.info('Created a set of TFs from the promoter column with size: ' + str(len(tf_set)))
        return tf_set

    def _create_peak_id_to_tf_dict(self) -> dict:
        """create a dictionary with peakid as keys and the values are lists of tfs to indicate the connection between tf and peak id"""
        self.logger.info('Creating a dictionary with peak id as keys and the values are lists of TFs')
        peak_id_to_tf_dict = {}
        for idx, row in self.promoter_to_tf_list_copy.iterrows():
            # Capitalize each TF name when creating the dictionary
            peak_id_to_tf_dict[row['peak_id']] = [tf.capitalize() for tf in row['TF_inSwissRegulon'].split(',')]
        self.logger.info('Created a dictionary with peak id as keys and the values are lists of TFs')
        return peak_id_to_tf_dict

    def _create_base_grn_with_emtpy_tfs(self, tf_set: set) -> pd.DataFrame:
        base_data = {
            'peak_id': self.promoter_to_tf_list_copy['peak_id'],
            'gene_short_name': self.promoter_to_tf_list_copy['gene_short_name']
        }
        # Add TF columns all at once
        tf_data = {tf: 0 for tf in tf_set}
        return pd.DataFrame({**base_data, **tf_data})

    def _fill_tf_base_grn(self, base_grn_unfilled:pd.DataFrame, peak_id_to_tfs:dict) -> pd.DataFrame:
        """Fill the base grn with a 1 value if tf is present for gene"""
        self.logger.info('Filling the base grn')
        for idx, row in base_grn_unfilled.iterrows():
            for tf in peak_id_to_tfs[row['peak_id']]:
                base_grn_unfilled.at[idx, tf] = 1
        base_grn_unfilled.sort_values(by='peak_id', inplace=True, key=lambda x: x.map(natsort.natsort_key))
        self.logger.info('Filled the base grn with shape: '+ str(base_grn_unfilled.shape))
        self.logger.info("Sample of base grn: \n" + str(base_grn_unfilled.head()))
        return base_grn_unfilled

    def _save_base_grn(self, base_grn:pd.DataFrame):
        """Save the base grn to a .parquet file"""
        self.logger.info('Saving the base grn to a parquet file')
        base_grn.to_parquet(self.output_dir)
        self.logger.info('Saved the base grn to a parquet file')

    def reset_data(self):
        """Reset the data to the original data"""
        #self.pearson_values_copy = self.pearson_values.copy()
        self.initial_promoter_gene_list_copy = self.initial_promoter_gene_list.copy()


    