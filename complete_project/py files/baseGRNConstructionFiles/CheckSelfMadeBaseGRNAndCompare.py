import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import logging

class CompareExampleBaseGRNToCustomMadeGRN:

    @staticmethod
    def compare_grn(ex_base_grn: pd.DataFrame, custom_grn: pd.DataFrame):
        logging.info("COMPARING THE 2 BASE GRNS START:")
        logging.info("Shape of the example base GRN: %s", ex_base_grn.shape)
        logging.info("Shape of the custom GRN: %s", custom_grn.shape)

        # subset the tf's from ex
        ex_base_tfs = [col for col in ex_base_grn.columns if col not in ['peak_id', 'gene_short_name']]
        ex_tfs = ex_base_grn[ex_base_tfs]

        # subset the tf's from custom
        custom_tfs = [col for col in custom_grn.columns if col not in ['peak_id', 'gene_short_name']]
        custom_tfs = custom_grn[custom_tfs]

        # compare shapes
        logging.info("Comparing the TFs")
        logging.info("Shape of the example base TFs: %s", ex_tfs.shape)
        logging.info("Shape of the custom TFs: %s", custom_tfs.shape)

        # compare the amount of 1s
        logging.info("Comparing the amount of 1s in the two GRNs")
        ex_avg_ones = ex_tfs.sum(axis=1).mean()  # Sum 1's in each row, then take mean
        custom_avg_ones = custom_tfs.sum(axis=1).mean()
        logging.info("Average number of 1's per row in example base GRN: %s", ex_avg_ones)
        logging.info("Average number of 1's per row in custom GRN: %s", custom_avg_ones)

        #compare the tfs
        ex_tf_set = set(ex_base_tfs)
        custom_tf_set = set(custom_tfs.columns)
        common_tfs = ex_tf_set.intersection(custom_tf_set)
        unique_to_ex = ex_tf_set.difference(custom_tf_set)
        unique_to_custom = custom_tf_set.difference(ex_tf_set)
        logging.info("Common TFs: %s", common_tfs)
        logging.info("TFs unique to example base GRN: %s", unique_to_ex)
        logging.info("TFs unique to custom GRN: %s", unique_to_custom)

        # Correlation of TFs
        correlation = ex_tfs.corrwith(custom_tfs, axis=0)
        logging.info("Correlation of TFs between the two GRNs: %s", correlation)

        # Distribution of 1s
        ex_ones_distribution = ex_tfs.sum(axis=1)
        custom_ones_distribution = custom_tfs.sum(axis=1)
        logging.info("Distribution of 1s in example base GRN: %s", ex_ones_distribution.describe())
        logging.info("Distribution of 1s in custom GRN: %s", custom_ones_distribution.describe())

        ex_genes = set(ex_base_grn['gene_short_name'])
        custom_genes = set(custom_grn['gene_short_name'])
        matching_genes = len(ex_genes.intersection(custom_genes))
        logging.info("Number of matching gene names: %s out of %s in ex_base and %s in custom",matching_genes, len(ex_genes), len(custom_genes))

        # Count matching TFs (position-independent)
        matching_tfs = len(common_tfs)  # Already calculated in your code
        logging.info("Number of matching TFs: %s out of %s in ex_base and %s in custom",matching_tfs, len(ex_tf_set), len(custom_tf_set))
        logging.info("COMPARING THE 2 BASE GRNS END")




        
        