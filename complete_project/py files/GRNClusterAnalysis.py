import celloracle as co
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import setuptools
import site
import seaborn as sns
import logging


class Analysis:

    def __init__(self, cellOc_object:co.Oracle, link_object:co.Links, plot_dir:str):
        plt.rcParams["figure.figsize"] = [6, 4.5]
        plt.rcParams["savefig.dpi"] = 300

        self.cellOc_object = cellOc_object
        self.link_object = link_object
        self.plot_dir = plot_dir
        logging.info("Analysis object created, performing anaylsis")
        self._perform_analysis()

    def _perform_analysis(self):
        if self.link_object is None:
            logging.info("No link model provided")
            print("No link model provided")
            return
        if self.cellOc_object is None:
            logging.info("No GRN model provided")
            print("No GRN model provided")
            return
       # self.link_object.plot_degree_distributions(plot_model=True, save=f"{self.plot_dir}/degree_distribution/")
        self.link_object.get_network_score()
        self.link_object.merged_score.head()
        self.link_object.plot_scores_as_rank(cluster="Megakaryocytes", n_gene=30,save=f"{self.plot_dir}/ranked_score/")
        print("Score ranked start")
        self.link_object.plot_score_comparison_2D(
            value="eigenvector_centrality",
            cluster1="Megakaryocytes",
            cluster2="Caudal Mesoderm",
            percentile=98,
            save=f"{self.plot_dir}/score_comparison_1")
        self.link_object.plot_score_comparison_2D(
            value="betweenness_centrality",
            cluster1="Megakaryocytes",
            cluster2="Caudal Mesoderm",
            percentile=98,
            save=f"{self.plot_dir}/score_comparison_2"
        )

        self.link_object.plot_score_comparison_2D(
            value="degree_centrality_all",
            cluster1="Megakaryocytes",
            cluster2="Caudal Mesoderm",
            percentile=98,
            save=f"{self.plot_dir}/score_comparison_3"
        )
        print("Score comparison done")
        self.link_object.plot_score_per_cluster(goi="Gata1", save=f"{self.plot_dir}/network_score_per_gene")
        self.link_object.plot_score_per_cluster(goi="Gata4")
        plt.subplots_adjust(left=0.15, bottom=0.3)
        plt.ylim([0.0, 0.040])
        self.link_object.plot_score_discributions(values=["degree_centrality_all"],method="boxplot",save=f"{self.plot_dir}/boxplot")
        plt.ylim([0.0, 0.28])
        self.link_object.plot_score_discributions(values=["eigenvector_centrality"],method="boxplot",save=f"{self.plot_dir}/boxplot2")
        self._plot_network_entropy_distributions(self.link_object, save=f"{self.plot_dir}/boxplot3")
        plt.savefig(f"{self.plot_dir}/boxplot3")

    def _plot_network_entropy_distributions(self,links, update_network_entropy=False, save=None):
        """
        Plot the distribution of network entropy.
        See the CellOracle paper for the detail.

        Args:
            links (Links object): See network_analisis.Links class for detail.
            values (list of str): The list of netwrok score type. If it is None, all network score (listed above) will be used.
            update_network_entropy (bool): Whether to recalculate network entropy.
            save (str): Folder path to save plots. If the folde does not exist in the path, the function create the folder.
                If None plots will not be saved. Default is None.
        """
        settings = {"save_figure_as": "png"}

        if links.entropy is None:
            links.get_network_entropy()

        if update_network_entropy:
            links.get_network_entropy()

        # fig = plt.figure()

        ax = sns.boxplot(data=links.entropy, x="cluster", y="entropy_norm",
                    palette=links.palette.palette.values,
                    order=links.palette.index.values, fliersize=0.0)

        ax.tick_params(axis="x", rotation=90)
        ax.set_ylim([0.6,1.1])

        if not save is None:
            os.makedirs(save, exist_ok=True)
            path = os.path.join(save, f"network_entropy_in_{links.name}_{links.threshold_number}.{settings['save_figure_as']}")
            ax.set_ylabel("normalized\nentropy")
            plt.savefig(path, transparent=True)
        plt.show()

