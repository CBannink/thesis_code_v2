import logging
import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from CellOracle import celloracle as co
from datetime import datetime
#import modified_celloracle as co

class Inference:

    def __init__(self, oracle_path: str, links_path:str, fit_for_grn_sim:bool=False):
        self.oracle = co.load_hdf5(oracle_path)
        self.links = co.load_hdf5(links_path)
        print(self.oracle)
        self.oracle.get_cluster_specific_TFdict_from_Links(self.links)
        print(self.oracle)
        if fit_for_grn_sim:
            print("Fitting for GRN simulation")
            self._fit_for_grn_sim()


    def _fit_for_grn_sim(self):
        self.oracle.fit_GRN_for_simulation(alpha=10,use_cluster_specific_TFdict=True)
        self.oracle.to_hdf5("../../celloracle_data/celloracle_object/GRN_trained.celloracle.oracle")

    def precompute(self):
        self.oracle._init_neighbors_umap()

    def perturb(self, perturb_condition: dict, propagate:int, calc_prob:bool=False):
        self.oracle._set_embedding_name("X_umap")
        print("Simulating perturbation")
        current_time = datetime.now().strftime("%H:%M:%S")
        print("Current Time =", current_time)
        self.oracle.simulate_shift(perturb_condition=perturb_condition,n_propagation=propagate)
        next_time = datetime.now().strftime("%H:%M:%S")
        #print the difference in time
        print("Current Time =", next_time)
        print("difference in time", datetime.strptime(next_time, "%H:%M:%S") - datetime.strptime(current_time, "%H:%M:%S"))
        if calc_prob:
            self._calc_prob_trans_shift()

    def init(self):
        self.oracle.init()
    def batch_perturbation(self, batch_size:int=512):
        #[18500, 10349], ' with tfs: ', [('T', 0.0), ('Mecom', 0.0)]
        # [4089, 18373], [21605, 27662]
        import random as rd
        rd.seed(39)
        #retrieve 16 random cells indices to perturb
        cells_to_perturb = []
        perturbations = []
        adata_indx = []
        for _ in range(batch_size):
            # cells_to_perturb.append(rd.randint(0, self.oracle.adata.shape[0]))
            intrd = rd.randint(0, self.oracle.adata.shape[0])
            while intrd in cells_to_perturb:
                intrd = rd.randint(0, self.oracle.adata.shape[0])
            cells_to_perturb.append(intrd)
            perturbations.append(("Atf6", 0.0))
            adata_indx.append(self.oracle.adata[intrd].obs.index.tolist()[0])


        print("all indices that are perturbed: ", adata_indx)
        print("all numerical indices:  ", cells_to_perturb)
            
        already_used_tfs =[]
        shifted_coord_dict = {}
        print("perturbing: ", len(cells_to_perturb))
        #start timer
        start = datetime.now()
        og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts =  self.oracle.training_phase_inference_batch(batch_size=batch_size,perturb_condition=perturbations, idxs=cells_to_perturb, n_propagation=3, n_neighbors=200)

        #print difference in timer
        print("operation took: ",datetime.now() - start ," seconds")
        print(delta_shifts)
        #shifted coord dict needs to have a list for each key with a list fo the shifted coords for that cell
        # shifted_coord_dict = {cell: [shifted_coord[i]] for i, cell in enumerate(cells_to_perturb)}
        # tf_perturbed_dict = {cell: perturbations[i][0] for i, cell in enumerate(cells_to_perturb)}
        print("how many shifts: ", len(shifted_coord_dict))
        # for i,idx in enumerate(cells_to_perturb):
        #     print(f"index: {idx} with celltype: ", self.oracle.adata.obs["celltype"][idx], " has shifted coords: ", shifted_coord_dict[idx], " with tf: ", tf_perturbed_dict[idx], " with delta shift: ", delta_shifts[i])

        #self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)
    def calculate_embedding_coord(self, idxs):
        idxs = [1378, 2945, 20623, 28406, 14050, 2002]
        print(len(idxs))
        for idx in idxs:
            delta_embed = self.oracle.delta_embedding
            shift = delta_embed[idx]
            adata_umap = self.oracle.adata.obsm["X_pca"][idx]
            #calculate total distance of the shift
            distance = np.linalg.norm(shift)
            print("distance: ", distance)
            print("celltype: ", self.oracle.adata.obs["celltype"][idx])
            print("shift: ", shift)
            print("adata pca: ", adata_umap)
            print("new coord: ", adata_umap + shift)

    def temp(self):
        self.oracle.temp()



    def _calc_prob_trans_shift(self):
        self.oracle.estimate_transition_prob(n_neighbors=200,knn_random=True,sampled_fraction=1)
        self.oracle.calculate_embedding_shift(sigma_corr=0.05)

    def plot_result_after_trans_prob(self):
        goi = "Gata1"
        fig, ax = plt.subplots(1, 2, figsize=[13, 6])
        scale = 30
        # Show quiver plot
        self.oracle.plot_quiver(scale=scale, ax=ax[0])
        ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
        # Show quiver plot that was calculated with randomized graph.
        self.oracle.plot_quiver_random(scale=scale, ax=ax[1])
        ax[1].set_title(f"Randomized simulation vector")
        plt.show()
    #
    # def violin_plot(self):
    #     sc.pl.violin(self.oracle.adata, keys="Gata1", groupby=self.oracle.cluster_column_name)
    #     plt.show()