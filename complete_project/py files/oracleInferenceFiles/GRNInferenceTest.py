import logging
import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import random as rd
from CellOracle import celloracle as co
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import pickle
import cupy as cp

class InferenceTest:

    def __init__(self, oracle_path: str, links_path:str, fit_for_grn_sim:bool=False):
        print("Loading links")
        self.links = co.load_hdf5(links_path)
        print("Loading oracle")
        self.oracle = co.load_hdf5(oracle_path)
        print("Oracle loaded")
        self.dir_oracle = os.path.dirname(oracle_path)
        self.oracle.get_cluster_specific_TFdict_from_Links(self.links)
        self.oracle_compare = None
        print("Cluster specific TF dict loaded")
        print(self.oracle)

        #only used for another ffunction tehese vars
        self.oracle_cupy = None
        self.oracle_numpy = None
        if fit_for_grn_sim:
            self._fit_for_grn_sim()
            
    def precompute(self):
        self.oracle._init_neighbors_umap()
        #self.oracle.precompute_kda_tree_on_pca_embedding()

    def compute_active_reg_genes(self):
        reg_genes = self.oracle.return_active_reg_genes()
        print("Active reg genes: ", reg_genes)
        all_genes = self.oracle.adata.var.index.tolist()
        # activity = self.oracle.adata.
        #i want to get the average number of genes that are active in each cell
        reg_gene_adata_indices = np.array([all_genes.index(g) for g in reg_genes])
        n_cells = self.oracle.adata.shape[0]
        # Initialize arrays to store the count of active genes FOR EACH CELL
        cell_activity_X_counts = np.zeros(n_cells, dtype=int)
        cell_activity_imputed_counts = np.zeros(n_cells, dtype=int)
        data_X = self.oracle.adata.X
        data_imputed_count = self.oracle.adata.layers["imputed_count"]
        for i in range(n_cells):
            active_genes_in_cell_X = 0
            active_genes_in_cell_imputed = 0

            for gene_idx in reg_gene_adata_indices:
                expression_X = data_X[i, gene_idx]
                expression_imputed_count = data_imputed_count[i, gene_idx]



                if expression_X > 0.1:
                    active_genes_in_cell_X += 1
                if expression_imputed_count > 0.1:
                    active_genes_in_cell_imputed += 1
            cell_activity_X_counts[i] = active_genes_in_cell_X
            cell_activity_imputed_counts[i] = active_genes_in_cell_imputed

        #print the average number of active genes
        print("Average number of active genes in X: ", np.mean(cell_activity_X_counts))
        print("Average number of active genes in imputed count: ", np.mean(cell_activity_imputed_counts))

    def getadata(self):
        return self.oracle.adata

    def print_pca(self):
        print(self.oracle.adata.obsm["X_pca"].shape)


    def _fit_for_grn_sim(self):
        self.oracle.fit_GRN_for_simulation(alpha=10,use_cluster_specific_TFdict=True)
        self.oracle.to_hdf5(self.dir_oracle + "/GRN_trained.celloracle.oracle")

    def over_write_neighbors_adata(self, new_adata):
        oracle_adata = self.oracle.adata
        neighbor_indices = new_adata.obsm["pca_neighbors"]
        sparse_matrix = new_adata.obsp["pca_neighbors_sparse"]
        oracle_adata.obsm["pca_neighbors"] = neighbor_indices
        # 4. Compute and store sparse connectivity matrix
        oracle_adata.uns["pca_neighbors_sparse"] = sparse_matrix
        oracle_adata.obsp["pca_neighbors_sparse"] = sparse_matrix
        #save it
        self.oracle.to_hdf5(self.dir_oracle + "/GRN_trained.celloracle.oracle")

    def test_init(self):
        print("Testing init")
        self.oracle.init(embedding_type="umap", n_neighbors=200, torch_approach = False)

    def batch_perturbation_defined(self, batch_size:int=16):

        rd.seed(7)
        #retrieve 16 random cells indices to perturb
        rd.seed(7)
        # retrieve all cell indices to perturb
        cells_to_perturb = []
        perturbations = []
        og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts =  self.oracle.training_phase_inference_batch(batch_size=batch_size,perturb_condition=perturbations, idxs=cells_to_perturb, n_propagation=3, n_neighbors=200)
        #shifted coord dict needs to have a list for each key with a list fo the shifted coords for that cell
        shifted_coord_dict = {cell: [shifted_coord[i]] for i, cell in enumerate(cells_to_perturb)}
        tf_perturbed_dict = {cell: perturbations[i][0] for i, cell in enumerate(cells_to_perturb)}
        print("detla shifts: ", delta_shifts)
        print('len of delta shifts: ', len(delta_shifts))
        for i,idx in enumerate(cells_to_perturb):
            print(f"index: {idx} with celltype: ", self.oracle.adata.obs["celltype"][idx], " has shifted coords: ", shifted_coord_dict[idx], " with tf: ", tf_perturbed_dict[idx], " with delta shift: ", delta_shifts[i])
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)


    def set_different_adata(self):
        otherData = sc.read_h5ad("../celloracle_data/scrna_final_celloc/V_unspliced_norm_log_W_MESC/untrimmed_og_umap/train_data.h5ad")
        sc.pl.umap(self.oracle.adata, color="celltype", title="Oracle")
        self.copy_data = self.oracle.adata.copy()
        self.oracle.set_adata(otherData)
        sc.pl.umap(self.oracle.adata, color="celltype", title="Oracle")


    def compare_multiple_cellocs(self, oracle_path:str, links_path:str, batch_size:int=64):
        if self.oracle_compare is None:
            self.oracle_compare = co.load_hdf5(oracle_path)
            links_ = co.load_hdf5(links_path)
            self.oracle_compare.get_cluster_specific_TFdict_from_Links(links_)
            self.oracle_compare.init(embedding_type="umap", n_neighbors=200, torch_approach = False)
        #get a list of indieces by the intesection of both adata indices


        adata = self.oracle.adata
        adata_new = self.oracle_compare.adata
        adata_idx = adata.obs.index
        adata_new_idx = adata_new.obs.index
        intersect_list = list(set(adata_idx).intersection(set(adata_new_idx)))
        cells_to_perturb = np.random.choice(intersect_list, size=batch_size, replace=False)
        #check if any celltypes of the intersect list are celltypes of mESCs
        if "mESC" in adata[cells_to_perturb].obs["celltype"].unique():
            print("Found mESC celltype in intersect list")

        cells_to_perturb_int  = [adata.obs.index.get_loc(cell) for cell in cells_to_perturb]
        cells_to_perturb_int_new = [adata_new.obs.index.get_loc(cell) for cell in cells_to_perturb]
        #then replace this umap with the umap in the adata of the ogringal one

        print(cells_to_perturb_int)
        print(cells_to_perturb_int_new)
        #find intersect of vars to perturb
        reg_genes =self.oracle.return_active_reg_genes()
        other_reg_genes = self.oracle_compare.return_active_reg_genes()
        #intersect
        intersect_reg_genes = list(set(reg_genes).intersection(set(other_reg_genes)))
        perturbs = np.random.choice(intersect_reg_genes, size=batch_size, replace=True)
        perturbs = [(perturbation, 0.0) for perturbation in perturbs]
        og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts = self.oracle.training_phase_inference_batch(batch_size=batch_size, perturb_condition=perturbs, idxs=cells_to_perturb_int, n_propagation=3,n_neighbors=200)
        og_umap_new, new_idx_new, shifted_coord_new, og_shifted_umap_new, delta_shifts_new = self.oracle_compare.training_phase_inference_batch(batch_size=batch_size, perturb_condition=perturbs, idxs=cells_to_perturb_int_new, n_propagation=3,n_neighbors=200)
        print("average delta shift: ", np.mean(delta_shifts))
        print("average delta shift new: ", np.mean(delta_shifts_new))
        shifted_coord_dict = {cell: [shifted_coord[i]] for i, cell in enumerate(cells_to_perturb_int)}
        tf_perturbed_dict = {cell: perturbs[i][0] for i, cell in enumerate(cells_to_perturb_int)}
        shifted_coord_dict_new = {cell: [shifted_coord_new[i]] for i, cell in enumerate(cells_to_perturb_int_new)}
        tf_perturbed_dict_new = {cell: perturbs[i][0] for i, cell in enumerate(cells_to_perturb_int_new)}
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)
        self.oracle_compare.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict_new, tf_perturbed_dict_new)
        sc.pl.umap(self.oracle.adata, color="celltype", title="Oracle")
        sc.pl.umap(self.oracle_compare.adata, color="celltype", title="Oracle compare")


    def batch_perturbation(self, batch_size:int=64, use_index_choice:bool=False):
        #[18500, 10349], ' with tfs: ', [('T', 0.0), ('Mecom', 0.0)]
        # [4089, 18373], [21605, 27662]
        rd.seed(7)
        #retrieve 16 random cells indices to perturb
        cells_to_perturb = []
        for _ in range(batch_size):
            # cells_to_perturb.append(rd.randint(0, self.oracle.adata.shape[0]))
            intrd = rd.randint(0, self.oracle.adata.shape[0])
            while intrd in cells_to_perturb:
                intrd = rd.randint(0, self.oracle.adata.shape[0])
            cells_to_perturb.append(intrd)
        # cells_to_perturb=[17150]#4089]#,20932]17150 20932


        already_used_tfs =[]
        shifted_coord_dict = {}
        perturbations = []
        print("perturbing: ", len(cells_to_perturb))
        for i in range(batch_size):
            tf = self.get_possible_tf_for_idx(cells_to_perturb[i], already_used_tfs)
            perturbations.append((tf,0.0))

        if use_index_choice:
            all_adata_idx = self.oracle.adata.obs.index
            cells_to_perturb = np.random.choice(all_adata_idx, size=batch_size, replace=False)
            #find the correct integer indices
            print(cells_to_perturb)
            cells_to_perturb = [self.oracle.adata.obs.index.get_loc(cell) for cell in cells_to_perturb]
            print(cells_to_perturb)
            all_vars = self.oracle.adata.var.index
            perturbations =  np.random.choice(all_vars, size=batch_size, replace=False)
            perturbations = [("Scrt2", 0.0) for perturbation in perturbations]
        # for i in range(len(self.oracle.adata.obs.index)):
        #     cells_to_perturb.append(i)
        #     perturbations.append(("Scrt2", 0.0))
        # perturbations = [("Scrt2", 0.0)]#,("Gli3", 0.0)]Scrt2
        #start timer
        start = datetime.now()
        og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts =  self.oracle.training_phase_inference_batch(batch_size=batch_size,perturb_condition=perturbations, idxs=cells_to_perturb, n_propagation=3, n_neighbors=200)
        #print difference in timer
        print("operation took: ",datetime.now() - start ," seconds")
        #shifted coord dict needs to have a list for each key with a list fo the shifted coords for that cell
        shifted_coord_dict = {cell: [shifted_coord[i]] for i, cell in enumerate(cells_to_perturb)}
        tf_perturbed_dict = {cell: perturbations[i][0] for i, cell in enumerate(cells_to_perturb)}
        print("how many shifts: ", len(shifted_coord_dict))
        for i,idx in enumerate(cells_to_perturb):
            print(f"index: {idx} with celltype: ", self.oracle.adata.obs["celltype"][idx], " has shifted coords: ", shifted_coord_dict[idx], " with tf: ", tf_perturbed_dict[idx], " with delta shift: ", delta_shifts[i])
        print("Average delta shift: ", np.mean(delta_shifts))
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)

    def batch_perturbation_seq(self, batch_size: int = 3):
        rd.seed(7)
        # retrieve 16 random cells indices to perturb
        cells_to_perturb = []
        for _ in range(batch_size):
            # cells_to_perturb.append(rd.randint(0, self.oracle.adata.shape[0]))
            intrd = rd.randint(0, self.oracle.adata.shape[0])
            while intrd in cells_to_perturb:
                intrd = rd.randint(0, self.oracle.adata.shape[0])
            cells_to_perturb.append(intrd)
        already_used_tfs = []
        shifted_coord_dict = {}
        tf_perturbed_dict = {}
        perturbations = []
        start = datetime.now()

        print("perturbing: ", len(cells_to_perturb))
        for z in range(4):
            perturbations.append([])
            for i in range(batch_size):
                if z==0:
                    already_used_tfs.append([])
                    shifted_coord_dict[cells_to_perturb[i]] = []
                    tf_perturbed_dict[cells_to_perturb[i]] = []
                tf = self.get_possible_tf_for_idx(cells_to_perturb[i], already_used_tfs[i])
                already_used_tfs[i].append(tf)
                perturbations[z].append((tf, 0.0))
            og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts = self.oracle.training_phase_inference_batch(
                batch_size=batch_size, perturb_condition=perturbations[z], idxs=cells_to_perturb, n_propagation=3,
                n_neighbors=200)
            for i in range (batch_size):
                shifted_coord_dict[cells_to_perturb[i]].append(shifted_coord[i])
                tf_perturbed_dict[cells_to_perturb[i]].append(perturbations[z][i][0])


        # print difference in timer
        print("operation took: ", datetime.now() - start, " seconds")
        # shifted coord dict needs to have a list for each key with a list fo the shifted coords for that cell




        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)

    def batch_perturbation_sin(self, batch_size:int=1):
        rd.seed(7)
        #retrieve 16 random cells indices to perturb
        cells_to_perturb = []
        for _ in range(batch_size):
            # cells_to_perturb.append(rd.randint(0, self.oracle.adata.shape[0]))
            cells_to_perturb.append(rd.randint(0, self.oracle.adata.shape[0]))
        cells_to_perturb = [11982]
        already_used_tfs =[]
        shifted_coord_dict = {}
        perturbations = []
        for i in range(batch_size):
            tf = self.get_possible_tf_for_idx(cells_to_perturb[i], already_used_tfs)
            tf = "Nfib"
            perturbations.append((tf,0.0))
        og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts =  self.oracle.training_phase_inference_batch(batch_size=batch_size,perturb_condition=perturbations, idxs=cells_to_perturb, n_propagation=3, n_neighbors=200)
        #shifted coord dict needs to have a list for each key with a list fo the shifted coords for that cell
        shifted_coord_dict = {cell: [shifted_coord[i]] for i, cell in enumerate(cells_to_perturb)}
        tf_perturbed_dict = {cell: perturbations[i][0] for i, cell in enumerate(cells_to_perturb)}
        print("detla shifts: ", delta_shifts)
        print('len of delta shifts: ', len(delta_shifts))
        for i,idx in enumerate(cells_to_perturb):
            print(f"index: {idx} with celltype: ", self.oracle.adata.obs["celltype"][idx], " has shifted coords: ", shifted_coord_dict[idx], " with tf: ", tf_perturbed_dict[idx], " with delta shift: ", delta_shifts[i])
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)

    def perform_some_random_perturbs(self, context):

        # cells_to_pertub, tfs = self.extract_index_tf(context)
        cells_to_pertub = [17150]
        tfs = ["Scrt2"]
        print("len: ", len(cells_to_pertub), " len perturbations: ", len(tfs))
        shifted_coord_dict = {}
        tf_perturbed_dict = {}
        save_shifted_coord=[]
        save_tfs = []
        save_shift =[]
        save_cells = []
        #currentt ime
        start = datetime.now()
        for idx,cell in enumerate(cells_to_pertub):
            # tf = self.get_possible_tf_for_idx(cell, already_used_tfs)
            tf = tfs[idx]
            value =0.0
            og_umap, new_idx, shifted_coord, og_shifted_umap, shift_umap = self.oracle.training_phase_inference(perturb_condition={tf: 0.0}, idx=cell, n_propagation=3)
            if cell not in shifted_coord_dict:
                shifted_coord_dict[cell] = []
            if cell not in tf_perturbed_dict:
                tf_perturbed_dict[cell] = []
            save_shifted_coord.append(shifted_coord)
            save_tfs.append(tf)
            save_cells.append(cell)
            save_shift.append(shift_umap)
            shifted_coord_dict[cell].append(shifted_coord)
            tf_perturbed_dict[cell].append(tf)
           # print(f"index: {cell} with celltype: ", self.oracle.adata.obs["celltype"][cell], " has shifted coords: ", shifted_coord, " with tf: ", tf, " with delta shift: ", shift_umap)
        print("operation took: ",datetime.now() - start ," seconds")
        for i in range(len(save_cells)):
            print(f"index: {save_cells[i]} with celltype: ", self.oracle.adata.obs["celltype"][save_cells[i]], " has shifted coords: ", save_shifted_coord[i], " with tf: ", save_tfs[i], " with delta shift: ", save_shift[i])
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)


    def calculate_embedding_coord(self, idx):
        delta_embed =self.oracle.delta_embedding
        shift = delta_embed[idx]
        adata_umap = self.oracle.adata.obsm["X_umap"][idx]
        return adata_umap+shift

    def perform_multiple_perturbs_batch(self,batch_size:int=1):
        rd.seed(7)
        #retrieve 16 random cells indices to perturb
        cells_to_perturb = []
        for _ in range(batch_size):
            # cells_to_perturb.append(rd.randint(0, self.oracle.adata.shape[0]))
            cells_to_perturb.append(1378)
        celltypes = { cell: self.oracle.adata.obs["celltype"][cell] for cell in cells_to_perturb}
        already_used_tfs =[]
        shifted_coord_dict = {}
        tf_perturbed_dict = {}
        for i in range(batch_size):
            # tf = self.get_possible_tf_for_idx(cells_to_perturb[i], already_used_tfs)
            tf = "Atf6"
            already_used_tfs.append(tf)
            tf_perturbed_dict[tf] = 0.0
        og_umap, new_idx, shifted_coord, og_shifted_umap, _ =  self.oracle.training_phase_inference_batch(batch_size=batch_size,perturb_condition=tf_perturbed_dict, idxs=cells_to_perturb, n_propagation=3, n_neighbors=200)
        shifted_coord_dict = {cell: shifted_coord[i] for i, cell in enumerate(cells_to_perturb)}
        tf_perturbed_dict = {"Atf6": 0.0}
        print("shifted coord dict: ", shifted_coord_dict)
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)



    def perform_multiple_perturbs(self):

        rd.seed(7)
        adata = self.oracle.adata
        #get a idx for cell wit celltype = "mESCs"
        idx_mESCs = adata.obs[adata.obs["celltype"] == "mESCs"].index[0]
        idx_mESCs = adata.obs.index.get_loc(idx_mESCs)

        #so now the idx is like a speical index, we want the list index

        celltypes = [self.oracle.adata.obs["celltype"][idx_mESCs]]
        celltype_idx_to_perturb = {celltypes[0]: idx_mESCs}
        idxs=[]
        shifted_coord_dict = {}
        tf_perturbed_dict = {}
        random = False
        tfs_done = []
        for celltype in celltypes:
            og_idx = celltype_idx_to_perturb[celltype]
            idx_loop_change = og_idx
            print(f"Original index {og_idx} with celltype: " , self.oracle.adata.obs["celltype"][idx_loop_change])

            # if we ever want to add some randomness
            if random:
                idx_loop_change = rd.randint(0, adata.shape[0])
                while not adata[idx_loop_change, tf].X != 0:
                    idx_loop_change = rd.randint(0, adata.shape[0])
            # end randomness
            for _ in range(1):
                idxs.append(idx_loop_change)
                tf = self.get_possible_tf_for_idx(idx_loop_change, tfs_done)
                tfs_done.append(tf)
                print(f"Perturbing {tf} in {celltype} at index {idx_loop_change}")
                og_umap, new_idx, shifted_coord, og_shifted_umap, _ = self.oracle.training_phase_inference(perturb_condition={tf: 0.0},idx=idx_loop_change, n_propagation=3)
                if og_idx not in shifted_coord_dict:
                    shifted_coord_dict[og_idx] = []
                if og_idx not in tf_perturbed_dict:
                    tf_perturbed_dict[og_idx] = []
                shifted_coord_dict[og_idx].append(shifted_coord)
                tf_perturbed_dict[og_idx].append(tf)
                idx_loop_change = new_idx
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)
        for i in idxs:
            print(f"index: {i} with celltype: ", self.oracle.adata.obs["celltype"][i])
        import scanpy as sc

        # Basic UMAP plot
        sc.pl.umap(self.oracle.adata, color="celltype")
        #check if there are any negative coords in umap:
        print(np.min(self.oracle.adata.obsm["X_umap"]))


    def compare_cupy_np_oracles(self, compare_with_og_implementation:bool=False, batch_size:int=256):
        data_path_new_data = os.path.join('../celloracle_data',"celloracle_object/new_promoter_without_mescs_trimmed_test_own_umap/ready_oracle.pkl")
        if self.oracle_cupy is None:
            with open(data_path_new_data, 'rb') as f: self.oracle_cupy = pickle.load(f)
            with open(data_path_new_data, 'rb') as f: self.oracle_np = pickle.load(f)
            self.oracle_cupy.init(embedding_type="umap", n_neighbors=200, torch_approach = False, cupy_approach=True)
            self.oracle_np.init(embedding_type="umap", n_neighbors=200, torch_approach = False, cupy_approach=False)

        adata = self.oracle_cupy.adata
        adata_new = self.oracle_np.adata
        adata_idx = adata.obs.index
        adata_new_idx = adata_new.obs.index
        intersect_list = list(set(adata_idx).intersection(set(adata_new_idx)))
        cells_to_perturb = np.random.choice(intersect_list, size=batch_size, replace=False)
        # check if any celltypes of the intersect list are celltypes of mESCs
        if "mESC" in adata[cells_to_perturb].obs["celltype"].unique():
            print("Found mESC celltype in intersect list")

        cells_to_perturb_int = [adata.obs.index.get_loc(cell) for cell in cells_to_perturb]
        cells_to_perturb_int_new = [adata_new.obs.index.get_loc(cell) for cell in cells_to_perturb]
        # then replace this umap with the umap in the adata of the ogringal one
        print("are ints lists the same?: ", cells_to_perturb_int == cells_to_perturb_int_new)
        # find intersect of vars to perturb
        reg_genes = self.oracle_cupy.return_active_reg_genes()
        other_reg_genes = self.oracle_np.return_active_reg_genes()
        # intersect
        intersect_reg_genes = list(set(reg_genes).intersection(set(other_reg_genes)))
        perturbs = np.random.choice(intersect_reg_genes, size=batch_size, replace=True)
        perturbs = [(perturbation, 0.0) for perturbation in perturbs]

        if compare_with_og_implementation:
            perturbs = [("Scrt2", 0.0) for perturbation in perturbs]
        try:
            cp.cuda.Device(0).use()
            print("CuPy is available and GPU is accessible.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            # Handle the case where CuPy cannot access the GPU
            print("CuPy cannot access GPU, falling back to CPU integration.")
        timer_cupy = datetime.now()
        og_umap, new_idx, shifted_coord, og_shifted_umap, delta_shifts = self.oracle_cupy.training_phase_inference_batch_cp(batch_size=batch_size, perturb_condition=perturbs, idxs=cells_to_perturb_int, n_propagation=3, n_neighbors=200)
        print("operation took: ", datetime.now() - timer_cupy, " seconds")
        timer_np = datetime.now()
        og_umap_new, new_idx_new, shifted_coord_new, og_shifted_umap_new, delta_shifts_new = self.oracle_np.training_phase_inference_batch(batch_size=batch_size, perturb_condition=perturbs, idxs=cells_to_perturb_int_new, n_propagation=3, n_neighbors=200)
        print("operation took: ", datetime.now() - timer_np, " seconds")
        #check if they produce the same results
        print("are the shifted coords the same?: ", np.allclose(shifted_coord, shifted_coord_new, atol=1e-5))
        shifted_coord_dict = {cell: [shifted_coord[i]] for i, cell in enumerate(cells_to_perturb_int)}
        tf_perturbed_dict = {cell: perturbs[i][0] for i, cell in enumerate(cells_to_perturb_int)}
        shifted_coord_dict_new = {cell: [shifted_coord_new[i]] for i, cell in enumerate(cells_to_perturb_int_new)}
        tf_perturbed_dict_new = {cell: perturbs[i][0] for i, cell in enumerate(cells_to_perturb_int_new)}
        self.oracle_cupy.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)
        self.oracle_np.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict_new,
                                                                                 tf_perturbed_dict_new)
        if compare_with_og_implementation:
            perturbs = {"Scrt2": 0.0}
            self.oracle_np.simulate_shift(perturb_condition=perturbs, n_propagation=3, GRN_unit="cluster")
            self.oracle_np.estimate_transition_prob(n_neighbors=200, knn_random=True)
            delta_shifts_og =  self.oracle_np.calculate_embedding_shift(sigma_corr=0.05)
            same_indices_for_delta_shifts =delta_shifts_og[cells_to_perturb_int]
            print("equal to delta shifts np: ", np.allclose(delta_shifts_new, same_indices_for_delta_shifts, atol=1e-3))
            print("equal to delta shifts cp: ", np.allclose(delta_shifts, same_indices_for_delta_shifts, atol=1e-3))
            print("equal to delta shifts np: ", np.allclose(delta_shifts_new, same_indices_for_delta_shifts, atol=1e-4))
            print("equal to delta shifts cp: ", np.allclose(delta_shifts, same_indices_for_delta_shifts, atol=1e-4))


    def do_some_nice_mappings(self):
        shifted_coord_dict = {7734:[]}
        tf_perturbed_dict = {7734:[]}
        tf = "Gata1"
        tf_perturbed_dict[7734].append(tf)
        shifted_coord_dict[7734].append(self.oracle.adata.obsm["X_umap"][21253])
        self.oracle.plot_umap_shifts_with_multiple_tfs_multiple_perturbs(shifted_coord_dict, tf_perturbed_dict)

    def debug(self):
        idx =507
        cellttype = "Early Motor Neurons"
        tf = "Rnf43"
        self.oracle.training_phase_inference(perturb_condition={tf:0.0}, idx=idx,n_propagation=3)


    def get_tf_with_counts(self, celltypes1, celltypes2):
        """Return a tf for a pair of celltypes that has a index for a cell with a count>0"""
        adata = self.oracle.adata
        tfs =[]
        for idx in range(len(celltypes1)):
            celltype1 = celltypes1[idx]
            celltype2 = celltypes2[idx]
            subset_adata = adata[adata.obs["celltype"].isin([celltype1, celltype2])]
            for tf in adata.var.index:
                if subset_adata[:, tf].X.sum() > 0 and tf not in tfs:
                    print(f"TF {tf} has counts for {celltype1} and {celltype2}")
                    tfs.append(tf)
                    break
        return tfs

    def get_possible_perturbs(self) -> list:
        return self.oracle.return_active_reg_genes()

    def get_possible_tf_for_idx(self, idx:int, list_of_used_tfs= None):
        adata = self.oracle.adata
        list_of_possible_tfs = self.get_possible_perturbs()
        tfs_that_are_in_var_index = [tf for tf in list_of_possible_tfs if tf in self.oracle.get_genes()]
        #hussle up the list
        rd.shuffle(tfs_that_are_in_var_index)
        #tfs_that_are_in_var_index = adata.var.index
        difference = len(tfs_that_are_in_var_index) - len(list_of_used_tfs)
        for tf in tfs_that_are_in_var_index:
            index = self.oracle.get_gene_index(tf)
            if not adata[idx, index].X.sum() > 0:
                continue
            if list_of_used_tfs is not None and tf in list_of_used_tfs:
                #print(f"TF {tf} already used")
                continue
            return tf

        raise ValueError(f"No TF with count > 0 for this cell with index {idx}")


    def perturb(self, perturb_condition: dict, propagate:int, calc_prob:bool=False):
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

    def perform_custom_inference_iteration(self, perturb_condition: dict, propagate:int, cell_idx:int = None):
        #generate random number for the perturbation
        if cell_idx is None:
            cell_idx = rd.randint(0, self.oracle.adata.shape[0])
            adata = self.oracle.adata
            #check if the row for cell_idx has a non zero value for the perturb_condition, dict consits of only one item, then make new random cell idx
            tf  = list(perturb_condition.keys())[0]
            print(tf)
            while adata[cell_idx, tf].X == 0:
                cell_idx = rd.randint(0, self.oracle.adata.shape[0])
        print("Cell index: ", cell_idx)
        self.oracle.training_phase_inference(perturb_condition=perturb_condition,idx= cell_idx,n_propagation=propagate )


    def _calc_prob_trans_shift(self):
        self.oracle.estimate_transition_prob(n_neighbors=200,knn_random=True,sampled_fraction=1)
        self.oracle.calculate_embedding_shift(sigma_corr=0.05)

    def plot_result_after_trans_prob(self):
        logging.info(self.oracle.adata.layers["imputed_count"])
        goi = "Gata1"
        fig, ax = plt.subplots(1, 2, figsize=[13, 6])
        scale = 8
        # Show quiver plot
        self.oracle.plot_adjusted_quiver(scale=scale, ax=ax[0])
        ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
        # Show quiver plot that was calculated with randomized graph.
        # self.oracle.plot_quiver_random(scale=scale, ax=ax[1])
        # ax[1].set_title(f"Randomized simulation vector")
        plt.show()

    def plot_pca(self):
        self.oracle.plot_pca()

    def get_oracle(self):
        return self.oracle

    def analyze(self):
        self.oracle.perform_PCA(n_components=50)
        self.oracle.analyze_precomputecd_pca(28459)

    def print_pca_embedding(self):
        print(self.oracle.adata.obsm["X_pca"][28459])

    def check_index(self):
        print(self.oracle.adata.obs["celltype"][28459])

    def retrieve_other_cells(self, cell_idx:int = 28459):
        celltype = self.oracle.adata.obs["celltype"][cell_idx]
        same_cells = self.oracle.adata[self.oracle.adata.obs["celltype"] == celltype]
        other_cells = same_cells[same_cells.obs.index != self.oracle.adata.obs.index[cell_idx]]
        print(len(other_cells))
        print(other_cells)
        pca_embed = other_cells.obsm["X_pca"]
        #calculate distance between the cells
        distances = []
        for i in range(len(pca_embed)):
            dist = np.linalg.norm(self.oracle.adata.obsm["X_pca"][cell_idx] - pca_embed[i])
            distances.append(dist)
        print(distances)

    def plot_umap(self):
        sc.pl.umap(self.oracle.adata, color="celltype")
        # plt.show()

    def reset_pca_and_umap(self, pca_n_components:int = 50, neighbors:int = 500):
        """"make umap by first pca embedding and then umap"""
        adata = self.oracle.adata
        sc.tl.pca(adata, n_comps=pca_n_components, svd_solver='arpack')
        knn = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1)
        knn.fit(adata.obsm["X_pca"])
        neighbor_indices_raw = knn.kneighbors(return_distance=False)
        # Remove self-references from indices (first column if it's self, last column if not)
        neighbor_indices = np.array([
            row[1:] if row[0] == i else row[:-1]
            for i, row in enumerate(neighbor_indices_raw)
        ])
        # 3. Store neighbor indices as array in obsm
        adata.obsm["pca_neighbors"] = neighbor_indices
        # 4. Compute and store sparse connectivity matrix
        sparse_matrix = knn.kneighbors_graph(mode="connectivity")
        adata.uns["pca_neighbors_sparse"] = sparse_matrix
        adata.obsp["pca_neighbors_sparse"] = sparse_matrix
        sc.pp.neighbors(adata, n_neighbors=15)
        sc.tl.umap(adata)
        knn_umap = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1)
        knn_umap.fit(adata.obsm["X_umap"])
        neighbor_indices_raw_umap = knn_umap.kneighbors(return_distance=False)
        # Remove self-references from indices (first column if it's self, last column if not)
        neighbor_indices_umap = np.array([
            row[1:] if row[0] == i else row[:-1]
            for i, row in enumerate(neighbor_indices_raw_umap)
        ])
        # 3. Store neighbor indices as array in obsm
        adata.obsm["umap_neighbors"] = neighbor_indices_umap
        # 4. Compute and store sparse connectivity matrix
        sparse_matrix_umap = knn_umap.kneighbors_graph(mode="connectivity")
        adata.uns["umap_neighbors_sparse"] = sparse_matrix_umap
        adata.obsp["umap_neighbors_sparse"] = sparse_matrix_umap

        print(self.oracle.adata.obsm["X_umap"].shape)
        print(self.oracle.adata.obsm["X_pca"].shape)
        
        return adata

    def save_trained_oracle(self):
        self.oracle.to_hdf5(self.dir_oracle + "/GRN_trained_.celloracle.oracle")


    def extract_index_tf(self,context_string):
        import re

        """
        Extracts indices and transcription factors (tfs) from a formatted string.

        Args:
            context_string: A single multi-line string where each line contains
                            'index: <number>' and 'with tf: <name>'.

        Returns:
            A tuple containing two lists: (list_of_indices, list_of_tfs).
            Returns ([], []) if the input string is empty or format is incorrect.
        """
        indices = []
        tfs = []

        # Regex pattern to find index and tf on each line
        # index:\s*(\d+)     -> Capture group 1: one or more digits after 'index:' and optional spaces
        # .*?                -> Non-greedily match any characters in between (handles celltype, coords)
        # with tf:\s*(\S+)   -> Capture group 2: one or more non-whitespace characters
        #                      after 'with tf:' and optional spaces
        # We assume tf is the word immediately following "with tf:" and before the next " with " or end of line
        pattern = re.compile(r"index:\s*(\d+).*?with tf:\s*(\S+)")

        lines = context_string.strip().splitlines()  # Split into lines and remove leading/trailing whitespace

        for line in lines:
            match = pattern.search(line)
            if match:
                try:
                    # Extract the captured groups
                    index_val = int(match.group(1))  # Convert index to integer
                    tf_val = match.group(2)  # TF name is a string

                    indices.append(index_val)
                    tfs.append(tf_val)
                except ValueError:
                    print(f"Warning: Could not convert index '{match.group(1)}' to int in line: {line}")
                except IndexError:
                    print(f"Warning: Regex pattern failed to capture expected groups in line: {line}")
            else:
                # Optionally print a warning for lines that don't match the pattern
                if line.strip():  # Only warn if the line wasn't just whitespace
                    print(f"Warning: Pattern not found in line: {line}")

        return indices, tfs
    

    #
    # def violin_plot(self):
    #     sc.pl.violin(self.oracle.adata, keys="Gata1", groupby=self.oracle.cluster_column_name)
    #     plt.show()