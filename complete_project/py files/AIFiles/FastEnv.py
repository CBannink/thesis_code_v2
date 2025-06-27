from enum import unique
import gymnasium as gym  # Still needed for spaces
from gymnasium import spaces
import numpy as np
import pickle
import os
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Union
from scipy.sparse import issparse  # Check if needed based on _get_current_expression_vector
import cupy as cp
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import VecEnv
import math
import time
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from sklearn.preprocessing import MinMaxScaler


###INFO:
# 1) CURRENTLY WE ONLY USE THE KNOCKOUTS, NOT THE ACTIVATIONS,
# 2) THE REWARDS ARE VERY SIMPLE ->WE IMPLEMENTED SAME CELL INDICES TRAKCING THOUGH
# 3) WE ONLY USE A 110 INPUT SPACE (THE NUMBER OF REG GENES), FOR SIMPLICITY, WE ONLY USE A REDUCED GENE SPACE OF ONLY THE REG GENES, MAYBE SCALE UP LATER
class CellOracleSB3VecEnv(VecEnv):  # Inherit from SB3 VecEnv
    metadata = {"render_modes": []}

    def __init__(self,
                 oracle_path: str, batch_size: int = 64, max_steps: int = 50,
                 step_penalty: float = -0.01, goal_bonus: float = 1.0, fail_penalty: float = -1.0,
                 distance_reward_scale: float = 1.0, same_cell_penalty: float = -0.2, gamma_distance: float = 1.0,
                 allow_gene_activation: bool = False, use_prev_perturbs: bool = False,
                 number_of_targets_curriculum: int = 4, standard_sd_factor: float = 1,
                 use_similarity: bool = False, embedding_name: str = "X_umap", transition_path: str = None, target_bounds = 0.2,
                 # Add sampling strategy args
                 initial_cell_idx: Optional[List[int]] = None,
                 target_cell_types: Optional[List[int]] = None):

        self._allow_gene_activation = allow_gene_activation  # Store for space creation
        self.max_steps = max_steps
        self.use_similarity = use_similarity
        self.wandb_run_id = None
        self.wandb_run_name = None
        self.check_path(oracle_path)  # Check path early
        self.cupy_integration_use = False  # Set to True if using CuPy
        self.number_of_episodes_started_completed = 0
        self.number_of_episodes_started_overal = 0
        self.number_of_goal_reached = 0
        self.number_of_episodes_started_periodic = 0
        self.goal_reached_periodic = 0
        self.period = 10000 #log the overal success rate
        self.step_counter = 0

        try:
            cp.cuda.Device(0).use()
            self.cupy_integration_use = True
            print("CuPy is available and GPU is accessible.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            # Handle the case where CuPy cannot access the GPU
            print("CuPy cannot access GPU, falling back to CPU integration.")
            self.cupy_integration_use = False

        with open(oracle_path, 'rb') as f:
            self.oracle = pickle.load(f)
        self.oracle.init(embedding_type=embedding_name, n_neighbors=200, torch_approach=False,
                         cupy_approach=self.cupy_integration_use, batch_size=batch_size)
        self.use_prev_perturbs = use_prev_perturbs
        #AS THIS OBVIOUSLY DOES NOT WORK WITH PREV PERTURBS, WE SET IT TO FALSE TO FORCE COMPLIANCE
        self.use_prev_perturbs = False
        self.embedding_name = embedding_name

        # Bio params
        self.n_cells = self.oracle.adata.n_obs
        self.all_genes = self.oracle.adata.var.index.tolist()
        self.genes_that_can_be_perturbed = self.oracle.return_active_reg_genes()  # used to convert the action idx to a gene name, the policy network output we receive is a single int, so we use that to index in this list
        self.number_of_reg_genes = len(self.genes_that_can_be_perturbed)
        self.celltypes = self.oracle.adata.obs[
            'celltype'].unique().tolist() if 'celltype' in self.oracle.adata.obs.columns else None
        self.high_ranges_dict = self._calculate_gene_activation_values(
            sd_factor=standard_sd_factor) if self._allow_gene_activation else None

        # important! we define the action space size, if we allow for gene act, we double the output nodes as the 2nd part of the output is for activation
        self.action_space_size = self.number_of_reg_genes
        self.action_space_size += self.number_of_reg_genes if self._allow_gene_activation else 0

        # Debugging paramas when required
        self.debug_cell_idx = initial_cell_idx if initial_cell_idx is not None else None
        self.debug_target_cell_types = target_cell_types if target_cell_types is not None else None

        observation_space = self._create_observation_space()
        action_space = self._create_action_space()

        super().__init__(num_envs=batch_size,
                         observation_space=observation_space,
                         action_space=action_space)

        # ENV trainer params etc
        self.celltype_to_one_hot = self._create_cell_type_to_hot_encoded_vec_dict(self.celltypes)
        self.reg_gene_to_full_idx = {name: i for i, name in enumerate(self.all_genes) if
                                     name in self.genes_that_can_be_perturbed}  # bit of a wierd thing, dictionary that maps a reg gene to its index within the adata, so quite similiar to the one below
        self.reg_gene_adata_indices = np.array([self.all_genes.index(g) for g in
                                                self.genes_that_can_be_perturbed])  # contains the integer indices where a reg gene resides within the adata
        self.umap_coords_normalized = self._create_normalized_embedding_coordinates()
        self.average_umap_coordinates,self.average_normalized_umap_coordinates = self.calculate_centre_umap_coords_target()  # Compute average UMAP coordinates for each cell type
        self.total_curriculum_targets = self._compute_total_curriculum_targets(
            number_of_targets_curriculum)  # Compute the total curriculum targets
        self.knockout_histories = np.zeros((batch_size, self.action_space_size))
        self.current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_start_times = np.array([time.time()] * self.num_envs, dtype=np.float32)
        self.total_goals_reached = 0
        # Reward params
        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.fail_penalty = fail_penalty
        self.distance_reward_scale = distance_reward_scale
        self.same_cell_penalty = same_cell_penalty
        self.gamma = gamma_distance

        # State params
        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_cell_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.current_target_cell_types = np.full(self.num_envs, None, dtype=object)
        self.current_phase = 1
        print("Number of celltypes: ", len(self.celltypes), " so without pcgs we have ", len(self.celltypes) - 3,
              " targets")
        # init the indices we can start from, only pgc version of cells
        cell_types_series = self.oracle.adata.obs['celltype']
        # we allow for all 3 pgcs as those are the closest after activating mescs
        pgc_mask = cell_types_series.str.contains("PGCs", na=False)
        pgc_positional_indices = np.where(pgc_mask)[0]
        pgc_positional_indices = pgc_positional_indices.astype(np.int32)
        self.temp_indices_to_choose_from = pgc_positional_indices
        # check if all celtypes fo these indices contain pgcs
        if not np.all(cell_types_series.iloc[pgc_positional_indices].str.contains("PGCs", na=False)):
            print("not gone right")

        self.transition_matrix, self.action_map  = self.load_transition_data(transition_matrix_path=transition_path)
        self.target_bounds = target_bounds
        self._actions: Optional[np.ndarray] = None
        self.plot_umap_with_goal_regions()
        # self.test_if_trans_matrix_gives_same_result_as_cellOc()

    def test_if_trans_matrix_gives_same_result_as_cellOc(self):
        #set seed
        np.random.seed(42)
        cell_indices_to_perturb_at_random = [1000,2500,434,5,12,19,29545, 1234, 2345, 3456]
        #choose randoma ctions to perform
        actions_to_perform = np.random.randint(0, self.action_space_size, size=len(cell_indices_to_perturb_at_random))
        perturb_conditions = []
        for i, action_idx in enumerate(actions_to_perform):
            if action_idx < self.number_of_reg_genes:
                gene_to_knockout = self.genes_that_can_be_perturbed[action_idx]
                perturb_conditions.append((gene_to_knockout, 0.0))
            else:
                gene_to_activate = self.genes_that_can_be_perturbed[action_idx % self.number_of_reg_genes]
                activation_value = self.high_ranges_dict[gene_to_activate]
                perturb_conditions.append((gene_to_activate, activation_value))


        # Use the transition matrix to get the new indices
        new_indices = []
        for cell_idx, action_idx in zip(cell_indices_to_perturb_at_random, actions_to_perform):
            new_indices.append(self.transition_matrix[cell_idx, action_idx])

        new_indices = np.array(new_indices, dtype=np.int32)
        # Now use the oracle to get the new indices
        _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch_cp(
            batch_size=self.num_envs, idxs=cell_indices_to_perturb_at_random,
            perturb_condition=perturb_conditions, n_neighbors=200,
            n_propagation=3, threads=4,
            knockout_prev_used=False
        )
        new_indices_oracle = np.array(new_idx_list, dtype=np.int32)
        # Check if the new indices are the same
        if np.array_equal(new_indices, new_indices_oracle):
            print("Transition matrix gives the same result as CellOracle.")
        else:
            print("Transition matrix and CellOracle give different results.")
            print("Transition matrix indices: ", new_indices)
            print("CellOracle indices: ", new_indices_oracle)
            print("Difference: ", np.sum(new_indices != new_indices_oracle))
            #print celltypes
            celltypes_transition = self.oracle.adata.obs['celltype'].iloc[new_indices].tolist()
            celltypes_oracle = self.oracle.adata.obs['celltype'].iloc[new_indices_oracle].tolist()
            print("Celltypes from transition matrix: ", celltypes_transition)
            print("Celltypes from CellOracle: ", celltypes_oracle)

        #another test
        cell_indices_to_perturb_at_random = [0,0,0,0,0,0,0,0,0,0]
        actions_to_perform = [0,1,2,3,4,5,6,7,8,9]
        perturb_conditions = []
        for i, action_idx in enumerate(actions_to_perform):
            if action_idx < self.number_of_reg_genes:
                gene_to_knockout = self.genes_that_can_be_perturbed[action_idx]
                perturb_conditions.append((gene_to_knockout, 0.0))

        cell_trans = []
        for cell_idx, action_idx in zip(cell_indices_to_perturb_at_random, actions_to_perform):
            result = self.transition_matrix[cell_idx, action_idx]
            cell_trans.append(result)

        _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch_cp(
            batch_size=self.num_envs, idxs=cell_indices_to_perturb_at_random,
            perturb_condition=perturb_conditions, n_neighbors=200,
            n_propagation=3, threads=4,
            knockout_prev_used=False
        )

        new_indices_oracle = np.array(new_idx_list, dtype=np.int32)
        # Check if the new indices are the same
        if np.array_equal(cell_trans, new_indices_oracle):
            print("Transition matrix gives the same result as CellOracle for the second test.")






    def _create_normalized_embedding_coordinates(self) -> np.ndarray:
        umap = self.oracle.adata.obsm[self.embedding_name].copy()
        # normalize the coords
        scaler = MinMaxScaler(feature_range=(-1, 1))
        umap = scaler.fit_transform(umap)
        print("UMAP coordinates normalized with StandardScaler with amx: " , np.max(umap), " and min: ", np.min(umap))
        return umap

    def set_wandb_id(self, wandb_run_id: str, wandb_run_name: str):
        """Set the wandb run id."""
        self.wandb_run_id = wandb_run_id
        self.wandb_run_name = wandb_run_name

    def _create_cell_type_to_hot_encoded_vec_dict(self, celltypes: List[str]) -> dict:
        """Create a mapping from cell type names to integers."""
        cell_type_to_hot_vec = {}
        # Declare this list in the order it is given in the paper, thus linking up the ints to the correct ones
        cell_type_string = ["Primitive Streak", "Caudal Epiblast", "Epiblast", "Naive PGCs", "Epidermis Progenitors",
                            "Caudal Mesoderm", "Parietal Endoderm", "PGCs", "(pre)Somitic/Wavefront",
                            "Nascent Mesoderm", "NMPs", "LP/Intermediate Mesoderm", "Neural Progenitors",
                            "(early) Somite", "Cardiac Mesoderm", "Endothelium", "Visceral Endoderm", "Dermomyotome",
                            "Erythrocytes", "Reprogramming PGCs", "Sclerotome", "Roof Plate Neural Tube",
                            "Floor Plate Neural Tube", "ExE Endoderm", "Cardiomyocytes", "Pharyngeal Mesoderm",
                            "Early Motor Neurons", "Late Motor Neurons", "Myotome", "Megakaryocytes"]
        one_hot_length = len(celltypes)
        for i, celltype in enumerate(cell_type_string):
            if celltype in celltypes:
                one_hot_vec = np.zeros(one_hot_length, dtype=np.float32)
                one_hot_vec[i] = 1.0
                cell_type_to_hot_vec[celltype] = one_hot_vec
            else:
                print(
                    "Warning: Cell type not found in predefined list. We might have excluded some data which causes this! , celltype: ",
                    celltype)
                raise ValueError(f"Cell type '{celltype}' not found in predefined list.")

        return cell_type_to_hot_vec

    def _create_action_space(self) -> spaces.Space:
        return spaces.Discrete(self.action_space_size)

    def _create_observation_space(self) -> spaces.Dict:
        # Uses self._number_of_reg_genes, self._max_steps
        low_bound, high_bound = -np.inf, np.inf
        obs_space_size = self.number_of_reg_genes + 2 + len(self.celltypes) + 2 + self.action_space_size
        return spaces.Dict({
            "current_state": spaces.Box(low=-1, high=1, shape=(self.number_of_reg_genes,),
                                        dtype=np.float32),
            "current_umap_coordinates": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "target_cell_type": spaces.Box(low=0.0, high=1.0, shape=(len(self.celltypes),), dtype=np.float32),
            "target_umap_coordinates": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })

    def reset(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(
            self._np_random_seed if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)

        self.current_cell_indices = rng.choice(self.temp_indices_to_choose_from, size=self.num_envs, replace=False)

        # now sample target celltypes
        self.current_target_cell_types = self._get_target_cell_types(self.current_cell_indices)
        # for debug purposses
        if self.debug_cell_idx is not None:
            self.current_cell_indices = np.array(self.debug_cell_idx)
        if self.debug_target_cell_types is not None:
            for i in range(self.num_envs):
                self.current_target_cell_types[i] = self.debug_target_cell_types[i]

        self.oracle.reset_info_during_training_for_batch_instance(np.arange(self.num_envs))
        self.current_steps.fill(0)
        self.current_episode_rewards.fill(0.0)
        self.current_episode_lengths.fill(0)
        self.current_episode_start_times[:] = time.time()
        self.knockout_histories.fill(0.0)
        self.total_goals_reached = 0
        self.number_of_episodes_started_completed = 0
        self.number_of_episodes_started_overal = 0
        self.number_of_goal_reached = 0
        self._actions = None  # Reset actions
        self.number_of_episodes_started_periodic = 0
        self.step_counter = 0
        self.goal_reached_periodic = 0
        
        

        return self._get_obs()

    def reset_for_inference(self, start_idx: int, target_name: str) -> Dict[str, np.ndarray]:
        """
        Manually resets the (single) environment to a specific start/target for inference.
        Assumes num_envs = 1.
        """
        if self.num_envs > 1:
            print("Warning: reset_for_inference is designed for a single environment (num_envs=1).")

        # Reset common state
        self.current_steps.fill(0)
        self.current_episode_rewards.fill(0.0)
        self.current_episode_lengths.fill(0)
        self.current_episode_start_times[:] = time.time()
        if hasattr(self, 'knockout_histories'):
            self.knockout_histories.fill(0.0)
        self.oracle.reset_info_during_training_for_batch_instance(np.arange(self.num_envs))
        self.total_goals_reached = 0
        self.number_of_episodes_started_completed = 0
        self.number_of_episodes_started_overal = 0
        self.number_of_goal_reached = 0
        self._actions = None  # Reset actions


        # Set the specific state
        self.current_cell_indices[0] = start_idx
        self.current_target_cell_types[0] = target_name

        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        """Stores the actions."""
        self._actions = actions

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        """Performs the simulation and returns results (synchronous execution).
            FOR NOW:
            1) WE DO NOT INTEGRATE A POLICY MASK (ACTIONS THAT ARE NOT POSSIBLE)
            2) WE DO INTEGRATE A FAILURE PENALTY


        """
        if self._actions is None:
            raise RuntimeError("Cannot call step_wait without calling step_async first.")

        actions = self._actions.copy()  # Copy actions to avoid modifying the original
        self._actions = None
        perturb_conditions = []
        action_indices = actions.flatten().astype(int)
        total_activations_per_step = 0
        action_idx_list_for_trans_matrix = actions.flatten().astype(int)  # Flattened action indices for transition matrix
        for i in range(self.num_envs):
            action_idx = action_indices[i]
            self.knockout_histories[i, action_idx] = 1.0
            # Basic check for action validity (important if not using MaskablePPO perfectly)
            if action_idx < self.number_of_reg_genes:
                # does not matter if we allow gene activation or not, we just set the value to 0.0
                gene_to_knockout = self.genes_that_can_be_perturbed[action_idx]
                perturb_conditions.append((gene_to_knockout, 0.0))  # Assuming knockout value is always 0.0
                continue
            if self._allow_gene_activation:
                if action_idx > 2 * self.number_of_reg_genes:
                    print("WARNING THIS SHOULD NOT HAPPEN MORE ACTIONS THAN 2*REG GENES")
                    raise ValueError("Action index out of bounds for gene activation.")
                gene_to_activate = self.genes_that_can_be_perturbed[action_idx % self.number_of_reg_genes]
                # Get the activation value from the high ranges dict
                activation_value = self.high_ranges_dict[gene_to_activate]
                perturb_conditions.append((gene_to_activate, activation_value))
                total_activations_per_step += 1

        n_neighbors_sim, n_propagation_sim, threads_sim = 200, 3, 4
        simulation_success = True
        try:
            # sgtart timer to keep track of how long it takes
            new_idx_list =  self.transition_matrix[self.current_cell_indices, action_idx_list_for_trans_matrix]
            new_cell_indices = np.array(new_idx_list, dtype=np.int32)  # Ensure correct type
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print the full traceback
            print('WATCH OUT THIS IS WRONG')
            print(e)
            simulation_success = False
            new_cell_indices = self.current_cell_indices

        # init all necessary info
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)  # For info dict

        if not simulation_success:
            print("THIS IS A PROBLEM IF THIS HAPPENS")
            # just terminate the entire episode if it fails, this indicates a problem
            rewards.fill(self.fail_penalty)
            dones.fill(True)
            truncations.fill(False)
            return self._get_obs(), rewards, dones, self._create_info_dict(
                goal_reached=np.zeros(self.num_envs, dtype=bool), truncated=np.zeros(self.num_envs, dtype=bool),
                simulation_failed=np.ones(self.num_envs, dtype=bool),
                same_cell_indices_as_before_mask=np.zeros(self.num_envs, dtype=bool),
                done_episode_rewards=self.current_episode_rewards, done_episode_lengths=self.current_episode_lengths,
                done_episode_start_times=self.current_episode_start_times)

        # The chance of this happening above is slim but we just implement it to be sure and catch if it goes wrong, can save alot of pain debugging DO NOT REMOVE LATER !

        # first set the neew info correctly
        same_cell_indices_as_before_mask = new_cell_indices == self.current_cell_indices
        rewards_distance_phi = self._reward_system_distance_calc(
            new_cell_indices)  # DO NOT CALL AFTER SETTING THE CURRENT CELL INDICES
        self.current_cell_indices = new_cell_indices
        self.current_steps += 1

        celltypes_of_new_indices = self.oracle.adata[new_cell_indices].obs['celltype'].to_numpy()
        #print the type of both:
 

        
        # TODO NAAR DEZE STATEMENT KIJKEN
        goal_reached_mask = self.which_cell_have_reached_target(new_cell_indices)  # Check if the new cell indices have reached the target cell types
        self.total_goals_reached += np.sum(goal_reached_mask)
        self.goal_reached_periodic += np.sum(goal_reached_mask)

        is_timeout_mask = self.current_steps >= self.max_steps  # yes this is an actual mask that is created DO NOT MESS WITH IT LATER
        # set to default step penalty
        rewards.fill(self.step_penalty)
        rewards += rewards_distance_phi  # add the distance reward
        # then add a possible time out penalty
        rewards[is_timeout_mask] += self.fail_penalty
        # but if goal reached then add the goal reached bonus
        rewards[goal_reached_mask] += self.goal_bonus
        # add additgional if same cell is returned, very bad perturbation
        rewards[same_cell_indices_as_before_mask] += self.same_cell_penalty

        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1
        dones[is_timeout_mask] = True
        dones[goal_reached_mask] = True
        done_and_goal = goal_reached_mask & dones
        self.number_of_goal_reached += np.sum(done_and_goal)
        # same for here, first do timeout, then overwrite with goal reached
        truncations[is_timeout_mask] = True
        truncations[goal_reached_mask] = False

        # so this is for logging purposes, we will calc averages and log then
        fail_pen = np.zeros(self.num_envs, dtype=np.float32)
        fail_pen[is_timeout_mask] += self.fail_penalty
        average_pen = np.mean(fail_pen)
        goal_reach = np.zeros(self.num_envs, dtype=np.float32)
        goal_reach[goal_reached_mask] += self.goal_bonus
        average_goal = np.mean(goal_reach)
        av_same_cell_indices_as_before_mask = np.zeros(self.num_envs, dtype=np.float32)
        av_same_cell_indices_as_before_mask[same_cell_indices_as_before_mask] += self.same_cell_penalty
        average_same_cell = np.mean(av_same_cell_indices_as_before_mask)
        average_distance = np.mean(rewards_distance_phi)

        #


        done_episode_rewards_for_info = self.current_episode_rewards.copy()
        done_episode_lengths_for_info = self.current_episode_lengths.copy()
        done_episode_start_times_for_info = self.current_episode_start_times.copy()

        obs_before_reset = self._get_obs()  # Get the observation before resetting

        info_dict = self._create_info_dict(goal_reached=goal_reached_mask, truncated=truncations,
                                           simulation_failed=np.zeros(self.num_envs, dtype=bool),
                                           same_cell_indices_as_before_mask=same_cell_indices_as_before_mask,
                                           obs_before_reset=obs_before_reset, average_goal=average_goal,
                                           average_penalty=average_pen, average_distance=average_distance,
                                           average_same_cell=average_same_cell,
                                           percentage_of_activation=total_activations_per_step / self.num_envs,
                                           current_cell_types=celltypes_of_new_indices,
                                           target_cell_types=self.current_target_cell_types,
                                           done_episode_rewards=done_episode_rewards_for_info,
                                           done_episode_lengths=done_episode_lengths_for_info,
                                           done_episode_start_times=done_episode_start_times_for_info)

        self._handle_batch_resets(dones)  # Handle individual resets

        obs_after_reset = self._get_obs()  # Get the observation after resetting, this is given back in this step
        return obs_after_reset, rewards, dones, info_dict

    def which_cell_have_reached_target(self, next_cell_indices:np.ndarray):

        has_same_celltype_mask = self.oracle.adata[next_cell_indices].obs['celltype'].to_numpy() == self.current_target_cell_types
        #now c heck if it is in the correct bounds
        average_umap_coords_target = np.array([self.average_umap_coordinates[t] for t in self.current_target_cell_types])
        distances_current_cells_to_target = np.linalg.norm(self.oracle.adata.obsm[self.embedding_name][next_cell_indices] - average_umap_coords_target, axis=1)
        within_bounds_mask = distances_current_cells_to_target <= self.target_bounds
        # print("so many in boudns: ",np.sum(within_bounds_mask) , " with this any correct celltype: ", + np.sum(has_same_celltype_mask), " and total: ", np.sum(within_bounds_mask & has_same_celltype_mask))
        # Combine both masks
        return has_same_celltype_mask & within_bounds_mask


    def _handle_batch_resets(self, dones: np.ndarray) -> None:
        """
          I NEED TO HANDLE OWN RESET BECAUSE OF THIS ENVIORNMENT
        handles batch instances reset!
        """
        reset_indices = np.where(dones)[0]
        num_to_reset = len(reset_indices)

        if num_to_reset <= 0:
            return
        # Reset the indices and steps for the reset instances do +current steps to not always get the same start indices..
        reset_rng = np.random.default_rng(self._np_random_seed + self.current_steps.sum() if hasattr(self,
                                                                                                     "_np_random_seed") and self._np_random_seed is not None else None)

        new_start_indices = reset_rng.choice(self.temp_indices_to_choose_from, size=num_to_reset, replace=False)

        new_target_types_arr = self._get_target_cell_types(new_start_indices, reset_rng)

        self.current_cell_indices[reset_indices] = new_start_indices
        # Ensure target types array is compatible (e.g., object type)
        self.current_target_cell_types[reset_indices] = new_target_types_arr
        self.current_steps[reset_indices] = 0
        self.current_episode_rewards[reset_indices] = 0.0
        self.current_episode_lengths[reset_indices] = 0
        self.current_episode_start_times[reset_indices] = time.time()
        self.knockout_histories[reset_indices, :] = 0.0  # Reset knockout history for these envs
        # reset the oracle info for the batch instance
        self.oracle.reset_info_during_training_for_batch_instance(reset_indices)
        self.number_of_episodes_started_completed += num_to_reset
        self.number_of_episodes_started_overal += num_to_reset
        self.number_of_episodes_started_periodic += num_to_reset
        self.step_counter +=1
        if self.step_counter >= self.period:
            self.number_of_episodes_started_periodic = 0
            self.step_counter = 0
            self.goal_reached_periodic = self.goal_reached_periodic


    def _reward_system_distance_calc(self, next_indices_after_perturb: np.ndarray) -> np.ndarray:
        """
        Implement the reward system here (if sim succesfull, otherwise its handled in stepwwait)
        REWARD SYSTEM: R = R_STEP + R_GOAL + R_FAIL + R_DIST_SHAPE(PHI)
        Returns: The reward array for the current step.
        CALL THIS FUNCTION BEFORE SETTING CURRENT INDICES TO NEXT OTHERWISE BAD THINGS HAPPEN IT WONT WORK!

        """
        if np.array_equal(self.current_cell_indices, next_indices_after_perturb):
            raise ValueError(
                "Current cell indices are the same as next indices after perturbation. This should not happen.")
        # Calculate distances to target cell types
        distances_next = self._calculate_umap_distances_to_target(next_indices_after_perturb)
        distances_before = self._calculate_umap_distances_to_target(self.current_cell_indices)
        if self.use_similarity:
            epsilon = 1e-8  # avoid division zero
            potential_next = 1.0 / (1.0 + distances_next + epsilon)
            potential_before = 1.0 / (1.0 + distances_before + epsilon)
            # PBRS: F(s, s') = γ * Φ(s') - Φ(s)
            phi_reward = self.gamma * potential_next - potential_before
            phi_reward *= self.distance_reward_scale
            return phi_reward
        else:
            costs_change = self.gamma * distances_next - distances_before
            phi_reward = -costs_change
            # scale if necessary, for now scale is 1 to just test out
            phi_reward *= self.distance_reward_scale
            return phi_reward

    def _calculate_umap_distances_to_target(self, cell_indices: np.ndarray) -> np.ndarray:
        """
        Calc distance betweeen current cell indice and the average umap coordinates of the target cell type

        Args:
            cell_indices: A numpy array of numerical indices for the cells.

        Returns:
            A numpy array of distances (float32), one for each input cell index.
        """

        current_umap_coords = self.oracle.adata.obsm[self.embedding_name][
            cell_indices]  # Get UMAP coordinates for the current cells
        diff = np.zeros_like(current_umap_coords)  # Initialize diff for later use
        for i in range(self.num_envs):
            target_type_name = self.current_target_cell_types[i]
            average_coord_celltype = self.average_umap_coordinates.get(target_type_name, None)
            if average_coord_celltype is None:
                raise ValueError(f"Average UMAP coordinates for target cell type '{target_type_name}' not found.")
            diff[i] = current_umap_coords[i] - average_coord_celltype

        distances = np.linalg.norm(diff, axis=1)  # Calculate distances
        return np.maximum(distances, 0.0).astype(np.float32)  # Ensure no negative distances

    def _create_info_dict(self, goal_reached: np.ndarray, truncated: np.ndarray, simulation_failed: np.ndarray,
                          same_cell_indices_as_before_mask: np.ndarray,
                          obs_before_reset: Dict[str, np.ndarray],
                          done_episode_rewards: np.ndarray,
                          done_episode_lengths: np.ndarray,
                          done_episode_start_times: np.ndarray,
                          percentage_of_activation: float = None,
                          average_goal: float = None,
                          average_penalty: float = None, average_distance: float = None,
                          average_same_cell: float = None, current_cell_types: np.ndarray = None,
                          target_cell_types: np.ndarray = None) -> List[Dict[str, Any]]:
        """Creates the info dictionary list."""
        infos = []
        is_done = goal_reached | truncated | simulation_failed
        # print how many instnances are done
        unique_cell_types = np.unique(current_cell_types) if current_cell_types is not None else None
        unique_target_cell_types = np.unique(target_cell_types) if target_cell_types is not None else None

        for i in range(self.num_envs):
            info_dict_per_env = {}
            info_dict_per_env["TimeLimit.truncated"] = truncated[i]
            if average_goal is not None:
                info_dict_per_env["step_avg/average_goal"] = average_goal
            if average_penalty is not None:
                info_dict_per_env["step_avg/average_penalty"] = average_penalty
            if average_distance is not None:
                info_dict_per_env["step_avg/average_distance"] = average_distance
            if average_same_cell is not None:
                info_dict_per_env["step_avg/average_same_cell"] = average_same_cell
                info_dict_per_env["diagnostics/same_cell_event_this_step"] = bool(same_cell_indices_as_before_mask[i])
            if current_cell_types is not None:
                info_dict_per_env["batch_diversity/current_cell_types_unique_in_batch"] = unique_cell_types
            if target_cell_types is not None:
                info_dict_per_env["batch_diversity/target_cell_types_unique_in_batch"] = unique_target_cell_types
            if percentage_of_activation is not None:
                info_dict_per_env["step_avg/percentage_of_activation"] = percentage_of_activation
            if is_done[i]:
                final_info_payload = {
                    "steps": self.max_steps if truncated[i] else self.current_steps[i],
                    "goal_reached": goal_reached[i],
                    "truncated": truncated[i],
                    "simulation_failed": simulation_failed[i],
                    "episode": {  # Standard Monitor keys
                        "r": done_episode_rewards[i],
                        "l": done_episode_lengths[i],
                        "t": time.time() - done_episode_start_times[i]
                    },
                }
                #for optuna bcause lazy
                info_dict_per_env["succes"] = goal_reached[i]
                info_dict_per_env["goal_reached_periodic"] = self.goal_reached_periodic/ self.number_of_episodes_started_periodic


                info_dict_per_env["terminal_observation"] = {k: v[i] for k, v in obs_before_reset.items()}
                # this looks weird to do double, but it messed with something and this seemed to fix it so no reomove later! its already integrated
                info_dict_per_env["episode"] = {
                    "r": float(done_episode_rewards[i]),
                    "l": int(done_episode_lengths[i]),
                    "t": float(time.time() - done_episode_start_times[i])
                }
                # do not remove messes it up!
                info_dict_per_env["final_info_payload"] = final_info_payload
            infos.append(info_dict_per_env)
        return infos

    def close(self) -> None:
        """Closes the environment."""
        print("Closing environment.")
        pass

    def get_attr(self, attr_name: str, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[Any]:
        # I GIVE UP, SOME FAULTY BUG IN MASKSABLEPPO MEESES EVYERHGING UP
        """Return attribute from vectorized environment."""
        print("get attribute called with name: ", attr_name)
        print("get attribute called with indices: ", indices)
        target_envs = self._get_target_envs(indices)
        if attr_name == "action_masks":
            print("this is called in attribute")
            full_action_masks = self.action_masks(True)
            print("full action masks: ", full_action_masks.shape)
            listreturn = []
            for i in range(self.num_envs):
                listreturn.append(full_action_masks[i])
            print("list return: ", len(listreturn), "shape of one: ", listreturn[0].shape)
            return listreturn
        return [getattr(self, attr_name) for _ in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> None:
        """Set attribute inside vectorized environment."""
        target_envs = self._get_target_envs(indices)
        # do not dare to mess with it just not iportant but do not  remove
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        else:
            # Check if it should be set on self.oracle?
            if hasattr(self.oracle, attr_name):
                print(f"Warning: Setting attribute '{attr_name}' on self.oracle.")
                setattr(self.oracle, attr_name, value)
            else:
                raise AttributeError(f"Attribute '{attr_name}' not found on environment or oracle.")

    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, List[int], np.ndarray]] = None,
                   **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        #do not edelete
        target_envs = self._get_target_envs(indices)
        if hasattr(self.oracle, method_name):
            method = getattr(self.oracle, method_name)
            result = method(*method_args, **method_kwargs)
            return [result for _ in target_envs]
        else:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                result = method(*method_args, **method_kwargs)
                return [result for _ in target_envs]  # check this later
            else:
                raise AttributeError(f"Method '{method_name}' not found on environment or oracle.")

    def env_is_wrapped(self, wrapper_class, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[bool]:
        """Check if environment is wrapped with a given wrapper."""
        target_envs = self._get_target_envs(indices)
        #not necessary anymore i think but preiovus version did need it but just put it in
        return [False for _ in target_envs]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seeds the random number generator of the vector environment."""
        self._np_random_seed = seed
        seeds = []
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        return seeds

    def _get_target_envs(self, indices: Optional[Union[int, List[int], np.ndarray]]) -> List[int]:
        """Helper to get target environment indices."""
        if indices is None:
            return list(range(self.num_envs))
        elif isinstance(indices, int):
            return [indices]
        elif isinstance(indices, (list, tuple, np.ndarray)):
            return list(indices)
        else:
            raise ValueError("Indices must be None, int, list, tuple, or np.ndarray")

    def check_path(self, oracle_path: str):
        if not oracle_path.endswith('.pkl'): raise ValueError('Pickle file needed')
        if not os.path.exists(oracle_path): raise ValueError(f'Path does not exist: {oracle_path}')

    def _get_current_expression_vector(self, cell_indices: np.ndarray) -> np.ndarray:
        try:
            valid_indices = np.array(cell_indices, dtype=int)
            valid_indices = valid_indices[valid_indices < self.oracle.adata.n_obs]  # Basic bounds check
            if len(valid_indices) != len(cell_indices):
                print(f"Warning: Some cell indices are out of bounds. Using only valid ones.")
                # How to handle? Return zeros for invalid? For now, proceed with valid. This indicates bigger issue.
                cell_indices = valid_indices
                if len(cell_indices) == 0: return np.zeros((0, self.number_of_reg_genes), dtype=np.float32)

            data = self.oracle.get_AI_input_for_cell_indices(cell_indices, self.reg_gene_adata_indices)
            if issparse(data):
                return data.toarray().astype(np.float32)
            else:
                # Handle potential views
                return np.array(data, dtype=np.float32) if not data.flags.writeable else data.astype(np.float32)

        except IndexError as e:
            print(f"IndexError fetching expression vector for indices {cell_indices}: {e}")
            # Return zeros matching expected batch size
            return np.zeros((len(cell_indices), self.number_of_reg_genes), dtype=np.float32)
        except Exception as e:
            print(f"Error fetching expression vector: {e}")
            return np.zeros((len(cell_indices), self.number_of_reg_genes), dtype=np.float32)

    def _get_target_state_vectors(self) -> np.ndarray:
        if self.current_target_cell_types is None:
            return np.zeros((self.num_envs, len(self.celltypes)), dtype=np.float32)
        target_state_vectors = np.zeros((self.num_envs, len(self.celltypes)), dtype=np.float32)
        for i in range(self.num_envs):
            target_state_vectors[i] = self.celltype_to_one_hot[self.current_target_cell_types[i]]
        return target_state_vectors

    def _get_current_and_target_embedding_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        current_umap_coords = self.umap_coords_normalized[self.current_cell_indices]
        target_umap_coords = np.zeros(((self.num_envs, 2)), dtype=np.float32)
        for index, target_celltype in enumerate(self.current_target_cell_types):
            # Get the average UMAP coordinates for the target cell type
            target_umap_coords[index] = self.average_normalized_umap_coordinates.get(target_celltype, np.zeros(2))
        return current_umap_coords, target_umap_coords

    def _get_target_cell_types(self, cell_indices: np.ndarray, rng_gen=None) -> List[str]:
        # Assuming cell_indices are valid and in the correct range
        current_cell_types = self.oracle.adata[cell_indices].obs['celltype']
        target_cell_types = []
        target_rng = None
        if rng_gen is None:
            target_rng = np.random.default_rng(
                self._np_random_seed if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)
        else:
            target_rng = rng_gen
        for i in range(len(current_cell_types)):
            phase_target = self.total_curriculum_targets[self.current_phase - 1]  # -1 because phase starts at
            target_types = phase_target[current_cell_types[i]]
            if target_types is None or len(target_types) == 0:
                raise ValueError("Target cell types not implemented yet")
            # pick a random target, the target does not include its own celltype
            target_cell_types.append(target_rng.choice(target_types))
        return np.array(target_cell_types, dtype=object)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        current_state_vecs = self._get_current_expression_vector(self.current_cell_indices)
        steps_left = (self.max_steps - self.current_steps).astype(np.float32).reshape(-1, 1)
        target_state_one_hot_vectors = self._get_target_state_vectors()
        current_umap_coordinates, target_umap_coordinates = self._get_current_and_target_embedding_coordinates()

        # Ensure shapes match observation space even if errors occurred, do not know why this would go wrong, get rid of this, just raide value error
        # if current_state_vecs.shape[0] != self.num_envs: current_state_vecs = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=np.float32)
        # if steps_left.shape[0] != self.num_envs: steps_left = np.zeros((self.num_envs, 1), dtype=np.float32)
        # if self.knockout_histories.shape[0] != self.num_envs: self.knockout_histories = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=bool)

        if current_state_vecs.shape[0] != self.num_envs:
            raise ValueError(f"Current state vectors shape mismatch: {current_state_vecs.shape} vs {self.num_envs}")
        if steps_left.shape[0] != self.num_envs:
            raise ValueError(f"Steps left shape mismatch: {steps_left.shape} vs {self.num_envs}")
        #print all max and min values of the input
        # print("Max value of current state vectors:", np.max(current_state_vecs), "Min value of current state vectors:", np.min(current_state_vecs))
        # print("Max value of target state one hot vectors:", np.max(target_state_one_hot_vectors), "Min value of target state one hot vectors:", np.min(target_state_one_hot_vectors))
        # print("Max value of current UMAP coordinates:", np.max(current_umap_coordinates), "Min value of current UMAP coordinates:", np.min(current_umap_coordinates))
        # print("Max value of target UMAP coordinates:", np.max(target_umap_coordinates), "Min value of target UMAP coordinates:", np.min(target_umap_coordinates))


        return {
            "current_state": current_state_vecs,
            "current_umap_coordinates": current_umap_coordinates,
            "target_cell_type": target_state_one_hot_vectors,
            "target_umap_coordinates": target_umap_coordinates,
            # "steps_left": steps_left
            # "knockout_history": self.knockout_histories.astype(np.int8)
        }

    # def _compute_average_umap_coordinates(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    #     """
    #     NOT USED ANYMMORE BUT JUST LEAVE IT IT FOR NOW BUT NOT USED AS WE NOW FILTER ON OUTLIERS IN ANOTHER NEW METHOD!
    #     Compute the average UMAP coordinates for any celltype-> used for distance calculation (and maybe nearest celltype neighbors?)
    #     """
    #     average_coordinates = {}
    #     average_coordinates_norm = {}
    #     for celltype in self.celltypes:
    #         boolean_mask_celltype = (self.oracle.adata.obs['celltype'] == celltype).values
    #         umap_cluster_coords = self.oracle.adata.obsm[self.embedding_name][boolean_mask_celltype]
    #         umap_normalized_coords = self.umap_coords_normalized[boolean_mask_celltype]
    #         average_coordinates[celltype] = np.mean(umap_cluster_coords, axis=0)
    #         average_coordinates_norm[celltype] = np.mean(umap_normalized_coords, axis=0)
    #     return average_coordinates, average_coordinates_norm

    def _compute_total_curriculum_targets(self, number_of_targets, start_with_pgcs=True) -> List:
        """init the total target curriculum using the _compute_nearest_celltype_targets func """
        if not start_with_pgcs:
            if self.average_umap_coordinates is None:
                raise ValueError("Average UMAP coordinates not computed. Call _compute_average_umap_coordinates first.")
            number_of_phases = (len(self.celltypes) // number_of_targets) + 1
            total_curriculum_targets = []
            for phase in range(1, number_of_phases + 1):
                phase_targets = self._compute_nearest_celltype_targets(phase, number_of_targets)
                total_curriculum_targets.append(phase_targets)
            return total_curriculum_targets

        number_of_phases = math.ceil(len(self.celltypes) / number_of_targets) + 1
        print("Number of phases for curriculum learning:", number_of_phases)
        total_curriculum_targets = []
        print("number of targets:", number_of_targets)
        for phase in range(1, number_of_phases + 1):
            phase_targets = self._compute_nearest_celltype_targets_only_pgcs(phase, number_of_targets)
            print(len(phase_targets), "targets for phase", phase)
            total_curriculum_targets.append(phase_targets)
        print("Total curriculum targets computed for phases:", len(total_curriculum_targets))
        print("Total curriculum targets for each phase:", [len(t) for t in total_curriculum_targets])
        print("Total number of celltypes in curriculum:", [len(t) for t in total_curriculum_targets[0].values()])


        return total_curriculum_targets

    def _compute_nearest_celltype_targets(self, phase: int, number_of_targets: int) -> dict:
        """
        THIS FUNCTION DOES NOT INCLUDE ITSELF IN TARGET!
        Compute the nearest celltype targets for all celltypes, this is used for curriculum learning
        Based on phase, this function spits out a dict of celltypes that are direct neighbors, or 2nd neighbors etc dependent on phase
        It returns a list of dictionary, with keys -> celltype, value -> list of possible target celltypes
        Variables: phase indicates how many different celltypes we allow for, like phase =1 -> equal to the first phase of training, phase =2 -> equal to the second phase of training, etc
        In the end, phase =1 allows for number_of_targets celltypes, phase 2 allows for 2*number_of_targets celltypes, etc
        NOT SURE IF WE WANT TO IMPLEMENT SLICING IN A WAY THAT THE CELLTYPES FROM THE PREVIOUS CELLTYPES ARE NOT INCLUDED AS THE FIRST PHASE CELLTYPES WOULD THEN BE VERY HEAVILY TARGETTED AND SEEN WAY MORE?
        """
        # Get the UMAP coordinates for the clusters
        celltype_dict = {}
        for celltype in self.celltypes:
            average_coord_celltype = self.average_umap_coordinates[celltype]
            distances = []
            for other_celltype in self.celltypes:
                if celltype != other_celltype:
                    average_coord_other_celltype = self.average_umap_coordinates[other_celltype]
                    distance = np.linalg.norm(average_coord_celltype - average_coord_other_celltype)
                    distances.append((other_celltype, distance))

            distances.sort(key=lambda x: x[1])
            # now choose the clostest one according to phase
            # get rid of the current celltype as this is not relevant
            closest_celltypes = []
            for i in range(phase * number_of_targets):
                if i < len(distances):
                    closest_celltypes.append(distances[i][0])
            celltype_dict[celltype] = closest_celltypes
        return celltype_dict

    def _compute_nearest_celltype_targets_only_pgcs(self, phase: int, number_of_targets: int) -> dict:
        # retrieve the celltypes with pgc
        celltype_dict = {}
        pgc_celltypes = [celltype for celltype in self.celltypes if 'PGC' in celltype]
        for celltype in pgc_celltypes:
            average_coord_celltype = self.average_umap_coordinates[celltype]
            distances = []
            for other_celltype in self.celltypes:
                if other_celltype in pgc_celltypes:
                    continue
                if celltype == other_celltype:  # this should be handled by pervius but just to be sure
                    continue
                average_coord_other_celltype = self.average_umap_coordinates[other_celltype]
                distance = np.linalg.norm(average_coord_celltype - average_coord_other_celltype)
                distances.append((other_celltype, distance))
            # sort
            distances.sort(key=lambda x: x[1])
            # now choose the clostest one according to phase
            # get rid of the current celltype as this is not relevant
            closest_celltypes = []
            for i in range(phase * number_of_targets):
                if i < len(distances):
                    closest_celltypes.append(distances[i][0])
            celltype_dict[celltype] = closest_celltypes
        return celltype_dict

    def _calculate_gene_activation_values(self, sd_factor: float=1.5) -> Dict[str, float]:
        """
        Calculates a target activation value for each reg gene based on mean + sd_factor*sd

        The value is calculated as: mean(expression) + sd_factor * SD(expression).
        Mean and SD are calculated after filtering outliers using the IQR method.
        Expression data is taken from self.oracle.adata.X.

        Args:
            sd_factor (float): The number of standard deviations to add to the mean.

        Returns:
            Dict[str, float]: A dictionary mapping gene names to their calculated activation values.
        """
        if sd_factor != 1.5:
            print("WARNING, CAN NOT USE NOW THE TRANS MATRIX")
        activation_values_dict = {}
        perturb_indices_in_adata = [self.oracle.adata.var.index.get_loc(g) for g in self.genes_that_can_be_perturbed
                                    if g in self.oracle.adata.var.index]
        expression_data = self.oracle.adata[:, perturb_indices_in_adata].X

        if issparse(expression_data):
            expression_data = expression_data.toarray()
        elif not isinstance(expression_data, np.ndarray):
            expression_data = np.asarray(expression_data)
        for i, gene_name in enumerate(self.genes_that_can_be_perturbed):

            if gene_name not in self.oracle.adata.var.index:
                continue

            gene_expr = expression_data[:, i]
            try:
                q1, q3 = np.percentile(gene_expr, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                mask_non_outlier = (gene_expr >= lower_bound) & (gene_expr <= upper_bound)
                filtered_expr = gene_expr[mask_non_outlier]
                mean_expr = np.mean(filtered_expr)
                sd_expr = np.std(filtered_expr)
                activation_val = mean_expr + sd_factor * sd_expr

                activation_values_dict[gene_name] = max(0.0, activation_val)

            except Exception as e:
                print("THIS SHOULD NOT HAPPEN: ", e)
                activation_values_dict[gene_name] = 0.0

        print(f"Finished calculating activation values for {len(activation_values_dict)} genes.")
        return activation_values_dict

    def set_phase(self, phase: int, step_increases: int = 5) -> bool:
        """
        Set the current phase for curriculum learning, done by a callback given to the ppo trainer
        """
        if phase > len(self.total_curriculum_targets):
            # not needed to do anything, just keep current phase
            return False
        phase_diff = phase - self.current_phase  # when setting it from saved file, we need to implement a phase diff
        self.max_steps += step_increases * phase_diff
        self.current_phase = phase
        return True

    def get_env_state(self) -> Dict[str, Any]:
        """Returns the current state of the environment relevant for curriculum learning."""
        return {
            "max_steps": self.max_steps,
            "current_phase": self.current_phase,
            "wandb_run_id": self.wandb_run_id,
            "wandb_run_name": self.wandb_run_name,
        }

    def set_env_state(self, state: Dict[str, Any]):
        """Sets the state of the environment from a loaded state."""
        self.max_steps = state.get("max_steps", self.max_steps)  # Default to current if not found
        self.current_phase = state.get("current_phase", self.current_phase)
        self.wandb_run_id = state.get("wandb_run_id", self.wandb_run_id)  # Default to current if not found
        self.wandb_run_name = state.get("wandb_run_name", self.wandb_run_name)

    def set_env_state_at_manually(self, current_phase: int = 5, max_steps: int = 66, wandb_id: str = "asdf",
                                  wandb_name: str = "asdf"):
        """
        Set the current state of the environment manually, used for debugging and testing purposes
        """
        self.current_phase = current_phase
        self.max_steps = max_steps
        self.wandb_run_id = wandb_id
        self.wandb_run_name = wandb_name

    def is_curriculum_finished(self) -> bool:
        """
        Checks if the current phase has reached the maximum number of curriculum phases.
        """
        return self.current_phase >= len(self.total_curriculum_targets)

    def get_current_goal_reached_percentage(self) -> float:
        """
        Returns the percentage of goals reached in the current phase.
        """
        if self.number_of_episodes_started_overal == 0:
            return 0.0
        return (self.number_of_goal_reached / self.number_of_episodes_started_overal) * 100.0

    def get_action_details(self, action_idx: int) -> str:
        """Converts an action index to a human-readable string."""
        if not (0 <= action_idx < self.action_space_size):
            return f"INVALID_ACTION_IDX_{action_idx}"

        if action_idx < self.number_of_reg_genes:
            gene_name = self.genes_that_can_be_perturbed[action_idx]
            return f"KO_{gene_name}"
        else:
            gene_index_in_list = action_idx - self.number_of_reg_genes
            gene_name = self.genes_that_can_be_perturbed[gene_index_in_list]
            return f"Activate_{gene_name}"

    def load_transition_data(self,transition_matrix_path: str, sd_factor: float = 1.5):


        with open(transition_matrix_path, 'rb') as f:
            transition_matrix = pickle.load(f)

        action_map = []
        #  Knockout actions
        for idx,gene_name in enumerate(self.genes_that_can_be_perturbed):
            action_map.append(gene_name)
        # Act
        for idx,gene_name in enumerate(self.genes_that_can_be_perturbed):
            action_map.append(gene_name)



        return transition_matrix, action_map

    def calculate_centre_umap_coords_target(self,
                                   outlier_percentile: float = 95) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Calculates the robust centroid for each cell type in an embedding.
        """
        target_centres = {}
        target_centres_norm = {}

        unique_celltypes = self.oracle.adata.obs["celltype"].unique()
        raw_embedding = self.oracle.adata.obsm[self.embedding_name]
        norm_embedding = self.umap_coords_normalized

        for celltype in unique_celltypes:
            cell_mask = self.oracle.adata.obs["celltype"] == celltype
            raw_cluster_coords = raw_embedding[cell_mask]
            norm_cluster_coords = norm_embedding[cell_mask]
            print(f"Calculating centroid for celltype: {celltype}, number of cells: {len(raw_cluster_coords)}")

            if len(raw_cluster_coords) == 0:
                continue
            temp_centre = np.mean(raw_cluster_coords, axis=0)
            distances = np.linalg.norm(raw_cluster_coords - temp_centre, axis=1)
            #select only 90% closest, get rid of outliers
            distance_cutoff = np.percentile(distances, outlier_percentile)
            core_cells_mask = distances < distance_cutoff
            core_raw_coords = raw_cluster_coords[core_cells_mask]
            core_norm_coords = norm_cluster_coords[core_cells_mask]

            final_raw_centroid = np.mean(core_raw_coords, axis=0)
            final_norm_centroid = np.mean(core_norm_coords, axis=0)
            print(f"Final centroid for celltype {celltype}: {final_raw_centroid}, norm: {final_norm_centroid}")


            target_centres[celltype] = final_raw_centroid
            target_centres_norm[celltype] = final_norm_centroid


        return target_centres, target_centres_norm

    def plot_umap_with_goal_regions(
            self,
            save_path: str = 'umap_with_goal_regions.png',
            show_plot: bool = False,
            plot_title: str = 'UMAP with Target Centroids and Success Radius',
            label_fontsize: int = 5,  # New parameter to control font size
            label_x_offset:float = 0.1,
            label_y_offset: float = 0.1 # Controls how far above the circle the label appears

    ) -> None:
        """
        Generates and saves a UMAP plot, overlaying robust centroids, success radius
        circles, and manually placed, smaller text labels for each cell type.
        """
        print(f"Generating UMAP plot with goal regions (radius: {self.target_bounds})...")

        adata_to_plot = self.oracle.adata
        centroids_to_plot = self.average_umap_coordinates
        success_radius = self.target_bounds
        celltype_key = 'celltype'

        ax = sc.pl.umap(
            adata_to_plot,
            color=celltype_key,
            legend_loc=None,
            frameon=False,
            show=False,
            title=plot_title
        )

        for celltype, centroid_coord in centroids_to_plot.items():
            success_circle = plt.Circle(
                xy=centroid_coord,
                radius=success_radius,
                edgecolor='black',
                facecolor='red',
                linewidth=0.5,
                alpha=0.25,
                zorder=5
            )
            ax.add_patch(success_circle)

            ax.scatter(
                centroid_coord[0],
                centroid_coord[1],
                marker='o',
                c='black',
                s=10, 
                zorder=10
            )
            ax.text(
                centroid_coord[0] +label_x_offset,
                centroid_coord[1] - success_radius - label_y_offset,
                celltype,
                fontsize=label_fontsize,
                color='black',
                fontweight='bold',
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=11,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none')
            )

        # 4. Adjust plot aesthetics
        ax.set_aspect('equal', adjustable='box')

        # 5. Save and/or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show_plot:
            plt.show()

        plt.close()
        sc.pl.umap(adata_to_plot, color=celltype_key, frameon=False, show=False)

    def get_total_goal_reached(self):
        return self.total_goals_reached

