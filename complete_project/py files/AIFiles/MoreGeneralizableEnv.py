import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.sparse import issparse
import cupy as cp
from stable_baselines3.common.vec_env import VecEnv
import math


class CellOracleSB3VecEnv(VecEnv):
    metadata = {"render_modes": []}

    def __init__(self,
                 oracle_path: str,
                 batch_size: int = 64,
                 max_steps: int = 50,
                 step_penalty: float = -0.01,
                 goal_bonus: float = 1.0,
                 fail_penalty: float = -1.0,
                 distance_reward_scale: float = 1.0,
                 same_cell_penalty: float = -0.2,
                 gamma_distance: float = 1.0,
                 allow_gene_activation: bool = False,
                 use_prev_perturbs: bool = False,
                 number_of_targets_curriculum: int = 4,
                 standard_sd_factor: float = 1,
                 use_similarity: bool = False,
                 initial_cell_idx: Optional[List[int]] = None,
                 target_cell_types: Optional[List[int]] = None):

        # --- Basic setup ---
        self._allow_gene_activation = allow_gene_activation
        self.max_steps = max_steps
        self.use_similarity = use_similarity
        self.wandb_run_id = None
        self.wandb_run_name = None
        self.check_path(oracle_path)
        try:
            cp.cuda.Device(0).use()
            self.cupy_integration_use = True
            print("CuPy is available and GPU is accessible.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            # Handle the case where CuPy cannot access the GPU
            print("CuPy cannot access GPU, falling back to CPU integration.")
            self.cupy_integration_use = False

        # Init counters
        self.number_of_episodes_started_completed = 0
        self.number_of_episodes_started_overal = 0
        self.number_of_goal_reached = 0

        # Load Oracle object
        with open(oracle_path, 'rb') as f:
            self.oracle = pickle.load(f)

        # Initialize oracle for use
        self.oracle.init(embedding_type="X_umap", n_neighbors=200, torch_approach=False,
                         cupy_approach=self.cupy_integration_use, batch_size=batch_size)
        self.use_prev_perturbs = use_prev_perturbs

        # --- DEFINE ALL NECESSARY ATTRIBUTES FIRST (THIS IS THE FIX) ---
        self.n_cells = self.oracle.adata.n_obs
        self.all_genes = self.oracle.adata.var.index.tolist()
        self.genes_that_can_be_perturbed = self.oracle.return_active_reg_genes()
        self.number_of_reg_genes = len(self.genes_that_can_be_perturbed)
        self.celltypes = self.oracle.adata.obs[
            'celltype'].unique().tolist() if 'celltype' in self.oracle.adata.obs.columns else None

        # This is the attribute that was missing before it was used
        self.reg_gene_adata_indices = np.array([self.all_genes.index(g) for g in self.genes_that_can_be_perturbed])
        self.reg_gene_to_full_idx = {name: i for i, name in enumerate(self.all_genes) if
                                     name in self.genes_that_can_be_perturbed}

        self.action_space_size = self.number_of_reg_genes
        self.action_space_size += self.number_of_reg_genes if self._allow_gene_activation else 0

        self.celltype_to_one_hot = self._create_cell_type_to_hot_encoded_vec_dict(self.celltypes)

        # --- NOW COMPUTE DERIVED DATA USING THE ATTRIBUTES ABOVE ---
        self.high_ranges_dict = self._calculate_gene_activation_values(
            sd_factor=standard_sd_factor) if self._allow_gene_activation else None
        self.average_expression_vectors = self._compute_average_expression_vectors()  # Now this will work
        self.total_curriculum_targets = self._compute_total_curriculum_targets(number_of_targets_curriculum)

        # --- Setup for Gym/SB3 ---
        self.debug_cell_idx = initial_cell_idx if initial_cell_idx is not None else None
        self.debug_target_cell_types = target_cell_types if target_cell_types is not None else None

        observation_space = self._create_observation_space()
        action_space = self._create_action_space()

        super().__init__(num_envs=batch_size,
                         observation_space=observation_space,
                         action_space=action_space)

        # --- Initialize state variables for the vectorized environment ---
        self.current_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_start_times = np.array([time.time()] * self.num_envs, dtype=np.float32)

        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.fail_penalty = fail_penalty
        self.distance_reward_scale = distance_reward_scale
        self.same_cell_penalty = same_cell_penalty
        self.gamma = gamma_distance

        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_cell_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.current_target_cell_types = np.full(self.num_envs, None, dtype=object)
        self.current_phase = 1

        cell_types_series = self.oracle.adata.obs['celltype']
        pgc_mask = cell_types_series.str.contains("PGCs", na=False)
        pgc_positional_indices = np.where(pgc_mask)[0].astype(np.int32)
        self.temp_indices_to_choose_from = pgc_positional_indices
        self._actions: Optional[np.ndarray] = None

    def set_wandb_id(self, wandb_run_id: str, wandb_run_name: str):
        self.wandb_run_id = wandb_run_id
        self.wandb_run_name = wandb_run_name

    def _create_cell_type_to_hot_encoded_vec_dict(self, celltypes: List[str]) -> dict:
        cell_type_to_hot_vec = {}
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
                raise ValueError(f"Cell type '{celltype}' not found in predefined list.")
        return cell_type_to_hot_vec

    def _create_action_space(self) -> spaces.Space:
        return spaces.Discrete(self.action_space_size)

    def _create_observation_space(self) -> spaces.Dict:
        low_bound, high_bound = -np.inf, np.inf
        return spaces.Dict({
            "current_state": spaces.Box(low=low_bound, high=high_bound, shape=(self.number_of_reg_genes,),
                                        dtype=np.float32),
            "target_state": spaces.Box(low=low_bound, high=high_bound, shape=(self.number_of_reg_genes,),
                                       dtype=np.float32),
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

        return self._get_obs()

    def reset_for_inference(self, start_idx: int, target_name: str) -> Dict[str, np.ndarray]:
        if self.num_envs > 1:
            print("Warning: reset_for_inference is designed for a single environment (num_envs=1).")

        self.current_steps.fill(0)
        self.current_episode_rewards.fill(0.0)
        self.current_episode_lengths.fill(0)
        self.current_episode_start_times[:] = time.time()
        self.oracle.reset_info_during_training_for_batch_instance(np.arange(self.num_envs))

        self.current_cell_indices[0] = start_idx
        self.current_target_cell_types[0] = target_name

        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        if self._actions is None:
            raise RuntimeError("Cannot call step_wait without calling step_async first.")

        actions = self._actions.copy()
        self._actions = None

        perturb_conditions = []
        action_indices = actions.flatten().astype(int)
        total_activations_per_step = 0
        for i in range(self.num_envs):
            action_idx = action_indices[i]
            if action_idx < self.number_of_reg_genes:
                gene_to_knockout = self.genes_that_can_be_perturbed[action_idx]
                perturb_conditions.append((gene_to_knockout, 0.0))
                continue
            if self._allow_gene_activation:
                if action_idx >= 2 * self.number_of_reg_genes:
                    raise ValueError("Action index out of bounds for gene activation.")
                gene_to_activate = self.genes_that_can_be_perturbed[action_idx % self.number_of_reg_genes]
                activation_value = self.high_ranges_dict.get(gene_to_activate, 0.0)
                perturb_conditions.append((gene_to_activate, activation_value))
                total_activations_per_step += 1

        simulation_success = True
        try:
            if self.cupy_integration_use:
                _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch_cp(
                    batch_size=self.num_envs, idxs=self.current_cell_indices.tolist(),
                    perturb_condition=perturb_conditions, n_neighbors=200, n_propagation=3, threads=4,
                    knockout_prev_used=self.use_prev_perturbs
                )
            else:
                _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch(
                    batch_size=self.num_envs, idxs=self.current_cell_indices.tolist(),
                    perturb_condition=perturb_conditions, n_neighbors=200, n_propagation=3, threads=4,
                    knockout_prev_used=self.use_prev_perturbs
                )
            new_cell_indices = np.array(new_idx_list, dtype=np.int32)
        except Exception as e:
            import traceback
            traceback.print_exc()
            simulation_success = False
            new_cell_indices = self.current_cell_indices.copy()

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)

        if not simulation_success:
            rewards.fill(self.fail_penalty)
            terminations.fill(True)
            obs_before_reset = self._get_obs()
            infos = self._create_info_dict(
                goal_reached=np.zeros(self.num_envs, dtype=bool),
                truncated=truncations,
                simulation_failed=np.ones(self.num_envs, dtype=bool),
                same_cell_indices_as_before_mask=np.zeros(self.num_envs, dtype=bool),
                obs_before_reset=obs_before_reset
            )
            self._handle_batch_resets(terminations)
            return self._get_obs(), rewards, terminations, truncations, infos

        same_cell_indices_as_before_mask = new_cell_indices == self.current_cell_indices
        rewards_distance_phi = self._reward_system_distance_calc(new_cell_indices)
        self.current_cell_indices = new_cell_indices
        self.current_steps += 1

        celltypes_of_new_indices = self.oracle.adata[new_cell_indices].obs['celltype'].to_numpy()
        goal_reached_mask = celltypes_of_new_indices == self.current_target_cell_types

        is_timeout_mask = self.current_steps >= self.max_steps
        rewards.fill(self.step_penalty)
        rewards += rewards_distance_phi
        rewards[is_timeout_mask] += self.fail_penalty
        rewards[goal_reached_mask] += self.goal_bonus
        rewards[same_cell_indices_as_before_mask] += self.same_cell_penalty

        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1

        terminations[goal_reached_mask] = True
        truncations[is_timeout_mask] = True

        done_and_goal = goal_reached_mask & terminations
        self.number_of_goal_reached += np.sum(done_and_goal)

        fail_pen = np.zeros(self.num_envs, dtype=np.float32)
        fail_pen[is_timeout_mask] = self.fail_penalty
        average_pen = np.mean(fail_pen)
        goal_reach = np.zeros(self.num_envs, dtype=np.float32)
        goal_reach[goal_reached_mask] = self.goal_bonus
        average_goal = np.mean(goal_reach)
        av_same_cell_indices_as_before_mask = np.zeros(self.num_envs, dtype=np.float32)
        av_same_cell_indices_as_before_mask[same_cell_indices_as_before_mask] = self.same_cell_penalty
        average_same_cell = np.mean(av_same_cell_indices_as_before_mask)
        average_distance = np.mean(rewards_distance_phi)

        obs_before_reset = self._get_obs()

        infos = self._create_info_dict(
            goal_reached=goal_reached_mask,
            truncated=truncations,
            simulation_failed=np.zeros(self.num_envs, dtype=bool),
            same_cell_indices_as_before_mask=same_cell_indices_as_before_mask,
            obs_before_reset=obs_before_reset,
            average_goal=average_goal,
            average_penalty=average_pen,
            average_distance=average_distance,
            average_same_cell=average_same_cell,
            percentage_of_activation=total_activations_per_step / self.num_envs,
            current_cell_types=celltypes_of_new_indices,
            target_cell_types=self.current_target_cell_types
        )

        dones = terminations | truncations
        self._handle_batch_resets(dones)

        return self._get_obs(), rewards, dones, infos


    def _handle_batch_resets(self, dones: np.ndarray) -> None:
        reset_indices = np.where(dones)[0]
        num_to_reset = len(reset_indices)

        if num_to_reset <= 0:
            return

        reset_rng = np.random.default_rng(self._np_random_seed + self.current_steps.sum() if hasattr(self,
                                                                                                     "_np_random_seed") and self._np_random_seed is not None else None)
        new_start_indices = reset_rng.choice(self.temp_indices_to_choose_from, size=num_to_reset, replace=False)
        new_target_types_arr = self._get_target_cell_types(new_start_indices, reset_rng)

        self.current_cell_indices[reset_indices] = new_start_indices
        self.current_target_cell_types[reset_indices] = new_target_types_arr
        self.current_steps[reset_indices] = 0
        self.current_episode_rewards[reset_indices] = 0.0
        self.current_episode_lengths[reset_indices] = 0
        self.current_episode_start_times[reset_indices] = time.time()
        self.oracle.reset_info_during_training_for_batch_instance(reset_indices)
        self.number_of_episodes_started_completed += num_to_reset
        self.number_of_episodes_started_overal += num_to_reset

    def _reward_system_distance_calc(self, next_indices_after_perturb: np.ndarray) -> np.ndarray:
        distances_next = self._calculate_expression_distances_to_target(next_indices_after_perturb)
        distances_before = self._calculate_expression_distances_to_target(self.current_cell_indices)
        if self.use_similarity:
            epsilon = 1e-8
            potential_next = 1.0 / (1.0 + distances_next + epsilon)
            potential_before = 1.0 / (1.0 + distances_before + epsilon)
            phi_reward = self.gamma * potential_next - potential_before
        else:
            costs_change = self.gamma * distances_next - distances_before
            phi_reward = -costs_change

        phi_reward *= self.distance_reward_scale
        return phi_reward

    def _calculate_expression_distances_to_target(self, cell_indices: np.ndarray) -> np.ndarray:
        current_expr_vectors = self._get_current_expression_vector(cell_indices)
        target_expr_vectors = self._get_target_expression_vectors()

        diff = current_expr_vectors - target_expr_vectors
        distances = np.linalg.norm(diff, axis=1)
        return np.maximum(distances, 0.0).astype(np.float32)

    def _create_info_dict(self, goal_reached: np.ndarray, truncated: np.ndarray, simulation_failed: np.ndarray,
                          same_cell_indices_as_before_mask: np.ndarray,
                          obs_before_reset: Dict[str, np.ndarray],
                          percentage_of_activation: float = None,
                          average_goal: float = None,
                          average_penalty: float = None, average_distance: float = None,
                          average_same_cell: float = None, current_cell_types: np.ndarray = None,
                          target_cell_types: np.ndarray = None) -> List[Dict[str, Any]]:
        infos = []
        is_done = goal_reached | truncated | simulation_failed
        unique_cell_types = np.unique(current_cell_types) if current_cell_types is not None else None
        unique_target_cell_types = np.unique(target_cell_types) if target_cell_types is not None else None

        for i in range(self.num_envs):
            info = {}
            if average_goal is not None: info["step_avg/average_goal"] = average_goal
            if average_penalty is not None: info["step_avg/average_penalty"] = average_penalty
            if average_distance is not None: info["step_avg/average_distance"] = average_distance
            if average_same_cell is not None:
                info["step_avg/average_same_cell"] = average_same_cell
                info["diagnostics/same_cell_event_this_step"] = bool(same_cell_indices_as_before_mask[i])
            if current_cell_types is not None: info[
                "batch_diversity/current_cell_types_unique_in_batch"] = unique_cell_types
            if target_cell_types is not None: info[
                "batch_diversity/target_cell_types_unique_in_batch"] = unique_target_cell_types
            if percentage_of_activation is not None: info[
                "step_avg/percentage_of_activation"] = percentage_of_activation

            if is_done[i]:
                info["final_observation"] = {k: v[i] for k, v in obs_before_reset.items()}
                info["final_info"] = {
                    "steps": self.max_steps if truncated[i] else self.current_steps[i],
                    "goal_reached": goal_reached[i],
                    "truncated": truncated[i],
                    "simulation_failed": simulation_failed[i],
                    "episode": {
                        "r": self.current_episode_rewards[i],
                        "l": self.current_episode_lengths[i],
                        "t": time.time() - self.current_episode_start_times[i]
                    },
                }
            infos.append(info)
        return infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(self, attr_name) for _ in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> None:
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        elif hasattr(self.oracle, attr_name):
            setattr(self.oracle, attr_name, value)
        else:
            raise AttributeError(f"Attribute '{attr_name}' not found on environment or oracle.")

    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, List[int], np.ndarray]] = None,
                   **method_kwargs) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            result = method(*method_args, **method_kwargs)
            return [result] * len(target_envs)
        elif hasattr(self.oracle, method_name):
            method = getattr(self.oracle, method_name)
            result = method(*method_args, **method_kwargs)
            return [result] * len(target_envs)
        else:
            raise AttributeError(f"Method '{method_name}' not found on environment or oracle.")

    def env_is_wrapped(self, wrapper_class: type, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[
        bool]:
        return [False] * self.num_envs

    def _get_target_envs(self, indices: Optional[Union[int, List[int], np.ndarray]]) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        elif isinstance(indices, int):
            return [indices]
        return list(indices)

    def check_path(self, oracle_path: str):
        if not oracle_path.endswith('.pkl'): raise ValueError('Pickle file needed')
        if not os.path.exists(oracle_path): raise ValueError(f'Path does not exist: {oracle_path}')

    def _get_current_expression_vector(self, cell_indices: np.ndarray) -> np.ndarray:
        data = self.oracle.get_AI_input_for_cell_indices(cell_indices, self.reg_gene_adata_indices)
        return data.toarray().astype(np.float32) if issparse(data) else np.array(data, dtype=np.float32)

    def _get_target_expression_vectors(self) -> np.ndarray:
        target_vectors = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=np.float32)
        for i, target_type in enumerate(self.current_target_cell_types):
            avg_vector = self.average_expression_vectors.get(target_type)
            if avg_vector is not None:
                target_vectors[i] = avg_vector
        return target_vectors

    def _get_target_cell_types(self, cell_indices: np.ndarray, rng_gen=None) -> np.ndarray:
        current_cell_types = self.oracle.adata[cell_indices].obs['celltype'].to_numpy()
        target_cell_types = []
        target_rng = rng_gen if rng_gen is not None else np.random.default_rng(
            self._np_random_seed if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)

        for i in range(len(current_cell_types)):
            phase_target_options = self.total_curriculum_targets[self.current_phase - 1].get(current_cell_types[i], [])
            if not phase_target_options:
                all_other_types = [ct for ct in self.celltypes if ct != current_cell_types[i] and 'PGC' not in ct]
                target_cell_types.append(
                    target_rng.choice(all_other_types) if all_other_types else current_cell_types[i])
            else:
                target_cell_types.append(target_rng.choice(phase_target_options))
        return np.array(target_cell_types, dtype=object)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        current_state_vecs = self._get_current_expression_vector(self.current_cell_indices)
        target_state_vecs = self._get_target_expression_vectors()

        return {
            "current_state": current_state_vecs,
            "target_state": target_state_vecs,
        }

    def _compute_average_expression_vectors(self) -> Dict[str, np.ndarray]:
        average_vectors = {}
        for celltype in self.celltypes:
            boolean_mask_celltype = (self.oracle.adata.obs['celltype'] == celltype).values
            cluster_reg_gene_expr = self._get_current_expression_vector(np.where(boolean_mask_celltype)[0])
            if cluster_reg_gene_expr.shape[0] > 0:
                average_vectors[celltype] = np.mean(cluster_reg_gene_expr, axis=0)
        return average_vectors

    def _compute_total_curriculum_targets(self, number_of_targets: int) -> List[Dict[str, List[str]]]:
        number_of_phases = math.ceil(len(self.celltypes) / number_of_targets) + 1
        total_curriculum_targets = []
        for phase in range(1, number_of_phases + 1):
            phase_targets = self._compute_nearest_celltype_targets_only_pgcs(phase, number_of_targets)
            total_curriculum_targets.append(phase_targets)
        return total_curriculum_targets

    def _compute_nearest_celltype_targets_only_pgcs(self, phase: int, number_of_targets: int) -> Dict[str, List[str]]:
        celltype_dict = {}
        pgc_celltypes = [celltype for celltype in self.celltypes if 'PGC' in celltype]
        for celltype in pgc_celltypes:
            avg_expr_celltype = self.average_expression_vectors[celltype]
            distances = []
            for other_celltype, avg_expr_other in self.average_expression_vectors.items():
                if other_celltype in pgc_celltypes or celltype == other_celltype:
                    continue
                distance = np.linalg.norm(avg_expr_celltype - avg_expr_other)
                distances.append((other_celltype, distance))

            distances.sort(key=lambda x: x[1])
            num_targets_for_phase = min(phase * number_of_targets, len(distances))
            celltype_dict[celltype] = [dist[0] for dist in distances[:num_targets_for_phase]]
        return celltype_dict

    def _calculate_gene_activation_values(self, sd_factor: float) -> Dict[str, float]:
        activation_values_dict = {}
        perturb_indices_in_adata = [self.oracle.adata.var.index.get_loc(g) for g in self.genes_that_can_be_perturbed if
                                    g in self.oracle.adata.var.index]
        expression_data = self.oracle.adata[:, perturb_indices_in_adata].X
        if issparse(expression_data): expression_data = expression_data.toarray()

        for i, gene_name in enumerate(self.genes_that_can_be_perturbed):
            if gene_name not in self.oracle.adata.var.index: continue
            gene_expr = expression_data[:, i]
            q1, q3 = np.percentile(gene_expr, [25, 75])
            iqr = q3 - q1
            filtered_expr = gene_expr[(gene_expr >= q1 - 3 * iqr) & (gene_expr <= q3 + 3 * iqr)]
            activation_val = np.mean(filtered_expr) + sd_factor * np.std(filtered_expr)
            activation_values_dict[gene_name] = max(0.0, activation_val)
        return activation_values_dict

    def set_phase(self, phase: int, step_increases: int = 5) -> bool:
        if phase > len(self.total_curriculum_targets): return False
        phase_diff = phase - self.current_phase
        self.max_steps += step_increases * phase_diff
        self.current_phase = phase
        return True

    def get_env_state(self) -> Dict[str, Any]:
        return {"max_steps": self.max_steps, "current_phase": self.current_phase, "wandb_run_id": self.wandb_run_id,
                "wandb_run_name": self.wandb_run_name}

    def set_env_state(self, state: Dict[str, Any]):
        self.max_steps = state.get("max_steps", self.max_steps)
        self.current_phase = state.get("current_phase", self.current_phase)
        self.wandb_run_id = state.get("wandb_run_id", self.wandb_run_id)
        self.wandb_run_name = state.get("wandb_run_name", self.wandb_run_name)

    def is_curriculum_finished(self) -> bool:
        return self.current_phase >= len(self.total_curriculum_targets)

    def get_current_goal_reached_percentage(self) -> float:
        return (
                           self.number_of_goal_reached / self.number_of_episodes_started_overal) * 100.0 if self.number_of_episodes_started_overal > 0 else 0.0

    def get_action_details(self, action_idx: int) -> str:
        if not (0 <= action_idx < self.action_space_size): return f"INVALID_ACTION_IDX_{action_idx}"
        if action_idx < self.number_of_reg_genes:
            return f"KO_{self.genes_that_can_be_perturbed[action_idx]}"
        else:
            return f"Activate_{self.genes_that_can_be_perturbed[action_idx - self.number_of_reg_genes]}"