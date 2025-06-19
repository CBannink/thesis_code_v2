# Environment.py

import gymnasium as gym  # Still needed for spaces
from gymnasium import spaces
import numpy as np
import pickle
import os
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Union
from scipy.sparse import issparse  # Check if needed based on _get_current_expression_vector

# --- Import SB3 VecEnv ---
from stable_baselines3.common.vec_env import VecEnv


# -------------------------

# Assuming your Oracle class structure is known (for type hinting if desired)
# class CellOracle:
#     adata: Any
#     def return_active_reg_genes(self) -> List[str]: pass
#     def training_phase_inference_batch(self, ...) -> Tuple[...]: pass
#     # Add other methods used, like reset_info_during_training if applicable


###INFO:
# 1) CURRENTLY WE ONLY USE THE KNOCKOUTS, NOT THE ACTIVATIONS,
# 2) THE REWARDS ARE VERY SIMPLE ->WE IMPLEMENTED SAME CELL INDICES TRAKCING THOUGH
# 3) WE ONLY USE A 110 INPUT SPACE (THE NUMBER OF REG GENES), FOR SIMPLICITY, WE ONLY USE A REDUCED GENE SPACE OF ONLY THE REG GENES, MAYBE SCALE UP LATER


class CellOracleSB3VecEnv(VecEnv):  # Inherit from SB3 VecEnv

    metadata = {"render_modes": []}  # Add metadata if needed

    def __init__(self,
                 oracle_path: str, batch_size: int = 64, max_steps: int = 50,
                 gene_activity_threshold: float = 0.01, target_distance_threshold: float = 0.1,
                 step_penalty: float = -0.01, goal_bonus: float = 1.0, fail_penalty: float = -1.0,
                 distance_reward_scale: float = 1.0,
                 allow_gene_activation: bool = False, high_range: float = None, use_prev_perturbs: bool = False,
                 # Add sampling strategy args
                 initial_cell_idx: Optional[List[int]] = None,
                 target_cell_types: Optional[List[int]] = None):

        # --- Setup required BEFORE calling super().__init__ ---
        self._allow_gene_activation = allow_gene_activation  # Store for space creation
        self._high_range = high_range  # Store for space creation
        self.max_steps = max_steps
        self.gene_activity_threshold = gene_activity_threshold

        # Load Oracle temporarily for space dimensions
        self.check_path(oracle_path)  # Check path early
        with open(oracle_path, 'rb') as f: self.oracle = pickle.load(f)
        self.oracle.init(embedding_type="umap", n_neighbors=200, torch_approach=False)
        self.use_prev_perturbs = use_prev_perturbs

        # Bio params
        self.n_cells = self.oracle.adata.n_obs
        self.all_genes = self.oracle.adata.var.index.tolist()
        self.genes_that_can_be_perturbed = self.oracle.return_active_reg_genes()  # used to convert the action idx to a gene name, the policy network output we receive is a single int, so we use that to index in this list
        self.number_of_reg_genes = len(self.genes_that_can_be_perturbed)
        self.celltypes = self.oracle.adata.obs[
            'celltype'].unique().tolist() if 'celltype' in self.oracle.adata.obs.columns else None

        # Debugging paramas when required
        self.debug_cell_idx = initial_cell_idx if initial_cell_idx is not None else None
        self.debug_target_cell_types = target_cell_types if target_cell_types is not None else None

        # Pass necessary dimension/config info directly
        observation_space = self._create_observation_space()
        action_space = self._create_action_space()

        # --- Call the SB3 VecEnv super().__init__ ---
        super().__init__(num_envs=batch_size,
                         observation_space=observation_space,
                         action_space=action_space)

        # ENV HELPER params
        self.celltype_to_one_hot = self._create_cell_type_to_hot_encoded_vec_dict(self.celltypes)
        self.reg_gene_to_full_idx = {name: i for i, name in enumerate(self.all_genes) if
                                     name in self.genes_that_can_be_perturbed}
        self.reg_gene_adata_indices = np.array([self.all_genes.index(g) for g in self.genes_that_can_be_perturbed])

        # Reward params
        self.target_distance_threshold = target_distance_threshold
        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.fail_penalty = fail_penalty
        self.distance_reward_scale = distance_reward_scale

        # State params
        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_cell_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.knockout_histories = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=bool)
        self.current_target_cell_types = np.full(self.num_envs, None, dtype=object)

        # Action buffer for async methods
        self._actions: Optional[np.ndarray] = None

    def _create_cell_type_to_hot_encoded_vec_dict(self, celltypes: List[str]) -> dict:
        """Create a mapping from cell type names to integers."""
        cell_type_to_hot_vec = {}
        # Declare this list in the order it is given in the paper, thus linking up the ints to the correct ones
        cell_type_string = ["mESCs", "Primitive Streak", "Caudal Epiblast", "Epiblast", "Naive PGCs",
                            "Epidermis Progenitors", "Caudal Mesoderm", "Parietal Endoderm", "PGCs",
                            "(pre)Somitic/Wavefront", "Nascent Mesoderm", "NMPs", "LP/Intermediate Mesoderm",
                            "Neural Progenitors", "(early) Somite", "Cardiac Mesoderm", "Endothelium",
                            "Visceral Endoderm", "Dermomyotome", "Erythrocytes", "Reprogramming PGCs", "Sclerotome",
                            "Roof Plate Neural Tube", "Floor Plate Neural Tube", "ExE Endoderm", "Cardiomyocytes",
                            "Pharyngeal Mesoderm", "Early Motor Neurons", "Late Motor Neurons", "Myotome",
                            "Megakaryocytes"]
        one_hot_length = len(celltypes)
        print(celltypes, " and with length: ", len(celltypes))
        for i, celltype in enumerate(cell_type_string):
            if celltype in celltypes:
                one_hot_vec = np.zeros(one_hot_length, dtype=np.float32)
                one_hot_vec[i] = 1.0
                cell_type_to_hot_vec[celltype] = one_hot_vec
            else:
                print(
                    "Warning: Cell type not found in predefined list. We might have excluded some data which causes this! , celltype: ",
                    celltype)

        return cell_type_to_hot_vec

    def _create_action_space(self) -> spaces.Space:
        # Uses self._number_of_reg_genes, self._allow_gene_activation, self._high_range
        if not self._allow_gene_activation:
            return spaces.Discrete(self.number_of_reg_genes)
        else:
            raise NotImplementedError("Dict action space for activation not supported by default SB3 policies.")
            # high_range = self._high_range if self._high_range is not None else 1.0
            # return spaces.Dict({"gene_index": spaces.Discrete(self._number_of_reg_genes), ... })

    def _create_observation_space(self) -> spaces.Dict:
        # Uses self._number_of_reg_genes, self._max_steps
        low_bound, high_bound = -np.inf, np.inf
        return spaces.Dict({
            "current_state": spaces.Box(low=low_bound, high=high_bound, shape=(self.number_of_reg_genes,),
                                        dtype=np.float32),
            "target_cell_type": spaces.Box(low=0.0, high=1.0, shape=(len(self.celltypes),), dtype=np.float32),
            # one hot encoded vector
            "steps_left": spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.float32)
            # "knockout_history": spaces.MultiBinary(self.number_of_reg_genes)
        })

    # --- SB3 VecEnv Abstract Methods Implementation ---

    def reset(self) -> Dict[str, np.ndarray]:
        """Resets this environment."""
        # Seeding: Rely on SB3's self.seed() method to set self._np_random_seed if needed
        rng = np.random.default_rng(
            self._np_random_seed if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)
        target_rng = np.random.default_rng(
            self._np_random_seed + 1 if hasattr(self, "_np_random_seed") and self._np_random_seed is not None else None)

        self.current_cell_indices = rng.integers(0, self.n_cells, size=self.num_envs, dtype=np.int32)

        # now sample target celltypes
        random_target_names = target_rng.choice(self.celltypes, size=self.num_envs, replace=True)
        for i, name in enumerate(random_target_names):
            self.current_target_cell_types[i] = name

        # for debug purposses
        if self.debug_cell_idx is not None:
            self.current_cell_indices = np.array(self.debug_cell_idx)
        if self.debug_target_cell_types is not None:
            for i in range(self.num_envs):
                self.current_target_cell_types[i] = self.debug_target_cell_types[i]

        self.oracle.reset_info_during_training()
        self.current_steps.fill(0)
        self.knockout_histories.fill(False)

        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        """Stores the actions."""
        self._actions = actions

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        """Performs the simulation and returns results (synchronous execution).
            FOR NOW:
            1) WE DO NOT INTEGRATE A POLICY MASK (ACTIONS THAT ARE NOT POSSIBLE)
            2) WE DO INTEGRATE THE MOST SIMPLE REWARD STRATEGY, -1 FOR EACH STEP, +10 FOR REACHING THE GOAL, -10 FOR FAILING


        """
        if self._actions is None:
            raise RuntimeError("Cannot call step_wait without calling step_async first.")

        actions = self._actions
        self._actions = None

        # update history for now, not implemented yet atm but for later on
        # history_list = []
        # for i in range(self.num_envs):
        #     hist = []
        #     for idx, knocked in enumerate(self.knockout_histories[i]):
        #         if knocked: hist.append((self.genes_that_can_be_perturbed[idx], 0.0))
        #     history_list.append(hist)

        perturb_conditions = []
        action_indices = actions.flatten().astype(int)
        for i in range(self.num_envs):
            action_idx = action_indices[i]
            # Basic check for action validity (important if not using MaskablePPO perfectly)
            if not 0 <= action_idx < self.number_of_reg_genes:
                print(f"ERROR in step_wait: Invalid action index {action_idx} received for env {i}.")
                # Handle error: Maybe force fail state? For now, proceed cautiously.
                # This shouldn't happen if action space is correct.
                action_idx = 0  # Default to first action? Risky.

            gene_to_knockout = self.genes_that_can_be_perturbed[action_idx]
            perturb_conditions.append((gene_to_knockout, 0.0))  # Assuming knockout value is always 0.0

        n_neighbors_sim, n_propagation_sim, threads_sim = 200, 3, 4
        simulation_success = True
        try:
            # perform sim
            _, new_idx_list, _, _, _ = self.oracle.training_phase_inference_batch(
                batch_size=self.num_envs, idxs=self.current_cell_indices.tolist(),
                perturb_condition=perturb_conditions, n_neighbors=n_neighbors_sim,
                n_propagation=n_propagation_sim, threads=threads_sim,
                knockout_prev_used=self.use_prev_perturbs
            )
            new_cell_indices = np.array(new_idx_list, dtype=np.int32)  # Ensure correct type
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print the full traceback
            print('WATCH OUT THIS IS WRONG')
            simulation_success = False
            new_cell_indices = self.current_cell_indices  # Stay in same state

        # init all necessary info
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        terminations = np.zeros(self.num_envs, dtype=bool)  # For info dict
        truncations = np.zeros(self.num_envs, dtype=bool)  # For info dict

        if not simulation_success:
            print("THIS IS A PROBLEM IF THIS HAPPENS")
            # just terminate the entire episode if it fails, this indicates a problem
            rewards.fill(self.fail_penalty)
            dones.fill(True)
            terminations.fill(True)
            truncations.fill(False)
            return self._get_obs(), rewards, dones, self._create_info_dict(
                goal_reached=np.zeros(self.num_envs, dtype=bool), timeout=np.zeros(self.num_envs, dtype=bool),
                simulation_failed=np.ones(self.num_envs, dtype=bool))

        # The chance of this happening above is slim but we just implement it to be sure and catch if it goes wrong, can save alot of pain debugging DO NOT REMOVE LATER !

        # Now we assume the simulation has been successful, we check if the new cell indices are the correct type of the target, give rewards for this step
        # IMPLEMENT LATER: CHECK IF THE INDICES ARE THE SAME AS THE ORIGINAL, THAT MEANS THAT THE PERTURBATION HAS SO LITTLE CHANGE THAT THE NEW NEIGHBOR IS STILL THE ORIGINAL, THUS NO EFFECT -> PENALIZE VERY HEAVILY
        # IMPORTANT: GET RID OF THE RETURN OF THE 2ND NEIGHBOR IF THE SAME AS ORIGINAL IN THE GET_POST_PERTURB_NN IN CELLORACLE

        # first set the neew info correctly
        same_cell_indices_as_before_mask = new_cell_indices == self.current_cell_indices
        self.current_cell_indices = new_cell_indices
        self.current_steps += 1

        celltypes_of_new_indices = self.oracle.adata[new_cell_indices].obs['celltype']
        goal_reached_mask = celltypes_of_new_indices == self.current_target_cell_types
        is_timeout_mask = self.current_steps >= self.max_steps  # yes this is an actual mask that is created DO NOT MESS WITH IT LATER

        # set to default step penalty
        rewards.fill(self.step_penalty)
        # then add a possible time out penalty
        rewards[is_timeout_mask] += self.fail_penalty
        # but if goal reached then add the goal reached bonus
        rewards[goal_reached_mask] += self.goal_bonus
        # add additgional if same cell is returned, very bad perturbation
        rewards[
            same_cell_indices_as_before_mask] += self.fail_penalty  # if the same cell is returned, we penalize heavily
        dones[is_timeout_mask] = True
        dones[goal_reached_mask] = True
        terminations[goal_reached_mask] = True
        # same for here, first do timeout, then overwrite with goal reached
        truncations[is_timeout_mask] = True
        truncations[goal_reached_mask] = False

        return self._get_obs(), rewards, dones, self._create_info_dict(goal_reached=goal_reached_mask,
                                                                       timeout=is_timeout_mask,
                                                                       simulation_failed=np.zeros(self.num_envs,
                                                                                                  dtype=bool))

    def _create_info_dict(self, goal_reached: np.ndarray, timeout: np.ndarray, simulation_failed: np.ndarray) -> list:
        """
            We create a dict for each env, logging some info we can use later in the dashboard I think?
            We adhere to the normal standards of the SB3 VecEnv, so we can use the SB3 monitor wrapper
        """
        obs = self._get_obs()
        infos = []
        for i in range(self.num_envs):
            info_dict_per_env = {
                "cell_type_env": self.oracle.adata[self.current_cell_indices[i]].obs['celltype'],
                "terminal_observation": {k: v[i] for k, v in obs.items()},
                "final_info": {
                    "steps": self.current_steps[i],
                    "goal_reached": goal_reached[i],
                    "timeout": timeout[i],
                    "simulation_failed": simulation_failed[i],
                },
                "TimeLimit.truncated": timeout[i],
            }
            infos.append(info_dict_per_env)
        return infos

    def close(self) -> None:
        """Closes the environment."""
        print("Closing environment.")
        # If self.oracle needs explicit closing, do it here
        pass

    def get_attr(self, attr_name: str, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[Any]:
        """Return attribute from vectorized environment."""
        # Required by SB3 VecEnv
        target_envs = self._get_target_envs(indices)
        return [getattr(self, attr_name) for _ in target_envs]  # Simplified: assumes attr is on self

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> None:
        """Set attribute inside vectorized environment."""
        # Required by SB3 VecEnv
        target_envs = self._get_target_envs(indices)
        # Simplified: Set on self, affects all 'conceptual' envs
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
        # Required by SB3 VecEnv
        target_envs = self._get_target_envs(indices)
        # Simplified: Try calling on self.oracle
        if hasattr(self.oracle, method_name):
            method = getattr(self.oracle, method_name)
            # Assume method is safe to call once and result applies to all conceptual envs
            result = method(*method_args, **method_kwargs)
            return [result for _ in target_envs]
        else:
            # Or maybe call on self?
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                result = method(*method_args, **method_kwargs)  # Check if method expects batch indices
                return [result for _ in target_envs]  # May need adjustment
            else:
                raise AttributeError(f"Method '{method_name}' not found on environment or oracle.")

    def env_is_wrapped(self, wrapper_class, indices: Optional[Union[int, List[int], np.ndarray]] = None) -> List[bool]:
        """Check if environment is wrapped with a given wrapper."""
        # Required by SB3 VecEnv
        target_envs = self._get_target_envs(indices)
        # This custom VecEnv isn't wrapping others in the standard way
        return [False for _ in target_envs]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seeds the random number generator of the vector environment."""
        # Required by SB3 VecEnv
        # Store the main seed. The actual RNG should be created/re-seeded in reset()
        self._np_random_seed = seed
        # Return a list of seeds, one for each conceptual env
        seeds = []
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        # If you need an RNG here, use numpy directly
        # self._np_random = np.random.default_rng(seed)
        return seeds

    def _get_target_envs(self, indices: Optional[Union[int, List[int], np.ndarray]]) -> List[int]:
        """Helper to get target environment indices."""
        # Required by get_attr, set_attr etc.
        if indices is None:
            return list(range(self.num_envs))
        elif isinstance(indices, int):
            return [indices]
        elif isinstance(indices, (list, tuple, np.ndarray)):
            return list(indices)
        else:
            raise ValueError("Indices must be None, int, list, tuple, or np.ndarray")

    # --- Helper methods ---
    # (Implementations from previous VecEnv version should mostly work)
    def check_path(self, oracle_path: str):
        if not oracle_path.endswith('.pkl'): raise ValueError('Pickle file needed')
        if not os.path.exists(oracle_path): raise ValueError(f'Path does not exist: {oracle_path}')

    def _get_current_expression_vector(self, cell_indices: np.ndarray) -> np.ndarray:
        try:
            # Ensure indices are valid before slicing
            valid_indices = np.array(cell_indices, dtype=int)
            valid_indices = valid_indices[valid_indices < self.oracle.adata.n_obs]  # Basic bounds check
            if len(valid_indices) != len(cell_indices):
                print(f"Warning: Some cell indices are out of bounds. Using only valid ones.")
                # How to handle? Return zeros for invalid? For now, proceed with valid. This indicates bigger issue.
                cell_indices = valid_indices
                if len(cell_indices) == 0: return np.zeros((0, self.number_of_reg_genes), dtype=np.float32)

            data = self.oracle.adata[cell_indices, self.reg_gene_adata_indices].X
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

    def _get_obs(self) -> Dict[str, np.ndarray]:
        current_state_vecs = self._get_current_expression_vector(self.current_cell_indices)
        steps_left = (self.max_steps - self.current_steps).astype(np.float32).reshape(-1, 1)
        target_state_one_hot_vectors = self._get_target_state_vectors()

        # Ensure shapes match observation space even if errors occurred, do not know why this would go wrong, get rid of this, just raide value error
        # if current_state_vecs.shape[0] != self.num_envs: current_state_vecs = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=np.float32)
        # if steps_left.shape[0] != self.num_envs: steps_left = np.zeros((self.num_envs, 1), dtype=np.float32)
        # if self.knockout_histories.shape[0] != self.num_envs: self.knockout_histories = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=bool)

        if current_state_vecs.shape[0] != self.num_envs:
            raise ValueError(f"Current state vectors shape mismatch: {current_state_vecs.shape} vs {self.num_envs}")
        if steps_left.shape[0] != self.num_envs:
            raise ValueError(f"Steps left shape mismatch: {steps_left.shape} vs {self.num_envs}")
        if self.knockout_histories.shape[0] != self.num_envs:
            raise ValueError(f"Knockout histories shape mismatch: {self.knockout_histories.shape} vs {self.num_envs}")

        return {
            "current_state": current_state_vecs,
            "target_cell_type": target_state_one_hot_vectors,
            "steps_left": steps_left
            # "knockout_history": self.knockout_histories.astype(np.int8)
        }

    def action_masks(self) -> np.ndarray:
        if self.current_cell_indices is None or len(self.current_cell_indices) != self.num_envs:
            return np.ones((self.num_envs, self.number_of_reg_genes), dtype=bool)

        return self._get_dynamic_action_masks()  # Call your existing helper

    def _get_dynamic_action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_envs, self.number_of_reg_genes), dtype=bool)
        try:
            # Handle potential index errors if current_cell_indices became invalid
            valid_indices_mask = self.current_cell_indices < self.oracle.adata.n_obs
            valid_current_indices = self.current_cell_indices[valid_indices_mask]
            if len(valid_current_indices) == 0: return masks  # No valid cells to get masks for

            current_states_full_adata_X = self.oracle.adata[valid_current_indices, :].X
            if issparse(current_states_full_adata_X): current_states_full_adata_X = current_states_full_adata_X.tocsr()

            env_indices = np.where(valid_indices_mask)[0]  # Indices in the batch corresponding to valid cells
            for idx_in_batch, i in enumerate(env_indices):  # i is the original batch index
                cell_adata_row = current_states_full_adata_X[idx_in_batch, :]
                for action_idx, gene_name in enumerate(self.genes_that_can_be_perturbed):
                    full_gene_idx = self.reg_gene_to_full_idx.get(gene_name)
                    if full_gene_idx is not None:
                        expression = cell_adata_row[0, full_gene_idx] if issparse(cell_adata_row) else cell_adata_row[
                            full_gene_idx]
                        if expression >= self.gene_activity_threshold and not self.knockout_histories[i, action_idx]:
                            masks[i, action_idx] = True
        except Exception as e:
            print(f"Error getting dynamic action masks: {e}")
            # Return all False masks on error
            return np.zeros((self.num_envs, self.number_of_reg_genes), dtype=bool)
        # print("Dynamic action masks calculated with shape: ", masks.shape)
        return masks

    def _calculate_distances(self) -> np.ndarray:
        # Ensure returns shape (num_envs,)
        current_vecs = self._get_current_expression_vector(self.current_cell_indices)
        # TODO THIS WILL NOT WORK CURRENTLY AS FOR THE TARGETS WE NEED TO GET THE TARGET CELLTYPE OR SOMETHING WITH THE DISTANCE TO THE AVERAGE CELL IN A CELLTYPE
        target_vecs = None
        # Handle potential shape mismatch if _get_current_expression_vector returned zeros
        if current_vecs.shape[0] != self.num_envs: current_vecs = np.zeros((self.num_envs, self.number_of_reg_genes),
                                                                           dtype=np.float32)
        if target_vecs.shape[0] != self.num_envs: target_vecs = np.zeros((self.num_envs, self.number_of_reg_genes),
                                                                         dtype=np.float32)

        distances = np.ones(self.num_envs, dtype=np.float32)
        norms_current = np.linalg.norm(current_vecs, axis=1)
        norms_target = np.linalg.norm(target_vecs, axis=1)
        valid_mask = (norms_current > 1e-8) & (norms_target > 1e-8)
        if np.any(valid_mask):
            dot_products = np.einsum('ij,ij->i', current_vecs[valid_mask], target_vecs[valid_mask])
            similarities = dot_products / (norms_current[valid_mask] * norms_target[valid_mask])
            similarities = np.clip(similarities, -1.0, 1.0)
            distances[valid_mask] = 1.0 - similarities
        return distances

    ### OLD REWARD METHOD###

    # rewards = np.zeros(self.num_envs, dtype=np.float32)
    #         dones = np.zeros(self.num_envs, dtype=bool)
    #         terminations_debug = np.zeros(self.num_envs, dtype=bool) # For info dict
    #         truncations_debug = np.zeros(self.num_envs, dtype=bool) # For info dict
    #
    #         rewards[simulation_failed] = self.fail_penalty
    #         dones[simulation_failed] = True
    #         terminations_debug[simulation_failed] = True
    #
    #         valid_envs = ~simulation_failed
    #         current_distances = self._calculate_distances()
    #
    #         if np.any(valid_envs):
    #             rewards[valid_envs] = self.step_penalty + (self.distance_reward_scale * -current_distances[valid_envs])
    #             reached_goal_mask = current_distances[valid_envs] <= self.target_distance_threshold
    #             goal_envs_global_indices = np.where(valid_envs)[0][reached_goal_mask]
    #             dones[goal_envs_global_indices] = True
    #             terminations_debug[goal_envs_global_indices] = True
    #             rewards[goal_envs_global_indices] += self.goal_bonus
    #
    #             ran_out_of_steps_mask = self.current_steps[valid_envs] >= self.max_steps
    #             timeout_mask_local = ran_out_of_steps_mask & (~dones[valid_envs])
    #             timeout_global_indices = np.where(valid_envs)[0][timeout_mask_local]
    #             dones[timeout_global_indices] = True
    #             truncations_debug[timeout_global_indices] = True
    #             rewards[timeout_global_indices] += self.fail_penalty
    #
    #         # 6. Check Stuck State
    #         next_action_masks = self._get_dynamic_action_masks() # Shape (num_envs, num_actions)
    #         stuck_mask = ~np.any(next_action_masks, axis=1)
    #         apply_stuck_penalty_mask = stuck_mask & (~dones)
    #         dones[apply_stuck_penalty_mask] = True
    #         truncations_debug[apply_stuck_penalty_mask] = True # Mark as truncation (stuck)
    #         rewards[apply_stuck_penalty_mask] = self.fail_penalty
    #         if np.any(apply_stuck_penalty_mask):
    #             stuck_indices = np.where(apply_stuck_penalty_mask)[0]
    #             print(f"Warning: Environments {stuck_indices} are stuck. Truncating.")
    #
    #
    #         # 7. Get Observations & Infos
    #         observations = self._get_obs()
    #         infos = []
    #         for i in range(self.num_envs):
    #             info_dict = {
    #                 "action_mask": next_action_masks[i],
    #                 "distance": current_distances[i],
    #             }
    #             if dones[i]:
    #                  # Standard keys for SB3 Monitor wrapper / Gymnasium v26+
    #                  info_dict["final_observation"] = {k: v[i] for k, v in observations.items()}
    #                  info_dict["final_info"] = {
    #                       "distance": current_distances[i],
    #                       "steps": self.current_steps[i],
    #                       "goal_reached": terminations_debug[i] and not simulation_failed[i],
    #                       "timeout": truncations_debug[i] and not (stuck_mask[i] and apply_stuck_penalty_mask[i]),
    #                       "stuck": stuck_mask[i] and apply_stuck_penalty_mask[i],
    #                       "simulation_failed": simulation_failed[i]
    #                  }
    #                  # Add 'TimeLimit.truncated=False' if termination is due to goal, True otherwise (timeout/stuck)
    #                  # This helps Monitor distinguish termination causes.
    #                  info_dict["TimeLimit.truncated"] = truncations_debug[i] and not terminations_debug[i]
    #                  # Also add terminal observation key
    #                  info_dict["terminal_observation"] = info_dict["final_observation"]
    #
    #             infos.append(info_dict)