import os
from datetime import datetime

# from platform import architecture # Unused import
import gymnasium as gym
# from Environment import CellOracleEnv as CellOracleCustomEnv
from typing import Callable
import optuna
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # Not strictly needed here
from Env import CellOracleSB3VecEnv as CellOracleCustomEnv
from Env_with_new_input import CellOracleSB3VecEnv as CellOracleCustomEnvNewInput
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker # Wrapper is needed!
import wandb
from gymnasium.vector import VectorEnv
from gymnasium import spaces
import numpy as np
from scipy.sparse import lil_matrix,issparse
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import os
import pickle

def run_training(config: dict):
    """
    Sets up the environment, agent, and runs the training loop based on the provided config.

    Args:
        config (dict): A dictionary containing all necessary parameters.
    """
    #init weights and biases
    wandbd_run_id  = config.get("WANDB_RUN_ID", None)
    #set wandname to the current date and parameter settings (rewards)
    wand_name  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+ str(config["GOAL_BONUS"]) + "_" + str(config["FAIL_PENALTY"]) + "_" + str(config["STEP_PENALTY"])+ "_" + str(config["DISTANCE_REWARD_SCALE"]) + "_" + str(config["ALLOW_GENE_ACTIVATION"])
    wandb.init(
        project="celloracle",
        config=config,
        name=wand_name,
        sync_tensorboard=True,
        id=wandbd_run_id,
        reinit=True,
    )

    print("\n--- Starting Training Run ---")
    print(f"Configuration: {config}")

    # --- Create Log Directory if it doesn't exist ---
    os.makedirs(config["LOG_DIR"], exist_ok=True)
    print(f"TensorBoard Log Directory: {config['LOG_DIR']}")

    # --- Input Validation ---
    if not os.path.exists(config["ORACLE_PATH"]):
         raise FileNotFoundError(f"Oracle file not found at: {config['ORACLE_PATH']}.")

    # --- Instantiate the Environment ---
    print(f"Initializing CellOracleCustomEnv with batch_size={config['BATCH_SIZE']}...")
    actual_wandb_run_id = wandb.run.id if wandb.run else None
    if not config.get("USE_NEW_INPUT", False):
        env = CellOracleCustomEnv(
            oracle_path=config["ORACLE_PATH"],
            batch_size=config["BATCH_SIZE"],
            max_steps=config["MAX_STEPS_PER_EPISODE"],
            step_penalty=config["STEP_PENALTY"],
            goal_bonus=config["GOAL_BONUS"],
            fail_penalty=config["FAIL_PENALTY"],
            distance_reward_scale=config["DISTANCE_REWARD_SCALE"],
            allow_gene_activation=config["ALLOW_GENE_ACTIVATION"],
            gamma_distance=config.get("DISTANCE_GAMMA", 0.99),
            number_of_targets_curriculum = config.get("TARGET_CELLS_PER_PHASE", 4),
            use_prev_perturbs=config.get("USE_PREV_KNOCKOUT", False),
            same_cell_penalty=config.get("SAME_CELL_PENALTY", -0.1),
            standard_sd_factor=config.get("STANDARD_SD_FACTOR", 1),
            use_similarity = config.get("USE_SIMILARITY_REWARD", False),
        )
    else:
        env = CellOracleCustomEnvNewInput(
            oracle_path=config["ORACLE_PATH"],
            batch_size=config["BATCH_SIZE"],
            max_steps=config["MAX_STEPS_PER_EPISODE"],
            step_penalty=config["STEP_PENALTY"],
            goal_bonus=config["GOAL_BONUS"],
            fail_penalty=config["FAIL_PENALTY"],
            distance_reward_scale=config["DISTANCE_REWARD_SCALE"],
            allow_gene_activation=config["ALLOW_GENE_ACTIVATION"],
            gamma_distance=config.get("DISTANCE_GAMMA", 0.99),
            number_of_targets_curriculum = config.get("TARGET_CELLS_PER_PHASE", 4),
            use_prev_perturbs=config.get("USE_PREV_KNOCKOUT", False),
            same_cell_penalty=config.get("SAME_CELL_PENALTY", -0.1),
            standard_sd_factor=config.get("STANDARD_SD_FACTOR", 1),
            use_similarity = config.get("USE_SIMILARITY_REWARD", False),
        )
    env.set_wandb_id(actual_wandb_run_id,wand_name)
    # env = ActionMasker(env, action_mask_fn=lambda env: env.action_masks())

    #because ppo and stable baselinses expects a gym.env instance and not a vecenv instance, we need to do some workarounds..
  #  env = DummyVecEnv([lambda: vec_env]) # Wrap the custom environment in a DummyVecEnv
    

    AgentClass = PPO #CAN NOT USE MASKABLEPPO BECAUSE THE AUTHORS CAN NOT CODE

    # Define policy_kwargs based on config
    policy_kwargs = dict(
        net_arch=dict(
            pi=config.get("PI_ARCH", [64, 64]),
            vf=config.get("VF_ARCH", [64, 64])
        ),
        activation_fn=config.get("ACTIVATION_FN", nn.Tanh)
    )
    print(f"Policy architecture: {policy_kwargs}")
    learning_rate = config.get("LEARNING_RATE", 3e-4)
    if config.get("USE_LEARNING_RATE_SCHEDULE"):
        scheduler = phaseScheduler(
            initial_value=learning_rate,
            end_value=1e-5,
            total_training_timesteps=config["TOTAL_TIMESTEPS"],
            initial_phase_duration=config["MAX_STEPS_FIRST_PHASE"],
            phase_duration_increase=config["TARGET_CELLS_PER_PHASE"],
            num_total_curriculum_phases=len(env.total_curriculum_targets)
        )
    else:
        scheduler = learning_rate

    #model callback
    os.makedirs(config['MODEL_SAVE_PATH'], exist_ok=True)


    model = AgentClass(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=config.get("VERBOSE", 1),
        learning_rate=scheduler,
        n_steps=config["PPO_N_STEPS"],
        batch_size=config["BATCH_SIZE"],
        n_epochs=config["PPO_N_EPOCHS"],
        gamma=config.get("GAMMA", 0.99),
        gae_lambda=config.get("GAE_LAMBDA", 0.95),
        clip_range=config.get("CLIP_RANGE", 0.2),
        ent_coef=config.get("ENT_COEF", 0.0),
        vf_coef=config.get("VF_COEF", 0.5),
        max_grad_norm=config.get("MAX_GRAD_NORM", 0.5),
        tensorboard_log=config["LOG_DIR"],
        device=config.get("DEVICE", "auto")
    )
    wandb.watch(model.policy, log="gradients", log_freq=1)  # Log gradients to WandB

    print(f"Agent ({AgentClass.__name__}) instantiated with architecture: {policy_kwargs['net_arch']}")

    # --- Train ---
    print(f"Starting training for {config['TOTAL_TIMESTEPS']} timesteps...")
    # Callbacks list is now empty or contains other callbacks if needed
    callbacks =[]
    wandb_callback = customWandBCallback()
    phase_callback = anotherCallBackForPhaseTracking(intial_max_for_first_phase = config["MAX_STEPS_FIRST_PHASE"], max_step_increase=config["MAX_STEP_INCREASE_PER_PHASE"], steps_per_phase_increase=config["PHASE_STEP_INCREASE"],hpo_disable_curriculum=  config.get("PARAMETER_TUNING", False))
    customCheckpointCallback = CustomCheckpointCallbackWithStates(save_freq=config["STEP_SAVE_FREQ"], save_path=config['MODEL_SAVE_PATH'], name_prefix="ppo_model", curriculum_callback=phase_callback, save_replay_buffer=False, save_vecnormalize=False, verbose=1, config=config)
    optuna_trial_obj = config.get("optuna_trial_obj", None)
    optunaCallback = None
    if optuna_trial_obj is not None:
        optunaCallback = SB3OptunaPruningCallback(trial=optuna_trial_obj, metric_name="rollout/ep_rew_mean",
                                                  report_interval_steps=config.get("OPTUNA_REPORT_INTERVAL", 1000),
                                                  verbose=1)
    if optunaCallback is None:
        callbacks.extend([customCheckpointCallback,wandb_callback, phase_callback])
    else:
        callbacks.extend([customCheckpointCallback,wandb_callback, phase_callback, optunaCallback])


    try:
        model.learn(
            total_timesteps=config["TOTAL_TIMESTEPS"],
            log_interval=config.get("LOG_INTERVAL", 1),
            callback=callbacks,  # Pass None or list of other callbacks
            reset_num_timesteps=config.get("RESET_NUM_TIMESTEPS", False)
        )
    except optuna.exceptions.TrialPruned:
        print("Trial pruned during model.learn(). Exiting training for this trial.")

    print("Training finished.")

    # --- Save ---
    save_path = f"{config['MODEL_SAVE_PATH']}.zip"
    print(f"Saving trained model to {save_path}...")
    model.save(config['MODEL_SAVE_PATH'])
    print("Model saved.")

    # --- Close ---
    env.close()
    print("Environment closed.")
    print("--- Training Run Finished ---")

    return model # Return the trained model

def linear_schedule(initial_value: float, end_value: float = 1e-6, start_steps_first_phase:int=50000, increase_steps_per_phase:int=20000, number_of_celltypes:int = 26, increase_target:int = 3) -> Callable[[float], float]:
    """
    Linear learning rate schedule. Decreases from initial_value to end_value.

    :param initial_value: Initial learning rate.
    :param end_value: Final learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    # Ensure end_value is smaller than initial_value
    if end_value >= initial_value:
        end_value = initial_value * 0.01 # Default to 1% if end_value is invalid
        print(f"Warning: end_value >= initial_value in linear_schedule. Setting end_value to {end_value:.2e}")

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        """
        return end_value + progress_remaining * (initial_value - end_value)

    return func

class phaseScheduler:
    def __init__(self,
                 initial_value: float,
                 end_value: float,
                 total_training_timesteps: int,  # config["TOTAL_TIMESTEPS"]
                 initial_phase_duration: int,
                 # Calculated: config["TOTAL_TIMESTEPS"] // config["DIVIDER_OF_TOTAL_STEPS_FOR_CURRICULUM_LEARNING"]
                 phase_duration_increase: int,  # config["PHASE_STEP_INCREASE"]
                 num_total_curriculum_phases: int  # Derived from len(env.total_curriculum_targets)
                 ):
        self.initial_value = initial_value
        self.end_value = end_value
        self.total_training_timesteps = total_training_timesteps
        self.initial_phase_duration = initial_phase_duration
        self.phase_duration_increase = phase_duration_increase
        self.num_total_curriculum_phases = num_total_curriculum_phases
        if self.end_value >= self.initial_value:
            self.end_value = self.initial_value * 0.01

        self.timesteps_for_curriculum_completion = self._calculate_curriculum_completion_steps()

    def _calculate_curriculum_completion_steps(self) -> int:
        """
        Calculates the total number of timesteps consumed by all defined curriculum phases.
        This matches the logic in `anotherCallBackForPhaseTracking` for how phase durations evolve.
        """
        timesteps_consumed_by_curriculum = self.initial_phase_duration

        for phase_num in range(1, self.num_total_curriculum_phases + 1):
            if phase_num < self.num_total_curriculum_phases:
                timesteps_consumed_by_curriculum += self.phase_duration_increase

        return timesteps_consumed_by_curriculum*1.5#give it some margin to learn more and then diall in

    def __call__(self, progress_remaining: float) -> float:
        """
        Callable method that SB3 will use.
        progress_remaining decreases from 1.0 (start of total_training_timesteps) to 0.0 (end).
        """
        # Calculate current absolute timesteps based on overall progress
        current_absolute_timesteps = self.total_training_timesteps * (1.0 - progress_remaining)
        current_absolute_timesteps = round(current_absolute_timesteps)  # Round to nearest int for comparison

        if current_absolute_timesteps < self.timesteps_for_curriculum_completion+ self.initial_phase_duration*5:
            # We are still within the timesteps allocated for curriculum phases
            return self.initial_value
        else:
            # Curriculum phases are notionally complete; start LR decay.
            # Calculate how many steps are dedicated to the decay period.
            decay_period_total_steps = self.total_training_timesteps - self.timesteps_for_curriculum_completion

            if decay_period_total_steps <= 0:
                # This means curriculum completion is at or after total training steps.
                # No decay period, or we are already at the end.
                return self.end_value  # Or self.initial_value if no decay was ever intended

            timesteps_into_decay_period = current_absolute_timesteps - self.timesteps_for_curriculum_completion


            progress_in_decay_period = timesteps_into_decay_period / decay_period_total_steps

            progress_remaining_in_decay = 1.0 - progress_in_decay_period

            progress_remaining_in_decay = max(0.0, min(1.0, progress_remaining_in_decay))

            current_lr = self.end_value + progress_remaining_in_decay * (self.initial_value - self.end_value)
            return current_lr

class customWandBCallback(BaseCallback):
    """
    Custom callback to log additional metrics to WandB.
    """
    def __init__(self,verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        #for now only rewards, later more if necessary, PROBABLY LOG THE ACTUAL WEIGHTS FROM TIME TO TIME
        mean_reward = np.mean(self.locals['rewards'])

        entropy_loss_value = self.logger.name_to_value.get('train/entropy_loss')

        if entropy_loss_value is not None:
            # Optional: Add scalar conversion if necessary (as discussed before)
            if hasattr(entropy_loss_value, 'item'):
                scalar_entropy_loss = entropy_loss_value.item()
            else:
                scalar_entropy_loss = float(entropy_loss_value)

            # Log to WandB, including the step
            wandb.log({"training/entropy_loss": scalar_entropy_loss})
        wandb.log({"training/mean_reward": mean_reward})
        wandb.log({"training/step": self.num_timesteps})  # Log the current step

        return True

class CustomCheckpointCallbackWithStates(CheckpointCallback):
    def __init__(self,
                 save_freq: int,
                 save_path: str,
                 name_prefix: str = "rl_model",
                 curriculum_callback: Optional[Any] = None, # Type hint as Any or the specific class
                 save_replay_buffer: bool = False,
                 save_vecnormalize: bool = False,
                 verbose: int = 0,
                 config: Optional[dict] = None):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.curriculum_callback = curriculum_callback
        # Ensure the save_path exists
        os.makedirs(self.save_path, exist_ok=True)
        if config is not None:
            self.config = config

    def _on_step(self) -> bool:
        continue_training = super()._on_step() #pretty sure this saves the model

        if self.n_calls > 0 and self.n_calls % self.save_freq == 0:
            model_filename_base = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            env_state = self.training_env.get_env_state()  # Directly call on VecEnv
            env_state_path = f"{model_filename_base}_env_state.pkl"
            with open(env_state_path, "wb") as f:
                pickle.dump(env_state, f)

            curriculum_state = self.curriculum_callback.get_state()
            curriculum_state_path = f"{model_filename_base}_curriculum_state.pkl"
            with open(curriculum_state_path, "wb") as f:
                pickle.dump(curriculum_state, f)
            if hasattr(self, 'config'):
                config_path = f"{model_filename_base}_config.pkl"
                with open(config_path, "wb") as f:
                    pickle.dump(self.config, f)


        return continue_training

class SB3OptunaPruningCallback(BaseCallback):
    """
    Stable Baselines3 callback for Optuna pruning.
    Stops training early if Optuna's pruner suggests it.
    """
    def __init__(self, trial: optuna.Trial, metric_name: str = "rollout/ep_rew_mean",
                 report_interval_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.trial = trial
        self.metric_name = metric_name # Metric to report to Optuna
        self.is_pruned = False
        self.report_interval_steps = report_interval_steps # How often to report to Optuna
        self.last_report_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_report_step < self.report_interval_steps:
            return True
        self.last_report_step = self.num_timesteps
        # Try to get the latest logged value for the specified metric
        latest_metric_value = self.model.logger.name_to_value.get(self.metric_name)
        if latest_metric_value is  None:
            return True # Continue training if metric is not available
        self.trial.report(float(latest_metric_value), self.num_timesteps)
        if self.trial.should_prune():
            print("pruning!!")
            self.is_pruned = True
            raise optuna.exceptions.TrialPruned()
        return True # Continue training

class anotherCallBackForPhaseTracking(BaseCallback):
    """
    Custom callback to track the phase of training.
    """
    def __init__(self, intial_max_for_first_phase,verbose=0, log_freq=2, max_step_increase:int =4, steps_per_phase_increase:int = 10000, hpo_disable_curriculum:bool = False):

        super().__init__(verbose)
        self.max_step_increase = max_step_increase
        self.steps_per_phase_increase = steps_per_phase_increase
        self.current_phase = 1
        self.timesteps_at_current_phase_start = 0  # Timestep count when the current phase began
        self.steps_required_for_current_phase = intial_max_for_first_phase
        self.initial_steps_for_phase_change = intial_max_for_first_phase
        self.log_freq = log_freq
        self.same_cell_event_count_period = 0
        self.n_calls_for_logging_period = 0
        # episodeinfo
        self.ep_count_period = 0
        self.ep_goal_success_count_period = 0
        self.ep_len_sum_period = 0
        self.total_ep_count = 0
        self.total_goal_success_count = 0
        self.hpo_disable_curriculum = hpo_disable_curriculum # Store it


    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the callback for saving."""
        state = {
            "current_phase": self.current_phase,
            "timesteps_at_current_phase_start": self.timesteps_at_current_phase_start,
            "steps_required_for_current_phase": self.steps_required_for_current_phase,
            "initial_steps_for_phase_change": self.initial_steps_for_phase_change,  # Save this too
            "ep_count_period": self.ep_count_period,
            "ep_goal_success_count_period": self.ep_goal_success_count_period,
            "ep_len_sum_period": self.ep_len_sum_period,
            "total_ep_count": self.total_ep_count,
            "total_goal_success_count": self.total_goal_success_count,
            "n_calls_snapshot": self.n_calls,
            "same_cell_event_count_period": self.same_cell_event_count_period,
            "n_calls_for_logging_period ": self.n_calls_for_logging_period,
        }
        return state

    def set_state(self, state: Dict[str, Any]):
        """Sets the state of the callback from a loaded state."""
        if self.verbose > 0:
            print(f"PhaseCallback: Attempting to set state from - {state}")
        self.current_phase = state.get("current_phase", 1)
        self.timesteps_at_current_phase_start = state.get("timesteps_at_current_phase_start", 0)
        self.initial_steps_for_phase_change = state.get("initial_steps_for_phase_change",self.initial_steps_for_phase_change)
        self.steps_required_for_current_phase = state.get("steps_required_for_current_phase",self.initial_steps_for_phase_change)
        self.ep_count_period = state.get("ep_count_period", 0)
        self.ep_goal_success_count_period = state.get("ep_goal_success_count_period", 0)
        self.ep_len_sum_period = state.get("ep_len_sum_period", 0.0)
        self.total_ep_count = state.get("total_ep_count", 0)
        self.total_goal_success_count = state.get("total_goal_success_count", 0)
        self.n_calls = state.get("n_calls_snapshot", 0)  # Set n_calls from the loaded state
        self.same_cell_event_count_period = state.get("same_cell_event_count_period", 0)
        self.n_calls_for_logging_period = state.get("n_calls_for_logging_period", 0)




    def _on_step(self) -> bool:
        #think i cna just acces the env like this
        timestep_current_phase = self.num_timesteps - self.timesteps_at_current_phase_start
        if timestep_current_phase >= self.steps_required_for_current_phase and not self.hpo_disable_curriculum:
            # Update the phase
            self.timesteps_at_current_phase_start = self.num_timesteps
            # Increase the steps required for the next phase
            if self.training_env.set_phase(self.current_phase, self.max_step_increase):
                self.current_phase += 1
                self.steps_required_for_current_phase += self.steps_per_phase_increase
            # Log the new phase to WandB
            wandb.log({"training/current_phase": self.current_phase})
            wandb.log({"training/steps_required_for_next_phase": self.steps_required_for_current_phase})
        # Log the current phase to WandB
        current_infos = self.locals.get("infos", [])
        if current_infos is None: current_infos = []  # Handle None case
        num_same_cell_events_this_rollout_step = 0 # Renamed for clarity, this is per _on_step call

        for info in current_infos:
            if info is None: continue  # Skip None entries if they occur
            if info.get("diagnostics/same_cell_event_this_step", False):
                num_same_cell_events_this_rollout_step += 1
            final_info = info.get("final_info", None)
            if final_info is None:
                continue
            self.ep_count_period += 1
            self.total_ep_count += 1
            goal_reached = final_info.get("goal_reached", False)
            if goal_reached:
                self.ep_goal_success_count_period += 1
                self.total_goal_success_count += 1

            ep_stats = final_info.get("episode", {})
            self.ep_len_sum_period += ep_stats.get("l", 0)

        self.same_cell_event_count_period += num_same_cell_events_this_rollout_step

        # Log the episode statistics to WandB
        print(self.n_calls)
        if self.n_calls  % self.log_freq == 0:
            log_dict = {}

            #  are these things we can get i should say so according to doct, we try otherwise skip
            sb3_metrics = [
                'train/entropy_loss', 'train/policy_loss', 'train/value_loss',
                'train/approx_kl', 'train/clip_fraction', 'train/explained_variance',
                'rollout/ep_len_mean', 'rollout/ep_rew_mean'  # Get Monitor stats if available
            ]
            for key in sb3_metrics:
                value = self.logger.name_to_value.get(key)
                if value is None:
                    continue
                scalar_value = value.item() if hasattr(value, 'item') else float(value)
                log_key = key.replace('train/', 'update/').replace('rollout/', 'episode/')
                log_dict[log_key] = scalar_value

            if current_infos and current_infos[0] is not None:
                info_step = current_infos[0]
                #my own things i log in info
                step_metrics_keys = [
                    "step_avg/average_goal",
                    "step_avg/average_penalty",
                    "step_avg/average_distance",
                    "step_avg/average_same_cell",
                    "step_avg/percentage_of_activation",
                    "batch_diversity/current_cell_types_unique_in_batch",
                    "batch_diversity/target_cell_types_unique_in_batch"
                ]
                for key in step_metrics_keys:
                    if key not in info_step:
                        continue
                    if key == "batch_diversity/current_cell_types_unique_in_batch":
                        unique_array = info_step.get(key)
                        if unique_array is not None:  # It could be None if current_cell_types was None
                            log_dict["batch_diversity/num_unique_current_cell_types"] = len(unique_array)
                        else:
                            log_dict["batch_diversity/num_unique_current_cell_types"] = 0
                    elif key == "batch_diversity/target_cell_types_unique_in_batch":
                        unique_array = info_step.get(key)
                        if unique_array is not None:
                            log_dict["batch_diversity/num_unique_target_cell_types"] = len(unique_array)
                        else:
                            log_dict["batch_diversity/num_unique_target_cell_types"] = 0
                    else:
                        log_dict[key] = info_step[key]

            log_dict["episode_period/same_cell_event_count"] = self.same_cell_event_count_period
            total_env_steps_this_log_period = self.log_freq * 256 #own batch size
            if total_env_steps_this_log_period > 0:
                log_dict[
                    "episode_period/same_cell_event_rate"] = self.same_cell_event_count_period / total_env_steps_this_log_period
            else:
                log_dict["episode_period/same_cell_event_rate"] = 0.0  # Or np.nan


            if self.ep_count_period > 0:
                success_rate_period = self.ep_goal_success_count_period / self.ep_count_period
                log_dict["episode/success_rate_period"] = success_rate_period

                avg_len_period = self.ep_len_sum_period / self.ep_count_period
                log_dict["episode/ep_len_mean_period"] = avg_len_period

                #log step success rate
                log_dict["episode/step_success_rate"] = self.ep_goal_success_count_period/self.ep_count_period

                # Calculate overall success rate up to this point
                if self.total_ep_count > 0:
                    overall_success_rate = self.total_goal_success_count / self.total_ep_count
                    log_dict["episode/success_rate_overall"] = overall_success_rate

                # Reset counters for the next aggregation period
                self.ep_count_period = 0
                self.ep_goal_success_count_period = 0
                self.ep_len_sum_period = 0
                self.same_cell_event_count_period = 0
                

            # --- Log current phase (ensure it's logged periodically) ---
            log_dict["training/current_phase"] = self.current_phase

            # --- Log collected metrics to WandB ---
            if log_dict:
                wandb.log(log_dict)

        return True



