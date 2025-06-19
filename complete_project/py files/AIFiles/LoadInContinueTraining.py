import os
import pickle
import torch.nn as nn # For nn.ReLU if used in config
from stable_baselines3 import PPO # Or your specific agent like MaskablePPO
import wandb
import datetime
from Env import CellOracleSB3VecEnv
from AI import anotherCallBackForPhaseTracking, customWandBCallback, CustomCheckpointCallbackWithStates


def run_resumed_training(config: dict):
    """
    Loads a checkpointed model and its associated states, then continues training.
    """

    if not config.get("LOAD_MODEL_PATH") or not os.path.exists(config["LOAD_MODEL_PATH"]):
        print(f"Error: LOAD_MODEL_PATH '{config.get('LOAD_MODEL_PATH')}' not specified or does not exist.")
        return

    os.makedirs(config["LOG_DIR"], exist_ok=True)
    os.makedirs(config["MODEL_SAVE_PATH"], exist_ok=True)

    model_zip_path = config["LOAD_MODEL_PATH"]
    model_dir_load = os.path.dirname(model_zip_path)


    
    model_filename_no_ext_load = os.path.splitext(os.path.basename(model_zip_path))[0]
    base_path_for_states_load = os.path.join(model_dir_load, model_filename_no_ext_load)

    loaded_curriculum_state = None
    curriculum_state_path_load = f"{base_path_for_states_load}_curriculum_state.pkl"
    if os.path.exists(curriculum_state_path_load):
        try:
            with open(curriculum_state_path_load, "rb") as f:
                loaded_curriculum_state = pickle.load(f)
            print(f"Successfully loaded curriculum state from: {curriculum_state_path_load}")
        except Exception as e:
            print(f"Error loading curriculum state from {curriculum_state_path_load}: {e}")
            loaded_curriculum_state = None # Proceed without it if loading fails
    else:
        print(f"Warning: Curriculum state file not found at {curriculum_state_path_load}.")

    loaded_env_state = None
    wandb_run_id_to_resume=None
    env_state_path_load = f"{base_path_for_states_load}_env_state.pkl"
    if os.path.exists(env_state_path_load):
        try:
            with open(env_state_path_load, "rb") as f:
                loaded_env_state = pickle.load(f)
            if loaded_env_state and "wandb_run_id" in loaded_env_state:
                wandb_run_id_to_resume = loaded_env_state["wandb_run_id"]
                print(f"Resuming from WandB run ID: {wandb_run_id_to_resume}")
            print(f"Successfully loaded environment state from: {env_state_path_load}")
        except Exception as e:
            print(f"Error loading environment state from {env_state_path_load}: {e}")
            loaded_env_state = None # Proceed without it
    else:
        print(f"Warning: Environment state file not found at {env_state_path_load}.")

    initial_max_steps = config.get("MAX_STEPS_PER_EPISODE", 50) # A sensible default
    if loaded_env_state and "max_steps" in loaded_env_state:
        initial_max_steps = loaded_env_state["max_steps"]
        print(f"Using max_steps from loaded environment state: {initial_max_steps}")
    else:
        print(f"Using initial max_steps from config: {initial_max_steps}")


    env = CellOracleSB3VecEnv(
        oracle_path=config["ORACLE_PATH"],
        batch_size=config["BATCH_SIZE"],
        max_steps=initial_max_steps,
        gene_activity_threshold=config["GENE_ACTIVITY_THRESHOLD"],
        target_distance_threshold=config["TARGET_DISTANCE_THRESHOLD"],
        step_penalty=config["STEP_PENALTY"],
        goal_bonus=config["GOAL_BONUS"],
        fail_penalty=config["FAIL_PENALTY"],
        distance_reward_scale=config["DISTANCE_REWARD_SCALE"],
        allow_gene_activation=config["ALLOW_GENE_ACTIVATION"],
        gamma_distance=config.get("GAMMA_DISTANCE_DISCOUNTER", 0.99), # PBRS gamma
        number_of_targets_curriculum = config.get("TARGET_CELLS_PER_PHASE", 4),
        use_prev_perturbs=config.get("USE_PREV_KNOCKOUT", False),
        same_cell_penalty=config.get("SAME_CELL_PENALTY", -0.1),
        standard_sd_factor=config.get("STANDARD_SD_FACTOR", 1),
        use_similarity = config.get("USE_SIMILARITY_REWARD", False),
    )
    wandb_id = wandb_run_id_to_resume if wandb_run_id_to_resume else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if loaded_env_state:
        print("loading in custom state")
        env.set_env_state(loaded_env_state)
    else:
        env.set_env_state_at_manually(current_phase=5, max_steps=84, wandb_id=wandb_id)


    phase_callback = anotherCallBackForPhaseTracking(
        steps_for_phase_change=config["TOTAL_TIMESTEPS"] // config["DIVIDER_OF_TOTAL_STEPS_FOR_CURRICULUM_LEARNING"],
        max_step_increase=config["MAX_STEP_INCREASE_PER_PHASE"],
        steps_per_phase_increase=config["PHASE_STEP_INCREASE"],
        verbose=config.get("CALLBACK_VERBOSE", 1)
    )
    # set state if loaded_curriculum_state saved
    if loaded_curriculum_state:
        phase_callback.set_state(loaded_curriculum_state)
        print("Applied loaded state to the curriculum callback.")


    custom_checkpoint_callback = CustomCheckpointCallbackWithStates(
        save_freq=config["STEP_SAVE_FREQ"],
        save_path=config['MODEL_SAVE_PATH'], # New save path for this run's checkpoints
        name_prefix=config.get("NEW_CHECKPOINT_PREFIX", "ppo_resumed"),
        curriculum_callback=phase_callback,
        verbose=config.get("CALLBACK_VERBOSE", 1)
    )

    # 3. WandB Callback (and others)
    callbacks_list = [custom_checkpoint_callback, phase_callback]
    wandb_run_name = config.get("WANDB_RUN_NAME", "resumed_run") + f"_from_{model_filename_no_ext_load}"
    wandb.init(
        project="celloracle",
        config=config,
        name=wandb_run_name,
        sync_tensorboard=True,
        resume="allow",
        id=wandb_id
    )
    wandb_callback = customWandBCallback(verbose=config.get("CALLBACK_VERBOSE", 1))
    callbacks_list.append(wandb_callback)


    # --- Load SB3 Model ---
    print(f"Loading SB3 model from: {model_zip_path}...")
    # Define policy_kwargs in case they are needed for loading with a new SB3 version
    # or if custom objects are involved, though often not strictly necessary for PPO.
    policy_kwargs = dict(
        net_arch=dict(
            pi=config.get("PI_ARCH", [64, 64]),
            vf=config.get("VF_ARCH", [64, 64])
        ),
        activation_fn=config.get("ACTIVATION_FN", nn.ReLU) # Make sure nn.ReLU is imported or defined
    )
    model = PPO.load(
        model_zip_path,
        env=env,
        device=config.get("DEVICE", "auto"),
        # policy_kwargs=policy_kwargs, # Usually not needed if loading a full model unless major changes
        # learning_rate=config.get("LEARNING_RATE"), # Optimizer state is loaded, so LR should resume.
                                                     # Only set if you explicitly want to override or start a new schedule.
        tensorboard_log=config["LOG_DIR"] # Point to the new log directory
    )
    print(f"Model loaded. Current num_timesteps: {model.num_timesteps}")
    if config.get("USE_WANDB_CALLBACK", True) and config.get("WANDB_WATCH_MODEL", True):
        wandb.watch(model.policy, log=config.get("WANDB_WATCH_LOG_TYPE", "all"), log_freq=config.get("WANDB_WATCH_FREQ", 1000))


    # --- Continue Training ---
    # TOTAL_TIMESTEPS in config should be the *new target total*.
    # model.learn() will run for TOTAL_TIMESTEPS - model.num_timesteps.
    print(f"Continuing training. Current timesteps: {model.num_timesteps}. Target total: {config['TOTAL_TIMESTEPS']}")
    if model.num_timesteps >= config['TOTAL_TIMESTEPS']:
        print("Model already trained to or beyond the target total timesteps. No further training will occur.")
    else:
        model.learn(
            total_timesteps=config["TOTAL_TIMESTEPS"],
            log_interval=config.get("LOG_INTERVAL", 1),
            callback=callbacks_list,
            reset_num_timesteps=False # CRUCIAL for resuming!
        )
        print("Resumed training finished.")

        # --- Save Final Model and States from the resumed run ---
        final_save_name_base = os.path.join(config['MODEL_SAVE_PATH'], f"{config.get('NEW_CHECKPOINT_PREFIX', 'ppo_resumed')}_final_{model.num_timesteps}_steps")
        model.save(f"{final_save_name_base}.zip")
        print(f"Saved final resumed model to {final_save_name_base}.zip")
        try:
            final_env_state = env.get_env_state()
            with open(f"{final_save_name_base}_env_state.pkl", "wb") as f: pickle.dump(final_env_state, f)
            final_curriculum_state = phase_callback.get_state()
            with open(f"{final_save_name_base}_curriculum_state.pkl", "wb") as f: pickle.dump(final_curriculum_state, f)
            print(f"Saved final environment and curriculum states for the resumed run.")
        except Exception as e:
            print(f"Error saving final states for resumed run: {e}")


    env.close()
    if config.get("USE_WANDB_CALLBACK", True): wandb.finish()
    print("--- Resumed Training Run Finished ---")


if __name__ == "__main__":
    # --- Define Configuration for Resuming ---
    # It's good practice to load this from a file or command-line arguments
    # For simplicity, defined here:
    resume_config = {
        # --- Paths ---
        "ORACLE_PATH": os.path.join('../celloracle_data', "celloracle_object/new_promoter_without_mescs_trimmed_test_own_umap", "ready_oracle.pkl"),
        # ** CRITICAL: Path to the model.zip you want to load and resume from **
        "LOAD_MODEL_PATH": os.path.join("../celloracle_data", "models", "ppo_version_3_5/8/2025", "ppo_model_custom_state_10_steps.zip"), # !! UPDATE THIS !!

        # ** CRITICAL: New paths for saving checkpoints and logs for THIS resumed run **
        "MODEL_SAVE_PATH": os.path.join("../celloracle_data", "models", "ppo_version_3_5/8/2025_resumed_run1"),
        "LOG_DIR": os.path.join('../celloracle_data', "celloracle_object/new_promoter_without_mescs_trimmed_test_own_umap', 'logs_notebook_exp1_resumed_run1"),
        "NEW_CHECKPOINT_PREFIX": "ppo_resumed_checkpoint", # Prefix for new checkpoints

        # --- Training Setup ---
        "BATCH_SIZE": 256,
        "TOTAL_TIMESTEPS": 2000000,   # The *new* desired total timesteps (e.g., original_total + more)
        "VERBOSE": 1,
        "LOG_INTERVAL": 1,
        "RESET_NUM_TIMESTEPS": False, # MUST BE FALSE FOR RESUMING
        "DEVICE": "auto",
        "STEP_SAVE_FREQ": 50000, # How often to save new checkpoints during this resumed run

        # --- PPO Hyperparameters (should generally match the original run unless you're intentionally changing them) ---
        "LEARNING_RATE": 3e-4, # If not using a schedule, or if you want to reset/change it
        "PPO_N_STEPS": 256,
        "PPO_N_EPOCHS": 6,
        "GAMMA": 0.99, # MDP Gamma
        "GAE_LAMBDA": 0.95,
        "CLIP_RANGE": 0.2,
        "ENT_COEF": 0.02,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "USE_LEARNING_RATE_SCHEDULE": False, # If you had one, ensure it resumes correctly or re-initialize

        # --- Environment Specific (should match the state being loaded or desired new state) ---
        "MAX_STEPS_PER_EPISODE_INITIAL": 54, # Initial value if no env_state is loaded
        "ALLOW_GENE_ACTIVATION": True,
        "STANDARD_SD_FACTOR": 1.5,
        "USE_SIMILARITY_REWARD": True,
        "DIVIDER_OF_TOTAL_STEPS_FOR_CURRICULUM_LEARNING": 10,
        "TARGET_CELLS_PER_PHASE": 2,
        "MAX_STEP_INCREASE_PER_PHASE": 6,
        "PHASE_STEP_INCREASE": 10000,
        "GENE_ACTIVITY_THRESHOLD": 0.01,
        "TARGET_DISTANCE_THRESHOLD": 0.1,
        "GAMMA_DISTANCE_DISCOUNTER": 1.0, # PBRS Gamma (e.g., if you changed it)
        "STEP_PENALTY": -0.07,
        "GOAL_BONUS": 7,
        "FAIL_PENALTY": -1,
        "SAME_CELL_PENALTY": -0.5,
        "DISTANCE_REWARD_SCALE": 50,
        "USE_PREV_KNOCKOUT": True,
        "ENV_VERBOSE": 1, # For environment's print statements
        "CALLBACK_VERBOSE": 1, # For callbacks' print statements


        # --- NN Architecture (must match the loaded model) ---
        "PI_ARCH": [512,256,216],
        "VF_ARCH": [512,256,216],
        "ACTIVATION_FN": nn.ReLU, # Make sure this matches (PPO.load usually handles this if a full model is saved)

        # --- WandB ---
        "USE_WANDB_CALLBACK": True,
        "WANDB_PROJECT": "celloracle_curriculum", # Your project name
        "WANDB_RUN_NAME": "ppo_v3.5_resumed",     # Name for this specific run
        "WANDB_WATCH_MODEL": True,
        "WANDB_WATCH_FREQ": 2048 # Every N steps
        # "WANDB_RUN_ID": None # Set this if you want to resume a specific wandb run online
    }

    run_resumed_training(resume_config)