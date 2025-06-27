import os
import pickle
import torch.nn as nn
from stable_baselines3 import PPO
import wandb
import datetime
from AI import (
    getCorrectEnv,
    WandbLoggingCallback,
    CurriculumCallback,
    CustomCheckpointCallbackWithStates
)


def run_resumed_training(config: dict):
    """
    Loads a checkpointed model and its associated states, then continues training.
    """
    model_zip_path = config.get("LOAD_MODEL_PATH")
    if not model_zip_path or not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Error: LOAD_MODEL_PATH '{model_zip_path}' not specified or does not exist.")

    os.makedirs(config["LOG_DIR"], exist_ok=True)
    os.makedirs(config["MODEL_SAVE_PATH"], exist_ok=True)

    model_dir_load = os.path.dirname(model_zip_path)
    model_filename_no_ext_load = os.path.splitext(os.path.basename(model_zip_path))[0]
    base_path_for_states_load = os.path.join(model_dir_load, model_filename_no_ext_load)

    print(f"--- Starting Resumed Training ---")
    print(f"Loading base model from: {model_zip_path}")

    loaded_curriculum_state = None
    curriculum_state_path_load = f"{base_path_for_states_load}_curriculum_state.pkl"
    if os.path.exists(curriculum_state_path_load):
        try:
            with open(curriculum_state_path_load, "rb") as f:
                loaded_curriculum_state = pickle.load(f)
            print(f"Successfully loaded curriculum state from: {curriculum_state_path_load}")
        except Exception as e:
            print(f"Warning: Could not load curriculum state from {curriculum_state_path_load}: {e}")

    loaded_env_state = None
    wandb_run_id_to_resume = config.get("WANDB_RUN_ID", None)  # Prioritize config ID
    env_state_path_load = f"{base_path_for_states_load}_env_state.pkl"
    if os.path.exists(env_state_path_load):
        try:
            with open(env_state_path_load, "rb") as f:
                loaded_env_state = pickle.load(f)
            # If no ID in config, try getting it from the saved env state
            if not wandb_run_id_to_resume and loaded_env_state and "wandb_run_id" in loaded_env_state:
                wandb_run_id_to_resume = loaded_env_state["wandb_run_id"]
            print(f"Successfully loaded environment state from: {env_state_path_load}")
        except Exception as e:
            print(f"Warning: Could not load environment state from {env_state_path_load}: {e}")

    initial_max_steps = config.get("MAX_STEPS_PER_EPISODE")
    if loaded_env_state and "max_steps" in loaded_env_state:
        initial_max_steps = loaded_env_state["max_steps"]
        print(f"Resuming with max_steps from loaded environment state: {initial_max_steps}")
        config["MAX_STEPS_PER_EPISODE"] = initial_max_steps  # Update config to be consistent


    env_type = config.get("ENV_TYPE")
    if not env_type:
        raise ValueError("`ENV_TYPE` must be specified in the config to load the correct environment.")

    print(f"Instantiating environment of type: {env_type}")
    env = getCorrectEnv(env_type, config)

    if loaded_env_state:
        env.env_method("set_env_state", loaded_env_state)
        print("Applied loaded state to the environment.")

    wandb_id = wandb_run_id_to_resume if wandb_run_id_to_resume else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_run_name = config.get("WANDB_RUN_NAME", "resumed_run") + f"_from_{model_filename_no_ext_load}"

    env.env_method("set_wandb_id", wandb_id, wandb_run_name)

    curriculum_callback = CurriculumCallback(
        initial_phase_duration=config.get("MAX_STEPS_FIRST_PHASE"),
        max_step_increase_per_phase=config["MAX_STEP_INCREASE_PER_PHASE"],
        phase_duration_increase=config["PHASE_STEP_INCREASE"],
        hpo_disable_curriculum=config.get("PARAMETER_TUNING", False),
        verbose=0
    )
    if loaded_curriculum_state:
        curriculum_callback.set_state(loaded_curriculum_state)
        print("Applied loaded state to the curriculum callback.")

    custom_checkpoint_callback = CustomCheckpointCallbackWithStates(
        save_freq=config["STEP_SAVE_FREQ"],
        save_path=config['MODEL_SAVE_PATH'],
        name_prefix=config.get("NEW_CHECKPOINT_PREFIX", "ppo_resumed"),
        curriculum_callback=curriculum_callback,
        config=config,
        verbose=1
    )

    wandb_logging_callback = WandbLoggingCallback(log_freq=config.get("CALLBACK_LOG_FREQ", 100))
    callbacks_list = [custom_checkpoint_callback, curriculum_callback, wandb_logging_callback]

    wandb.init(
        project=config.get("WANDB_PROJECT", "celloracle"),
        config=config,
        name=wandb_run_name,
        sync_tensorboard=True,
        resume="allow",
        id=wandb_id
    )

    print(f"Loading SB3 model from: {model_zip_path}...")
    model = PPO.load(
        model_zip_path,
        env=env,
        device=config.get("DEVICE", "auto"),
        tensorboard_log=config["LOG_DIR"]
    )
    print(f"Model loaded. Resuming from timestep: {model.num_timesteps}")
    wandb.watch(model.policy, log="all", log_freq=config.get("WANDB_WATCH_FREQ", 1000))

    # --- 8. Continue Training ---
    if model.num_timesteps >= config['TOTAL_TIMESTEPS']:
        print("Model already trained to or beyond the target total timesteps. No further training will occur.")
    else:
        print(f"Continuing training up to {config['TOTAL_TIMESTEPS']} total timesteps...")
        model.learn(
            total_timesteps=config["TOTAL_TIMESTEPS"],
            log_interval=config.get("LOG_INTERVAL", 1),
            callback=callbacks_list,
            reset_num_timesteps=False
        )
        print("Resumed training finished.")

        # Save the final version of the resumed model
        final_save_path = os.path.join(config['MODEL_SAVE_PATH'], "final_model_resumed.zip")
        model.save(final_save_path)
        print(f"Saved final resumed model to {final_save_path}")

    env.close()
    wandb.finish()
    print("--- Resumed Training Run Finished ---")


if __name__ == "__main__":
    resume_config = {
        # --- Paths ---
        "ORACLE_PATH": os.path.join('../celloracle_data',
                                    "celloracle_object/new_promoter_without_mescs_trimmed_test_own_umap",
                                    "ready_oracle.pkl"),
        "LOAD_MODEL_PATH": os.path.join("../celloracle_data", "models", "2024_05_22_12_05_59_model",
                                        "ppo_model_10_steps.zip"),  # !! UPDATE THIS !!
        "MODEL_SAVE_PATH": os.path.join("../celloracle_data", "models", "2024_05_22_12_05_59_resumed_run"),
        "LOG_DIR": os.path.join('../celloracle_data', "logs", "2024_05_22_12_05_59_resumed_run"),
        "NEW_CHECKPOINT_PREFIX": "ppo_resumed_checkpoint",
        "WAND_B_LOG_DIR_BASE": os.path.join('../celloracle_data', "optuna_hpo_wandb_logs2"),

        "ENV_TYPE": "NEW_INPUT",

        "BATCH_SIZE": 256,
        "TOTAL_TIMESTEPS": 150000,
        "VERBOSE": 1,
        "LOG_INTERVAL": 1,
        "STEP_SAVE_FREQ": 10000,

        "PPO_N_STEPS": 256,
        "PPO_N_EPOCHS": 12,
        "GAMMA": 0.98,
        "GAE_LAMBDA": 0.95,
        "CLIP_RANGE": 0.2,
        "ENT_COEF": 0.03,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LEARNING_RATE": 1.8e-4,

        "MAX_STEPS_PER_EPISODE": 30,
        "ALLOW_GENE_ACTIVATION": True,
        "STANDARD_SD_FACTOR": 1,
        "USE_SIMILARITY_REWARD": True,
        "USE_PREV_KNOCKOUT": True,
        "DISTANCE_REWARD_SCALE": 92.88,

        "MAX_STEPS_FIRST_PHASE": 75000,
        "MAX_STEP_INCREASE_PER_PHASE": 6,
        "PHASE_STEP_INCREASE": 50000,
        "TARGET_CELLS_PER_PHASE": 4,
        "PARAMETER_TUNING": False,

        "DISTANCE_GAMMA": 0.99,
        "STEP_PENALTY": -0.05,
        "GOAL_BONUS": 19.90,
        "FAIL_PENALTY": -3,
        "SAME_CELL_PENALTY": -1.54,

        "PI_ARCH": [128, 128, 128],
        "VF_ARCH": [128, 128, 128],
        "ACTIVATION_FN": nn.ReLU,

        "CALLBACK_LOG_FREQ": 100,
        "WANDB_PROJECT": "celloracle",
        "WANDB_RUN_NAME": "ppo_resumed_run",
        "WANDB_WATCH_FREQ": 1000,
        "WANDB_RUN_ID": None
    }

    run_resumed_training(resume_config)