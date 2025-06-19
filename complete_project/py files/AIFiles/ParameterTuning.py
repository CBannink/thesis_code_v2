# run_optuna_hpo.py

import optuna
import os
import copy
import time  # For unique run IDs, not strictly necessary if trial.number is enough
from datetime import datetime
import torch.nn as nn  # For default activation function
import numpy as np
import wandb

# --- Your Custom Modules ---
# Ensure these can be imported. If they are in a subdirectory,
# you might need to adjust Python's path or use relative imports if this script becomes part of a package.
import AI as ai_module

# from Env import CellOracleSB3VecEnv # Or your Env_with_new_input
# from AI import SB3OptunaPruningCallback # Assuming it's defined in AI.py or imported there

# --- Base Configuration (Modify as needed) ---
# This will be the starting point, Optuna will override parts of it.


def objective(trial: optuna.Trial, base_config: dict) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # --- 1. Setup Configuration for this Specific Trial ---
    trial_config = copy.deepcopy(base_config)

    # Suggest hyperparameters to be tuned by Optuna
    trial_config["LEARNING_RATE"] = trial.suggest_float("LEARNING_RATE", 3e-5, 3e-4, log=True)
    trial_config["PPO_N_STEPS"] = trial.suggest_categorical("PPO_N_STEPS", [128,256, 384, 512, 1024])
    trial_config["PPO_N_EPOCHS"] = trial.suggest_int("PPO_N_EPOCHS", 5, 20)
    trial_config["CLIP_RANGE"] = trial.suggest_float("CLIP_RANGE", 0.1, 0.4)
    trial_config["ENT_COEF"] = trial.suggest_float("ENT_COEF", 0.0, 0.03)
    trial_config["VF_COEF"] = trial.suggest_float("VF_COEF", 0.3, 0.7)
    trial_config["GAMMA"] = trial.suggest_float("GAMMA", 0.98, 0.999)

    # Suggest environment reward hyperparameters
    trial_config["STEP_PENALTY"] = trial.suggest_float("STEP_PENALTY", -0.2, -0.01)
    trial_config["FAIL_PENALTY"] = trial.suggest_float("STEP_PENALTY", -0.5, -5)
    trial_config["GOAL_BONUS"] = trial.suggest_float("GOAL_BONUS", 5.0, 25.0)
    trial_config["SAME_CELL_PENALTY"] = trial.suggest_float("SAME_CELL_PENALTY", -3.0, -0.5)
    trial_config["DISTANCE_REWARD_SCALE"] = trial.suggest_float("DISTANCE_REWARD_SCALE", 10.0, 120.0, log=True)
    trial_config["DISTANCE_GAMMA"] = trial.suggest_float("DISTANCE_GAMMA", 0.95, 1.0)

    # Suggest neural network architecture
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    trial_config["PI_ARCH"] = [base_config.get("PI_ARCH", [512])[0],
                               hidden_size]  # Example: Keep first layer fixed, tune second
    trial_config["VF_ARCH"] = [base_config.get("VF_ARCH", [512])[0], hidden_size]

    # --- Setup unique paths and IDs ---
    trial_suffix = f"trial_{trial.number}"
    trial_config["MODEL_SAVE_PATH"] = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], trial_suffix)
    trial_config["LOG_DIR"] = os.path.join(base_config["LOG_DIR_BASE"], trial_suffix)
    trial_config["WANDB_RUN_ID"] = f"optuna_{base_config.get('STUDY_NAME_PREFIX', 'hpo')}_{trial_suffix}"
    trial_config[
        "WANDB_RUN_NAME"] = f"t{trial.number}_lr{trial_config['LEARNING_RATE']:.1e}_ent{trial_config['ENT_COEF']:.2f}"

    trial_config["optuna_trial_obj"] = trial
    trial_config["TOTAL_TIMESTEPS"] = base_config["TOTAL_TIMESTEPS_HPO"]

    os.makedirs(trial_config["MODEL_SAVE_PATH"], exist_ok=True)
    os.makedirs(trial_config["LOG_DIR"], exist_ok=True)

    print(f"\n---> Starting Optuna Trial {trial.number} <---")
    print(
        f"  LR: {trial_config['LEARNING_RATE']:.3e}, ENT_COEF: {trial_config['ENT_COEF']:.3f}, GOAL: {trial_config['GOAL_BONUS']:.1f}")

    # Initialize variables for the finally block
    trial_status = "unknown"
    metric_to_optimize = -float('inf')  # Default for failed/pruned trials

    try:
        # --- 2. Run Training ---
        # Assumes run_training is modified to only return the model object
        trained_model = ai_module.run_training(trial_config)

        # --- 3. Extract Final Metric (This code only runs if training was not pruned) ---
        if hasattr(trained_model, 'ep_info_buffer') and trained_model.ep_info_buffer:

            # <<< --- THE FIX IS APPLIED HERE --- >>>
            # 1. Convert the deque to a list before slicing.
            ep_info_list = list(trained_model.ep_info_buffer)

            # 2. Now you can safely slice the list.
            num_episodes = len(ep_info_list)
            # e.g., average the last 20% of episodes, with a minimum of 10 and max of 50
            num_last_eps_to_avg = min(max(10, int(num_episodes * 0.2)), 50)

            recent_rewards = [
                ep_info['r'] for ep_info in ep_info_list[-num_last_eps_to_avg:]
                if 'r' in ep_info and not np.isnan(ep_info['r'])
            ]

            if recent_rewards:
                metric_to_optimize = np.mean(recent_rewards)
            else:
                print(f"Trial {trial.number}: Valid recent rewards not found in ep_info_buffer.")
        else:
            print(f"Trial {trial.number}: 'ep_info_buffer' is empty or not found. Cannot determine final metric.")

        trial_status = "completed"
        print(f"Optuna Trial {trial.number} {trial_status}. Final Metric (Mean Reward): {metric_to_optimize:.4f}")

    except optuna.exceptions.TrialPruned:
        # The SB3OptunaPruningCallback should raise this exception
        trial_status = "pruned"
        print(f"Optuna Trial {trial.number} pruned.")
        raise  # Re-raise the exception for Optuna's main loop to handle

    except Exception as e:
        trial_status = "failed"
        print(f"Optuna Trial {trial.number} FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to make Optuna mark the trial as FAIL

    finally:
        # --- 4. Finalize Logging (runs for completed, pruned, and failed trials) ---
        if wandb.run is not None:
            print(f"Finishing W&B run for trial {trial.number} (Status: {trial_status})")
            # Log final status and metric to WandB summary for easy viewing
            wandb.run.summary["trial_status_optuna"] = trial_status
            wandb.run.summary["final_metric_for_optuna"] = metric_to_optimize

            exit_code = 0
            if trial_status == "pruned":
                exit_code = 2
            elif trial_status == "failed":
                exit_code = 1
            wandb.finish(exit_code=exit_code)

    return metric_to_optimize


def run_optuna_hpo(base_config):
    study_name = "ppo_celloracle_hpo_" + datetime.now().strftime('%Y%m%d_%H%M')
    storage_name = f"sqlite:///{study_name}.db"  # Save results in a SQLite DB

    # --- Create Base Directories ---
    os.makedirs(base_config["MODEL_SAVE_PATH_BASE"], exist_ok=True)
    os.makedirs(base_config["LOG_DIR_BASE"], exist_ok=True)

    # --- Pruner Configuration ---
    # TODO: Adjust pruner parameters based on TOTAL_TIMESTEPS_HPO and SB3OptunaPruningCallback.report_interval_steps
    # `n_warmup_steps` here is the number of *intermediate steps reported to Optuna*,
    # NOT the number of Optuna trials.
    # Example: If TOTAL_TIMESTEPS_HPO = 200k and SB3OptunaPruningCallback reports every 10k SB3 steps,
    # then `n_warmup_steps = 5` means wait for 5 reports (i.e., 50k SB3 steps) before pruning.
    hpo_total_sb3_timesteps = base_config["TOTAL_TIMESTEPS_HPO"]
    # Assuming SB3OptunaPruningCallback reports roughly every `hpo_total_sb3_timesteps / 10`
    # This is a guess for `report_interval_steps` used in SB3OptunaPruningCallback
    # You should set `report_interval_steps` explicitly in SB3OptunaPruningCallback.
    # Let's assume SB3OptunaPruningCallback's report_interval_steps is ~20k for 200k total.
    assumed_reports_per_trial = max(1, hpo_total_sb3_timesteps // 20000)  # e.g., 10 reports for 200k total

    pruner_n_warmup_reports = max(1, assumed_reports_per_trial // 4)  # Prune after 1/3rd of reports

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Number of Optuna trials to complete before pruning can occur.
        n_warmup_steps=pruner_n_warmup_reports,  # Number of *reports to Optuna* before this trial can be pruned.
        # e.g., if 3, wait for 3 calls to trial.report() for this trial.
        interval_steps=1  # Check for pruning at every call to trial.report() after n_warmup_steps.
    )


    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,  # Number of random exploration trials before TPE starts optimizing
        multivariate=True,  # Consider correlations between hyperparameters
        group=True  # Exploit group structure if hyperparameters are conditional
    )

    # --- Create and Run Study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",  # We want to maximize reward
        sampler=sampler,
        pruner=pruner
    )
    n_hpo_trials_to_run = 30
    study_timeout_seconds = 3600 * 768
    print(f"\n--- Starting Optuna HPO Study: {study_name} ---")
    try:
        print("test")
        study.optimize(
            lambda trial: objective(trial, base_config),
            n_trials=n_hpo_trials_to_run,
            timeout=study_timeout_seconds,
            show_progress_bar=True,
        )
    except Exception as e_study:
        print(f"ERROR weird: {e_study}")
        import traceback
        traceback.print_exc()

    print("\n--- Optuna Study Finished ---")
    print(f"Study dashboard command: optuna-dashboard {storage_name}")
    print(f"Number of finished trials in study: {len(study.trials)}")

    try:
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        complete_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        print(f"  Trials: Completed={complete_trials}, Pruned={pruned_trials}, Failed={failed_trials}")

        if complete_trials > 0:
            best_trial = study.best_trial
            print("\nBest trial:")
            print(f"  Value (Maximized Metric): {best_trial.value:.4f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

            # Save best parameters to a file
            best_params_filename = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], f"{study_name}_best_params.txt")
            with open(best_params_filename, "w") as f:
                f.write(f"Study Name: {study_name}\n")
                f.write(f"Best trial number: {best_trial.number}\n")
                f.write(f"Best trial value: {best_trial.value}\n\n")
                f.write("Best Hyperparameters:\n")
                for key, value in best_trial.params.items():
                    f.write(f"  {key}: {value}\n")
            print(f"Best parameters saved to {best_params_filename}")
        else:
            print("No trials completed successfully, cannot determine the best one.")

    except ValueError:
        print("ERROR: No completed trials found.")
    except Exception as e_results:
        print("WRONG", e_results)
