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
    trial_config["PPO_N_STEPS"] = trial.suggest_categorical("PPO_N_STEPS", [256, 512, 1024,2048])
    trial_config["PPO_N_EPOCHS"] = trial.suggest_int("PPO_N_EPOCHS", 5, 20)
    trial_config["CLIP_RANGE"] = trial.suggest_float("CLIP_RANGE", 0.1, 0.4)
    trial_config["ENT_COEF"] = trial.suggest_float("ENT_COEF", 0.0, 0.03)
    trial_config["GAMMA"] = trial.suggest_float("GAMMA", 0.97, 0.999)

    # Suggest environment reward hyperparameters
    trial_config["STEP_PENALTY"] = trial.suggest_float("STEP_PENALTY", -0.2, -0.01)
    trial_config["FAIL_PENALTY"] = trial.suggest_float("FAIL_PENALTY", -15, -1)
    trial_config["GOAL_BONUS"] = trial.suggest_float("GOAL_BONUS", 1, 15)
    trial_config["SAME_CELL_PENALTY"] = trial.suggest_float("SAME_CELL_PENALTY", -3.0, -0.5)
    trial_config["DISTANCE_REWARD_SCALE"] = trial.suggest_float("DISTANCE_REWARD_SCALE", 5.0, 80.0, log=True)
    trial_config["DISTANCE_GAMMA"] = trial.suggest_float("DISTANCE_GAMMA", 0.95, 1.0)

    # Suggest neural network architecture
    trial_config["PI_ARCH"] = [512,256]
    trial_config["VF_ARCH"] = [512,256]

    # --- Setup unique paths and IDs ---
    trial_suffix = f"trial_{trial.number}"
    trial_config["MODEL_SAVE_PATH"] = os.path.join(base_config["MODEL_SAVE_PATH_BASE"], trial_suffix)
    trial_config["LOG_DIR"] = os.path.join(base_config["LOG_DIR_BASE"], trial_suffix)
    trial_config[
        "WANDB_RUN_NAME"] = f"t{trial.number}_lr{trial_config['LEARNING_RATE']:.1e}_ent{trial_config['ENT_COEF']:.2f}"


    trial_config["optuna_trial_obj"] = trial
    trial_config["TOTAL_TIMESTEPS"] = base_config["TOTAL_TIMESTEPS_HPO"]

    os.makedirs(trial_config["MODEL_SAVE_PATH"], exist_ok=True)
    os.makedirs(trial_config["LOG_DIR"], exist_ok=True)

    print(f"\n---> Starting Optuna Trial {trial.number} <---")
    print("  --- Hyperparameters for this Trial ---")
    print(f"  PPO Agent:")
    print(f"    Learning Rate: {trial_config['LEARNING_RATE']:.6f}")  # More precision
    print(f"    N_Steps: {trial_config['PPO_N_STEPS']}")
    print(f"    N_Epochs: {trial_config['PPO_N_EPOCHS']}")
    print(f"    Clip Range: {trial_config['CLIP_RANGE']:.4f}")
    print(f"    Entropy Coef: {trial_config['ENT_COEF']:.6f}")
    print(f"    Gamma (Discount): {trial_config['GAMMA']:.4f}")

    print(f"  Reward Function:")
    print(f"    Goal Bonus: {trial_config['GOAL_BONUS']:.4f}")
    print(f"    Fail Penalty: {trial_config['FAIL_PENALTY']:.4f}")
    print(f"    Step Penalty: {trial_config['STEP_PENALTY']:.4f}")
    print(f"    Same Cell Penalty: {trial_config['SAME_CELL_PENALTY']:.4f}")
    print(f"    Distance Reward Scale: {trial_config['DISTANCE_REWARD_SCALE']:.4f}")
    print(f"    Distance Gamma (PBRS): {trial_config['DISTANCE_GAMMA']:.4f}")
    print("  ------------------------------------")


    metric_to_optimize = 0.0
    trial_status = "unknown"
    try:
        trained_model = ai_module.run_training(trial_config)
        env = trained_model.get_env()
        total_goals_list = env.env_method("get_total_goal_reached")
        if total_goals_list:
            metric_to_optimize = float(total_goals_list[0])
        else:
            metric_to_optimize = 0.0

        trial_status = "completed"
        print(f"Optuna Trial {trial.number} {trial_status}. Final Metric (Mean Reward): {metric_to_optimize:.4f}")

    except optuna.exceptions.TrialPruned:
        trial_status = "pruned"
        print(f"Optuna Trial {trial.number} pruned.")
        raise

    except Exception as e:
        trial_status = "failed"
        print(f"Optuna Trial {trial.number} FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
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

    os.makedirs(base_config["MODEL_SAVE_PATH_BASE"], exist_ok=True)
    os.makedirs(base_config["LOG_DIR_BASE"], exist_ok=True)

    hpo_total_sb3_timesteps = base_config["TOTAL_TIMESTEPS_HPO"]
    assumed_reports_per_trial = max(1, hpo_total_sb3_timesteps // 20000)

    hpo_report_interval = base_config.get("OPTUNA_REPORT_INTERVAL", 1000)  # Get from config
    min_timesteps_before_pruning = 200000

    n_warmup_steps_calculated = min_timesteps_before_pruning // hpo_report_interval

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=n_warmup_steps_calculated,
        interval_steps=1
    )


    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=5,
        multivariate=True,
        group=True
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    n_hpo_trials_to_run = 60

    print("running number of trials:" , n_hpo_trials_to_run)
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
