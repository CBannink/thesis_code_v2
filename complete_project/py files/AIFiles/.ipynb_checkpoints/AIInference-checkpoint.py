import os
import pickle
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from AI import CellOracleSB3VecEnv as CellOracleCustomEnv
from stable_baselines3 import PPO


def run_inference_on_single_cell(
        model_path: str,
        oracle_path: str,
        config: Dict,
        start_cell_idx: int,
        target_cell_type: str,
        max_inference_steps: int = 100,
        deterministic: bool = True  ) -> Optional[Tuple[List[str], bool, int]]:


    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    inference_config = config.copy()
    inference_config["BATCH_SIZE"] = 1

    env = CellOracleCustomEnv(
        oracle_path=oracle_path,
        batch_size=inference_config["BATCH_SIZE"],
        max_steps=max_inference_steps,  # Use the inference step limit
        step_penalty=inference_config.get("STEP_PENALTY", -0.1),
        goal_bonus=inference_config.get("GOAL_BONUS", 10),
        fail_penalty=inference_config.get("FAIL_PENALTY", -1),
        distance_reward_scale=inference_config.get("DISTANCE_REWARD_SCALE", 50),
        same_cell_penalty=inference_config.get("SAME_CELL_PENALTY", -0.5),
        gamma_distance=inference_config.get("DISTANCE_GAMMA", 1.0),
        allow_gene_activation=inference_config.get("ALLOW_GENE_ACTIVATION", True),
        use_prev_perturbs=inference_config.get("USE_PREV_KNOCKOUT", False),
        number_of_targets_curriculum=inference_config.get("TARGET_CELLS_PER_PHASE", 4),
        use_similarity=inference_config.get("USE_SIMILARITY_REWARD", True),
    )

    model = PPO.load(model_path, env=env)


    try:
        obs, _ = env.reset(options={'initial_cell_idx': start_cell_idx, 'target_cell_type': target_cell_type})
    except Exception as e:
        print(f"Error during env.reset(). Ensure your reset method can handle options: {e}")
        print("Please modify your reset method to accept an 'options' dictionary.")
        return None

    action_sequence = []
    action_details = []
    goal_reached = False

    for step in range(max_inference_steps):
        action_idx, _states = model.predict(obs, deterministic=deterministic)
        action_detail_str = env.env_method("get_action_details", int(action_idx[0]))[0]  # Using env_method
        action_sequence.append(int(action_idx[0]))
        action_details.append(action_detail_str)

        print(f"Step {step + 1}: Predicted Action -> {action_detail_str}")

        obs, reward, terminated, truncated, infos = env.step(action_idx)

        if infos[0].get('goal_reached', False):
            print(f"\nSUCCESS: Target cell type '{target_cell_type}' reached in {step + 1} steps!")
            goal_reached = True
            break

        if terminated[0] or truncated[0]:
            print(f"\nEpisode ended at step {step + 1} without reaching goal.")
            break

    if not goal_reached and not (terminated[0] or truncated[0]):
        print(f"\nInference finished: Max steps ({max_inference_steps}) reached without achieving goal.")

    env.close()
    return action_details, goal_reached, step + 1