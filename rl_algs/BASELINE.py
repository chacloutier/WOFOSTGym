import wandb
import time
import torch
import numpy as np
import gymnasium as gym
from argparse import Namespace
from rl_algs.rl_utils import setup, RL_Args
from dataclasses import dataclass
import random  # Needed for interleaved logic
import utils 

@dataclass
class Args(RL_Args):
    """
    Arguments for the Baseline agent matching the 'Learning Under Constraints' experiment.
    """
    total_timesteps: int = 1000000
    """total timesteps of the experiment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    
    max_n: float = 40.0
    max_w: float = 20.0
    max_k: float = 40.0
    max_p: float = 40.0 
    
    seed: int = 1
    checkpoint_frequency: int = 100000

def estimate_season_length(envs, seed):
    obs, _ = envs.reset(seed=seed)
    done = False
    T = 0
    zero_action = utils.action_to_numpy(envs.envs[0], {"n": 0, "p": 0, "k": 0, "irrig": 0})
    while not done:
        _, _, terminations, truncations, _ = envs.step(zero_action)
        done = np.logical_or(terminations, truncations)[0]
        T += 1
    return T

def train(kwargs: Namespace) -> None:
    args = kwargs.alg
    run_name = f"Baseline/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer, device, envs = setup(kwargs, args, run_name)

    global_step = 0
    start_time = time.time()
    
    N_LIMIT = args.max_n 
    WATER_LIMIT = args.max_w
    K_LIMIT = args.max_k
    P_LIMIT = args.max_p
    FERT_UNIT = envs.envs[0].unwrapped.fert_amount
    IRRIG_UNIT = envs.envs[0].unwrapped.irrig_amount

    # 2. Run Baseline
    while global_step < args.total_timesteps:
        obs, _ = envs.reset(seed=args.seed)
        done = False
        episodic_return = 0
        
        # Track cumulative usage
        total_n = 0
        total_w = 0
        total_k = 0
        total_p = 0

        while not done:
            global_step += 1
            
            action_dict = {"n": 0, "p": 0, "k": 0, "irrig": 0}
            
            # 1. Calculate deficits (How much are we missing?)
            n_deficit = max(0, (N_LIMIT - total_n)/N_LIMIT)
            p_deficit = max(0, (P_LIMIT - total_p)/P_LIMIT)
            k_deficit = max(0, (K_LIMIT - total_k)/K_LIMIT)
            w_deficit = max(0, (WATER_LIMIT - total_w)/WATER_LIMIT)
            
            # 2. Find the biggest problem
            deficits = {
                'n': n_deficit,
                'p': p_deficit,
                'k': k_deficit,
                'irrig': w_deficit
            }
            
            # Get the resource with the highest deficit
            most_urgent_need = max(deficits, key=deficits.get)
            
            # 3. If the deficit is significant (> 0), act on it
            if deficits[most_urgent_need] > 0:
                action_dict[most_urgent_need] = 1
                
                # Update trackers
                if most_urgent_need == 'n': total_n += FERT_UNIT
                elif most_urgent_need == 'irrig': total_w += IRRIG_UNIT
                elif most_urgent_need == 'k': total_k += FERT_UNIT
                elif most_urgent_need == 'p': total_p += FERT_UNIT
            
            action_int = utils.action_to_numpy(envs.envs[0], action_dict)
            next_obs, reward, terminations, truncations, infos = envs.step(action_int)
            
            episodic_return += reward[0]
            done = np.logical_or(terminations, truncations)[0]

            if done:
                # Log the Reward
                writer.add_scalar("charts/average_reward", episodic_return, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                
                # Log the Constraints so they appear in WandB exactly like PPO!
                if isinstance(infos, dict) and "track/total_n" in infos:
                    writer.add_scalar("constraints/total_n", infos["track/total_n"][0], global_step)
                    writer.add_scalar("constraints/total_p", infos["track/total_p"][0], global_step)
                    writer.add_scalar("constraints/total_k", infos["track/total_k"][0], global_step)
                    writer.add_scalar("constraints/total_w", infos["track/total_w"][0], global_step)
                    writer.add_scalar("constraints/violation_rate", infos["track/is_violating"][0], global_step)
                elif isinstance(infos, list) and len(infos) > 0 and "track/total_n" in infos[0]:
                    writer.add_scalar("constraints/total_n", infos[0]["track/total_n"], global_step)
                    writer.add_scalar("constraints/total_p", infos[0]["track/total_p"], global_step)
                    writer.add_scalar("constraints/total_k", infos[0]["track/total_k"], global_step)
                    writer.add_scalar("constraints/total_w", infos[0]["track/total_w"], global_step)
                    writer.add_scalar("constraints/violation_rate", infos[0]["track/is_violating"], global_step)

    envs.close()
    writer.close()