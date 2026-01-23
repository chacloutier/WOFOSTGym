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
    
    max_n: float = 80.0
    max_w: float = 40.0
    max_k: float = 100.0
    max_p: float = 100.0 
    
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
            
            # --- INTERLEAVED LOGIC ---
            # Identify all needs that are currently unsatisfied
            needs = []
            if total_n < N_LIMIT: needs.append('n')
            if total_w < WATER_LIMIT: needs.append('irrig')
            if total_k < K_LIMIT: needs.append('k')
            if total_p < P_LIMIT: needs.append('p')
            
            # Randomly select one need to fulfill this step
            # This spreads resources out rather than doing N then W then K...
            if needs:
                choice = random.choice(needs)
                action_dict[choice] = 1
                
                # Update trackers
                if choice == 'n': total_n += FERT_UNIT
                elif choice == 'irrig': total_w += IRRIG_UNIT
                elif choice == 'k': total_k += FERT_UNIT
                elif choice == 'p': total_p += FERT_UNIT
            
            action_int = utils.action_to_numpy(envs.envs[0], action_dict)
            next_obs, reward, terminations, truncations, infos = envs.step(action_int)
            
            episodic_return += reward[0]
            done = np.logical_or(terminations, truncations)[0]

            if done:
                writer.add_scalar("charts/average_reward", episodic_return, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()