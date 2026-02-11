import wandb
import time
from dataclasses import dataclass

from argparse import Namespace
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Optional
from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy


@dataclass
class Args(RL_Args):
    total_timesteps: int = 1000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 650
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    checkpoint_frequency: int = 500

    # --- PPO-Lagrangian Specific Args ---
    # Note: cost_limit is now implicitly 1.0 per resource due to normalization
    lagrangian_learning_rate: float = 5e-2
    initial_lambda: float = 1.0
    
    # Specific Cost parameters
    cost_gamma: float = 0.999 
    """Discount factor specific to cost/usage"""
    cost_gae_lambda: float = 0.95
    """GAE lambda specific to cost/usage"""

    # --- Constraint Thresholds ---
    max_n: float = 80.0
    max_p: float = 80.0
    max_k: float = 80.0
    max_w: float = 40.0

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer: nn.Module, std: np.ndarray = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class UsageTracker:
    """
    Tracks cumulative resource usage from environment infos to calculate 
    per-step incremental usage rates.
    """
    def __init__(self, num_envs, args):
        self.num_envs = num_envs
        self.args = args
        self.prev_totals = np.zeros((num_envs, 4)) # N, P, K, W
        
    def reset(self, env_indices=None):
        if env_indices is None:
            self.prev_totals.fill(0.0)
        else:
            self.prev_totals[env_indices] = 0.0

    def extract_step_usage(self, infos, dones) -> np.ndarray:
        """
        Returns array of shape (num_envs, 4) representing normalized delta usage.
        Normalization: delta / max_limit
        """
        current_totals = np.zeros((self.num_envs, 4))
        
        # 1. Extract current cumulative totals from infos
        if "track/total_n" in infos: # Vectorized Env
            for i in range(self.num_envs):
                current_totals[i, 0] = infos["track/total_n"][i]
                current_totals[i, 1] = infos["track/total_p"][i]
                current_totals[i, 2] = infos["track/total_k"][i]
                current_totals[i, 3] = infos["track/total_w"][i]
        elif isinstance(infos, list): # List of Dicts
            for i, info in enumerate(infos):
                current_totals[i, 0] = info.get("track/total_n", 0.0)
                current_totals[i, 1] = info.get("track/total_p", 0.0)
                current_totals[i, 2] = info.get("track/total_k", 0.0)
                current_totals[i, 3] = info.get("track/total_w", 0.0)

        # 2. Calculate Deltas
        deltas = current_totals - self.prev_totals
        
        # Clip negative deltas (happens on reset boundary if not handled strictly)
        deltas = np.maximum(deltas, 0.0) 
        
        # 3. Update State
        self.prev_totals = current_totals.copy()
        
        # If env finished, reset the tracker for that env so next step starts from 0
        if np.any(dones):
            env_indices = np.where(dones)[0]
            self.reset(env_indices)

        # 4. Normalize by limits
        # [N, P, K, W]
        limits = np.array([self.args.max_n, self.args.max_p, self.args.max_k, self.args.max_w])
        norm_deltas = deltas / (limits + 1e-8)
        
        return norm_deltas


class PPOLag(nn.Module, Agent):
    def __init__(self, envs: gym.Env, args: Args, state_fpath: str = None) -> None:
        super().__init__()
        self.env = envs
        self.args = args

        obs_dim = np.array(envs.single_observation_space.shape).prod()

        # 1. Reward Critic (Standard V_R)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # 2. Rate/Usage Critic (Predicts 4 separate resource streams)
        # Output dim is 4: [N_rate, P_rate, K_rate, W_rate]
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 4), std=1.0),
        )

        # 3. Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        # 4. Lagrange Multipliers (4 Independent Learnable Parameters)
        self.log_lagrange_multiplier = nn.Parameter(
            torch.log(torch.tensor([args.initial_lambda]*4)), 
            requires_grad=True
        )

        if state_fpath is not None:
            try:
                self.load_state_dict(torch.load(state_fpath, weights_only=True))
            except:
                raise Exception(f"Error loading state dictionary from {state_fpath}")

    def get_lagrange_multiplier(self):
        """Returns the current values of lambdas (strictly positive)"""
        return torch.exp(self.log_lagrange_multiplier)

    def get_action(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Helper for inference"""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Reward Value, Cost Value[4])"""
        return self.critic(x), self.cost_critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # Returns: action, log_prob, entropy, reward_val, cost_val(Batch, 4)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), self.cost_critic(x)


def train(kwargs: Namespace) -> None:
    """
    PPO-Lagrangian Training Function (Rate-Based)
    """
    args = kwargs.alg
    run_name = f"PPOLag-Rate/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)
    agent = PPOLag(envs, args).to(device)
    usage_tracker = UsageTracker(args.num_envs, args)
    
    # Optimizer for Policy and Both Value Networks
    optimizer = optim.Adam(
        list(agent.actor.parameters()) + list(agent.critic.parameters()) + list(agent.cost_critic.parameters()),
        lr=args.learning_rate, 
        eps=1e-5
    )

    # Separate Optimizer for the Lagrange Multipliers
    lagrange_optimizer = optim.Adam([agent.log_lagrange_multiplier], lr=args.lagrangian_learning_rate)

    # Storage buffers
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Multi-dimensional buffers (Steps, Envs, 4)
    costs = torch.zeros((args.num_steps, args.num_envs, 4)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs, 4)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    usage_tracker.reset()

    for iteration in range(1, args.num_iterations + 1):
        if global_step % args.checkpoint_frequency == 0:
            torch.save(agent.state_dict(), f"{kwargs.save_folder}{run_name}/agent.pt")
            if kwargs.track:
                wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout Phase ---
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, cost_value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                cost_values[step] = cost_value # Shape (Num_Envs, 4)

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            
            # --- UPDATED: Rate/Usage Calculation ---
            step_usage = usage_tracker.extract_step_usage(infos, next_done)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            costs[step] = torch.tensor(step_usage).to(device) # Shape (Num_Envs, 4)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            
            # --- Logging ---
            if global_step % args.checkpoint_frequency == 0:
                writer.add_scalar("charts/average_reward", eval_policy(agent, envs, kwargs, device), global_step)
                if "track/total_n" in infos:
                    writer.add_scalar("constraints/total_n", infos["track/total_n"][0], global_step)
                    writer.add_scalar("constraints/total_n", infos["track/total_n"][0], global_step)
                    writer.add_scalar("constraints/total_p", infos["track/total_p"][0], global_step)
                    writer.add_scalar("constraints/total_k", infos["track/total_k"][0], global_step)
                    writer.add_scalar("constraints/total_w", infos["track/total_w"][0], global_step)
                    writer.add_scalar("constraints/violation_rate", infos["track/is_violating"], global_step)

        # --- GAE Calculation ---
        with torch.no_grad():
            next_value, next_cost_value = agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1)
            next_cost_value = next_cost_value.reshape(args.num_envs, 4)
            
            # 1. Reward GAE
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            # 2. Cost GAE (Vectorized for 4 Dimensions)
            cost_advantages = torch.zeros_like(costs).to(device)
            last_cost_gaelam = torch.zeros((args.num_envs, 4)).to(device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # Broadcast nonterminal to (Envs, 1) for broadcasting against (Envs, 4)
                    nextnonterminal = (1.0 - next_done).unsqueeze(1)
                    next_c_values = next_cost_value
                else:
                    nextnonterminal = (1.0 - dones[t + 1]).unsqueeze(1)
                    next_c_values = cost_values[t + 1]
                
                delta_cost = costs[t] + args.cost_gamma * next_c_values * nextnonterminal - cost_values[t]
                cost_advantages[t] = last_cost_gaelam = delta_cost + args.cost_gamma * args.cost_gae_lambda * nextnonterminal * last_cost_gaelam

            returns = advantages + values
            cost_returns = cost_advantages + cost_values

        # Flatten buffers
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Flatten multi-dim buffers
        b_cost_advantages = cost_advantages.reshape(-1, 4)
        b_cost_returns = cost_returns.reshape(-1, 4)
        b_cost_values = cost_values.reshape(-1, 4)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        
        # --- Update Phase ---
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newcostvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # --- Lagrangian Advantage Calculation ---
                cur_lambdas = agent.get_lagrange_multiplier().detach() # Shape (4,)
                
                mb_advantages = b_advantages[mb_inds]
                mb_cost_advantages = b_cost_advantages[mb_inds]

                # Normalize Reward Advantages
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    # Normalize Cost Advantages per dimension to preserve relative scales
                    mb_cost_advantages = (mb_cost_advantages - mb_cost_advantages.mean(dim=0)) / (mb_cost_advantages.std(dim=0) + 1e-8)
                
                # Combine: Reward Adv - sum(Lambda_i * Cost_Adv_i)
                # Weighted Sum across the 4 dimensions
                weighted_cost_adv = (mb_cost_advantages * cur_lambdas.unsqueeze(0)).sum(dim=1)
                combined_advantages = mb_advantages - weighted_cost_adv

                # Policy Loss
                pg_loss1 = -combined_advantages * ratio
                pg_loss2 = -combined_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Reward Value Loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Cost Value Loss (Mean Squared Error across all 4 dimensions)
                # newcostvalue is (Minibatch, 4)
                if args.clip_vloss:
                    cv_loss_unclipped = (newcostvalue - b_cost_returns[mb_inds]) ** 2
                    cv_clipped = b_cost_values[mb_inds] + torch.clamp(
                        newcostvalue - b_cost_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    cv_loss_clipped = (cv_clipped - b_cost_returns[mb_inds]) ** 2
                    cv_loss_max = torch.max(cv_loss_unclipped, cv_loss_clipped)
                    cv_loss = 0.5 * cv_loss_max.mean()
                else:
                    cv_loss = 0.5 * ((newcostvalue - b_cost_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                
                # Total Loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + cv_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # --- Lagrange Multiplier Update ---
        # We check expected usage against normalized limit (1.0)
        mean_cost_return = b_cost_returns.mean(dim=0) # Shape (4,)
        violation = mean_cost_return - 1.0 # Limit is implicitly 1.0 due to normalization
        
        # Vectorized Lambda Update
        lambda_loss = -(agent.get_lagrange_multiplier() * violation.detach()).sum()
        
        lagrange_optimizer.zero_grad()
        lambda_loss.backward()
        lagrange_optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/lambda_mean", agent.get_lagrange_multiplier().mean().item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", cv_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    envs.close()
    writer.close()