import wandb
import time
from dataclasses import dataclass
from argparse import Namespace
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy

# =====================================================
# Arguments
# =====================================================

@dataclass
class Args(RL_Args):
    total_timesteps: int = 1_000_000
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
    cv_coef: float = 0.5

    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    checkpoint_frequency: int = 500

    # --- Lagrangian / Constraint Control ---
    lagrangian_learning_rate: float = 5e-2
    initial_lambda: float = 1.0
    fixed_lambda: bool = False

    pid_kp: float = 0.1          # proportional gain
    target_limit: float = 0.75   # desired mean usage rate

    # --- Resource limits (for normalization) ---
    max_n: float = 40.0
    max_p: float = 40.0
    max_k: float = 40.0
    max_w: float = 20.0
    terminate_on_violation: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# =====================================================
# Utilities
# =====================================================

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def log_constraint_infos(writer, infos, global_step):
    if isinstance(infos, dict):
        if "track/total_n" in infos:
            writer.add_scalar("constraints/total_n", infos["track/total_n"][0], global_step)
            writer.add_scalar("constraints/total_p", infos["track/total_p"][0], global_step)
            writer.add_scalar("constraints/total_k", infos["track/total_k"][0], global_step)
            writer.add_scalar("constraints/total_w", infos["track/total_w"][0], global_step)
        if "track/is_violating" in infos:
            writer.add_scalar(
                "constraints/violation_rate",
                float(infos["track/is_violating"][0]),
                global_step,
            )

    elif isinstance(infos, list) and len(infos) > 0:
        info = infos[0]
        if "track/total_n" in info:
            writer.add_scalar("constraints/total_n", info.get("track/total_n", 0.0), global_step)
            writer.add_scalar("constraints/total_p", info.get("track/total_p", 0.0), global_step)
            writer.add_scalar("constraints/total_k", info.get("track/total_k", 0.0), global_step)
            writer.add_scalar("constraints/total_w", info.get("track/total_w", 0.0), global_step)
            writer.add_scalar(
                "constraints/violation_rate",
                float(info.get("track/is_violating", 0.0)),
                global_step,
            )


class UsageTracker:
    def __init__(self, num_envs: int, args: Args):
        self.num_envs = num_envs
        self.args = args
        self.prev_totals = np.zeros((num_envs, 4))

    def reset(self, env_indices=None):
        if env_indices is None:
            self.prev_totals[:] = 0.0
        else:
            self.prev_totals[env_indices] = 0.0

    def extract_step_usage(self, infos, dones):
        current = np.zeros((self.num_envs, 4))

        if isinstance(infos, dict) and "track/total_n" in infos:
            for i in range(self.num_envs):
                current[i, 0] = infos["track/total_n"][i]
                current[i, 1] = infos["track/total_p"][i]
                current[i, 2] = infos["track/total_k"][i]
                current[i, 3] = infos["track/total_w"][i]
        elif isinstance(infos, list):
            for i, info in enumerate(infos):
                current[i, 0] = info.get("track/total_n", 0.0)
                current[i, 1] = info.get("track/total_p", 0.0)
                current[i, 2] = info.get("track/total_k", 0.0)
                current[i, 3] = info.get("track/total_w", 0.0)

        deltas = np.maximum(current - self.prev_totals, 0.0)
        self.prev_totals[:] = current

        if np.any(dones):
            self.reset(np.where(dones)[0])

        limits = np.array([self.args.max_n, self.args.max_p,
                           self.args.max_k, self.args.max_w])
        return deltas / (limits + 1e-8)


# =====================================================
# Agent
# =====================================================

class PPOLag(nn.Module, Agent):
    def __init__(self, envs: gym.Env, args: Args):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = envs.single_action_space.n

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 4), std=1.0),
        )

        self.log_lagrange_multiplier = nn.Parameter(
            torch.log(torch.ones(4) * args.initial_lambda)
        )

    def get_lagrange_multiplier(self):
        clamped_log_lambda = torch.clamp(self.log_lagrange_multiplier, min=-10.0)
        return torch.exp(clamped_log_lambda)

    def get_action(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(x),
            self.cost_critic(x),
        )

# =====================================================
# Training
# =====================================================

def train(kwargs: Namespace):
    args: Args = kwargs.alg
    run_name = f"PPOLagRate/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)
    agent = PPOLag(envs, args).to(device)
    usage_tracker = UsageTracker(args.num_envs, args)

    optimizer = optim.Adam(
        list(agent.actor.parameters()) +
        list(agent.critic.parameters()) +
        list(agent.cost_critic.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    lagrange_optimizer = optim.Adam(
        [agent.log_lagrange_multiplier],
        lr=args.lagrangian_learning_rate,
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    costs = torch.zeros((args.num_steps, args.num_envs, 4)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs, 4)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    usage_tracker.reset()

    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            # current_ent_coef = frac * args.ent_coef
            current_ent_coef = args.ent_coef
        else:
            current_ent_coef = args.ent_coef

        # ---------------- Rollout ----------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logp, _, value, cost_val = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                cost_values[step] = cost_val

            actions[step] = action
            logprobs[step] = logp

            next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            
            if args.terminate_on_violation:
                violated = np.zeros(args.num_envs, dtype=bool)
                if isinstance(infos, dict) and "track/total_n" in infos:
                    violated = (
                        (np.array(infos["track/total_n"]) > args.max_n) |
                        (np.array(infos["track/total_p"]) > args.max_p) |
                        (np.array(infos["track/total_k"]) > args.max_k) |
                        (np.array(infos["track/total_w"]) > args.max_w)
                    )
                elif isinstance(infos, list):
                    for i, info in enumerate(infos):
                        if (info.get("track/total_n", 0.0) > args.max_n or
                            info.get("track/total_p", 0.0) > args.max_p or
                            info.get("track/total_k", 0.0) > args.max_k or
                            info.get("track/total_w", 0.0) > args.max_w):
                            violated[i] = True
                
                for i in range(args.num_envs):
                    if violated[i] and not term[i] and not trunc[i]:
                        term[i] = True
                        if args.num_envs == 1:
                            next_obs, _ = envs.reset(seed=args.seed)
                            usage_tracker.reset()
                        else:
                            print("WARNING: Early termination triggered, but manual reset for num_envs > 1 requires a custom wrapper.")

            next_done = np.logical_or(term, trunc)

            if global_step % args.checkpoint_frequency == 0:
                writer.add_scalar("charts/average_reward", eval_policy(agent, envs, kwargs, device), global_step)
                log_constraint_infos(writer, infos, global_step)
                
                torch.save(agent.state_dict(), f"{kwargs.save_folder}{run_name}/agent.pt")
                if kwargs.track:
                    wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

            step_cost = usage_tracker.extract_step_usage(infos, next_done)
            scaled_reward = reward * 10.0

            rewards[step] = torch.tensor(scaled_reward, device=device, dtype=torch.float32)
            costs[step] = torch.tensor(step_cost, device=device, dtype=torch.float32)

            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.tensor(next_done, device=device, dtype=torch.float32)

        # ---------------- Reward & Cost GAE ----------------
        advantages = torch.zeros_like(rewards)
        cost_advantages = torch.zeros_like(costs) # [UPDATED] Initialize cost advantages [num_steps, num_envs, 4]
        
        lastgaelam = 0
        lastgaelam_cost = torch.zeros((args.num_envs, 4), device=device) # [UPDATED] Track GAE for 4 cost resources
        
        with torch.no_grad():
            next_value = agent.critic(next_obs).flatten()
            next_cost_value = agent.cost_critic(next_obs) # [UPDATED] Get next predicted cost
            
        for t in reversed(range(args.num_steps)):
            nextnonterminal = 1.0 - (next_done if t == args.num_steps - 1 else dones[t + 1])
            
            # --- Standard Reward GAE ---
            nextval = next_value if t == args.num_steps - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * nextval * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            # --- Cost GAE [UPDATED] ---
            next_nonterm_cost = nextnonterminal.unsqueeze(-1) # [UPDATED] Broadcast shape for 4 resources
            next_cval = next_cost_value if t == args.num_steps - 1 else cost_values[t + 1]
            
            delta_c = costs[t] + args.gamma * next_cval * next_nonterm_cost - cost_values[t]
            cost_advantages[t] = lastgaelam_cost = delta_c + args.gamma * args.gae_lambda * next_nonterm_cost * lastgaelam_cost

        returns = advantages + values
        cost_returns = cost_advantages + cost_values # [UPDATED] Calculate final cost returns

        # ---------------- Flatten ----------------
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1).long()
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # [UPDATED] Use the forward-looking GAE returns and advantages, not the raw daily costs
        b_cost_targets = cost_returns.reshape(-1, 4)
        b_cost_values = cost_values.reshape(-1, 4)
        b_cost_advantages = cost_advantages.reshape(-1, 4)

        # ---------------- PID (P-term) ----------------
        with torch.no_grad():
            total_batch_costs = costs.sum(dim=0)
            episodes_per_env = torch.clamp(dones.sum(dim=0).unsqueeze(-1), min=1.0) 
            
            mean_episodic_cost = (total_batch_costs / episodes_per_env).mean(dim=0)
            
            violation = mean_episodic_cost - args.target_limit
            p_term_lambda = torch.clamp(args.pid_kp * violation, min=0.0)
        
        lam = agent.get_lagrange_multiplier().detach() + p_term_lambda
        
        # [NOTE] Your logic here is flawless: multiplying the 4 cost advantages by their respective 4 lambdas, 
        # and summing them up to create a single penalty scalar.
        b_weighted_cost = (b_cost_advantages * lam.unsqueeze(0)).sum(dim=1)
        b_combined_adv = b_advantages - b_weighted_cost

        if args.norm_adv:
            b_combined_adv = (b_combined_adv - b_combined_adv.mean()) / (b_combined_adv.std() + 1e-8)

        # ---------------- PPO Update ----------------
        inds = np.arange(args.batch_size)
        epoch_entropy = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = inds[start:start + args.minibatch_size]

                _, newlogp, entropy, newv, newcv = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )

                ratio = (newlogp - b_logprobs[mb]).exp()
                epoch_entropy.append(entropy.mean().item())

                combined_adv = b_combined_adv[mb]

                pg_loss = torch.max(
                    -combined_adv * ratio,
                    -combined_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                v_loss = 0.5 * (newv.view(-1) - b_returns[mb]).pow(2).mean()
                
                # [UPDATED] This loss function now trains against the GAE returns, predicting future cumulative usage!
                cv_loss = 0.5 * (newcv - b_cost_targets[mb]).pow(2).mean()

                loss = (
                    pg_loss
                    - current_ent_coef * entropy.mean()
                    + args.vf_coef * v_loss
                    + args.cv_coef * cv_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---------------- Dual Update (I-term) ----------------
        if not args.fixed_lambda:
            lambda_loss = -(agent.log_lagrange_multiplier * violation.detach()).sum()
            lagrange_optimizer.zero_grad()
            lambda_loss.backward()
            lagrange_optimizer.step()

        # ---------------- Logging ----------------
        writer.add_scalar("charts/lambda_mean", agent.get_lagrange_multiplier().mean().item(), global_step)
        writer.add_scalar("constraints/mean_episodic_cost", mean_episodic_cost.mean().item(), global_step)
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", cv_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", np.mean(epoch_entropy), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("debug/adv_mean", b_advantages.mean(), global_step)
        writer.add_scalar("debug/adv_std", b_advantages.std(), global_step)
        writer.add_scalar("debug/pg_loss", pg_loss.item(), global_step)
        writer.add_scalar("debug/v_loss", v_loss.item(), global_step)

    envs.close()
    writer.close()