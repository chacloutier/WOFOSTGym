import wandb
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
from typing import Optional
from argparse import Namespace
import gymnasium as gym
from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy

@dataclass
class Args(RL_Args):
    total_timesteps: int = 1000000
    learning_rate: float = 1e-3
    """Learning rate for the reward critic optimizer"""
    num_envs: int = 1
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    max_grad_norm: float = 0.5
    checkpoint_frequency: int = 500
    
    # TRPO Specific Hyperparameters
    max_kl: float = 0.01
    cg_iters: int = 10
    line_search_backtrack_ratio: float = 0.8
    line_search_max_steps: int = 10
    damping: float = 0.1

    # TRPO-Lagrangian (Safe RL) Specific Hyperparameters
    cost_limit: float = 25.0
    """The maximum allowed expected cost return per episode"""
    penalty_learning_rate: float = 5e-2
    """Learning rate for the Lagrange multiplier (lambda)"""
    cost_gae_lambda: float = 0.95
    """GAE lambda specifically for cost advantages"""
    cost_gamma: float = 0.99
    """Discount factor for costs"""
    initial_penalty: float = 0.01
    """Initial value for the Lagrange multiplier"""
    
    # --- Constraint Thresholds (Matching CPO/PPO-Lag) ---
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

def extract_step_cost(infos: dict, args: Args, num_envs: int) -> np.ndarray:
    """
    Calculates cost based on specific nutrient constraints matching CPO logic.
    Returns array of shape (num_envs,) with 1.0 for violation, 0.0 for safe.
    """
    costs = np.zeros(num_envs)
    
    # Handle Vectorized Environment (Dictionary of Arrays)
    if "track/total_n" in infos:
        for i in range(num_envs):
            # Extract cumulative totals for this environment instance
            n = infos["track/total_n"][i]
            p = infos["track/total_p"][i]
            k = infos["track/total_k"][i]
            w = infos["track/total_w"][i]
            
            # Check constraints
            # if (n > args.max_n or 
            #     p > args.max_p or 
            #     k > args.max_k or 
            #     w > args.max_w):
            #     costs[i] = 1.0
            # else:
            #     costs[i] = 0.0
            
            if n > args.max_n:
                costs[i] += (n - args.max_n) / args.max_n
            if k > args.max_k:
                costs[i] += (k - args.max_k) / args.max_k
            if p > args.max_p:
                costs[i] += (p - args.max_p) / args.max_p
            if w > args.max_w:
                costs[i] += (w - args.max_w) / args.max_w
                
    # Handle List of Dicts (Standard Gym) or missing keys
    elif isinstance(infos, list):
        for i, info in enumerate(infos):
            n = info.get("track/total_n", 0.0)
            p = info.get("track/total_p", 0.0)
            k = info.get("track/total_k", 0.0)
            w = info.get("track/total_w", 0.0)
            
            if (n > args.max_n or p > args.max_p or k > args.max_k or w > args.max_w):
                costs[i] = 1.0
            else:
                costs[i] = 0.0
                
    return costs * 0.001

class TRPO_Lagrangian_Agent(nn.Module, Agent):
    def __init__(self, envs: gym.Env):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n
        
        # Reward Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Cost Critic (Safe RL)
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_cost_value(self, x):
        return self.cost_critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), self.cost_critic(x)

# --- TRPO Optimization Helpers ---

def get_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    prev_ind = 0
    for p in model.parameters():
        flat_size = p.numel()
        p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
        prev_ind += flat_size

def conjugate_gradient(Avp_fn, b, iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        _Avp = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, _Avp) + 1e-8)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def train(kwargs: Namespace) -> None:
    args = kwargs.alg
    run_name = f"TRPO-Lag/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)
    agent = TRPO_Lagrangian_Agent(envs).to(device)
    
    # 1. Optimizers
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.learning_rate)
    optimizer_cost_critic = optim.Adam(agent.cost_critic.parameters(), lr=args.learning_rate)
    
    # Lagrange Multiplier (Penalty Parameter)
    log_penalty = torch.nn.Parameter(torch.log(torch.tensor(args.initial_penalty, device=device)))
    optimizer_penalty = optim.Adam([log_penalty], lr=args.penalty_learning_rate)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        
        # --- 1. Collect Rollouts ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, cost_value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                cost_values[step] = cost_value.flatten()
                
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # --- UPDATED: Specific Cost Calculation ---
            step_cost = extract_step_cost(infos, args, args.num_envs)
            costs[step] = torch.tensor(step_cost).to(device).view(-1)
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

            # --- Logging specific constraints ---
            if global_step % args.checkpoint_frequency == 0 and "track/total_n" in infos:
                 writer.add_scalar("constraints/total_n", infos["track/total_n"][0], global_step)
                 writer.add_scalar("constraints/violation_rate", step_cost.mean(), global_step)

        # --- 2. GAE Calculation ---
        with torch.no_grad():
            _, _, _, next_value, next_cost_value = agent.get_action_and_value(next_obs)
            next_value = next_value.reshape(1, -1)
            next_cost_value = next_cost_value.reshape(1, -1)
            
            # Reward GAE
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
            returns = advantages + values
            
            # Cost GAE
            cost_advantages = torch.zeros_like(costs).to(device)
            last_cost_gaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_c_values = next_cost_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_c_values = cost_values[t + 1]
                
                delta_c = costs[t] + args.cost_gamma * next_c_values * nextnonterminal - cost_values[t]
                cost_advantages[t] = last_cost_gaelam = delta_c + args.cost_gamma * args.cost_gae_lambda * nextnonterminal * last_cost_gaelam
            cost_returns = cost_advantages + cost_values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_cost_advantages = cost_advantages.reshape(-1)
        b_cost_returns = cost_returns.reshape(-1)

        # --- 3. Update Lagrange Multiplier (Penalty) ---
        
        # We use the mean of the discounted cost returns as our estimate for J_C
        mean_cost_return = b_cost_returns.mean()
        violation = mean_cost_return - args.cost_limit
        
        penalty_loss = -log_penalty * violation.detach()
        
        optimizer_penalty.zero_grad()
        penalty_loss.backward()
        optimizer_penalty.step()
        
        penalty_val = torch.exp(log_penalty).item()

        # --- 4. TRPO Actor Update ---
        
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            b_cost_advantages = (b_cost_advantages - b_cost_advantages.mean()) / (b_cost_advantages.std() + 1e-8)
        
        # Combined Advantage: Reward - lambda * Cost
        b_combined_advantages = b_advantages - penalty_val * b_cost_advantages

        with torch.no_grad():
            old_logits = agent.actor(b_obs)
            old_probs = Categorical(logits=old_logits)

        def get_loss():
            new_logits = agent.actor(b_obs)
            new_probs = Categorical(logits=new_logits)
            ratio = torch.exp(new_probs.log_prob(b_actions) - b_logprobs)
            # Maximize the combined advantage
            return (ratio * b_combined_advantages).mean()

        def fvp(v):
            new_logits = agent.actor(b_obs)
            new_probs = Categorical(logits=new_logits)
            kl = torch.distributions.kl.kl_divergence(old_probs, new_probs).mean()
            grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            kl_v = (flat_grad_kl * v).sum()
            grads_v = torch.autograd.grad(kl_v, agent.actor.parameters())
            flat_grad_grad_kl = torch.cat([g.view(-1) for g in grads_v]).detach()
            return flat_grad_grad_kl + v * args.damping

        loss = get_loss()
        grads = torch.autograd.grad(loss, agent.actor.parameters())
        loss_grad = torch.cat([g.view(-1) for g in grads]).detach()
        
        step_dir = conjugate_gradient(fvp, loss_grad, iters=args.cg_iters)
        shs = 0.5 * (step_dir * fvp(step_dir)).sum(0, keepdim=True)
        
        if shs > 0:
            lm = torch.sqrt(shs / args.max_kl)
            full_step = step_dir / lm
            old_params = get_flat_params(agent.actor)
            
            for i in range(args.line_search_max_steps):
                fraction = args.line_search_backtrack_ratio ** i
                new_params = old_params + fraction * full_step
                set_flat_params(agent.actor, new_params)
                
                with torch.no_grad():
                    new_loss = get_loss()
                    new_logits = agent.actor(b_obs)
                    new_probs = Categorical(logits=new_logits)
                    kl = torch.distributions.kl.kl_divergence(old_probs, new_probs).mean()
                
                if kl <= args.max_kl and new_loss > loss:
                    break
                
                if i == args.line_search_max_steps - 1:
                    set_flat_params(agent.actor, old_params)

        # --- 5. Critic Updates ---
        for epoch in range(args.update_epochs):
            inds = np.arange(args.batch_size)
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = inds[start:end]
                
                # Reward Critic
                v_loss = 0.5 * ((agent.get_value(b_obs[mb_inds]).view(-1) - b_returns[mb_inds]) ** 2).mean()
                optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

                # Cost Critic
                c_loss = 0.5 * ((agent.get_cost_value(b_obs[mb_inds]).view(-1) - b_cost_returns[mb_inds]) ** 2).mean()
                optimizer_cost_critic.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(agent.cost_critic.parameters(), args.max_grad_norm)
                optimizer_cost_critic.step()

        # Logging
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", c_loss.item(), global_step)
        writer.add_scalar("losses/penalty", penalty_val, global_step)
        writer.add_scalar("charts/cost_return", mean_cost_return, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()