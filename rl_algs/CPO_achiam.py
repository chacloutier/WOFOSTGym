"""
Code to train a CPO Agent (Constrained Policy Optimization)
Based on: https://arxiv.org/abs/1705.10528 (Achiam et al. 2017)
"""

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
from typing import Optional, Tuple
from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy

# ------------------------------------------------------------------------------
# 1. CPO Arguments
# ------------------------------------------------------------------------------
@dataclass
class Args(RL_Args):
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    
    # CPO Specific Hyperparameters
    target_kl: float = 0.01
    """Maximum allowed KL divergence per step"""
    cost_limit: float = 25.0 
    """The maximum allowed cost (safety constraint)"""
    damping: float = 0.1
    """Damping for the Fisher Information Matrix (FIM)"""
    cg_iters: int = 10
    """Number of Conjugate Gradient iterations"""
    line_search_iters: int = 10
    """Number of line search backtracking steps"""
    line_search_coeff: float = 0.8
    """Backtracking coefficient"""
    vf_lr: float = 1e-3
    """Learning rate for Value Function and Cost Value Function"""
    vf_iters: int = 80
    """Number of iterations to train value functions"""
    checkpoint_frequency: int = 500
    """How often to save the agent during training"""
    
    # Computed at runtime
    batch_size: int = 0
    num_iterations: int = 0

# ------------------------------------------------------------------------------
# 2. Mathematical Helpers (Conjugate Gradient & HVP)
# ------------------------------------------------------------------------------
def flat_grad(grads, params, detach=True):
    """
    Flatten gradients.
    CRITICAL FIX: Iterates over the `grads` tuple provided by autograd.grad.
    """
    grad_flatten = []
    for g, p in zip(grads, params):
        if g is None:
            g = torch.zeros_like(p)
        if detach:
            g = g.detach()
        grad_flatten.append(g.view(-1))
    return torch.cat(grad_flatten)

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, new_params):
    prev_ind = 0
    for p in model.parameters():
        flat_size = int(np.prod(list(p.size())))
        p.data.copy_(new_params[prev_ind:prev_ind + flat_size].view(p.size()))
        prev_ind += flat_size

def conjugate_gradients(Avp_func, b, nsteps, residual_tol=1e-10):
    """
    Conjugate Gradient algorithm to solve Ax = b where A is the FIM.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = Avp_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def extract_cost(infos: dict, cost_keys: list = ["nitrogen", "phosphorous"]) -> np.ndarray:
    """
    Extracts a scalar cost from the complex WOFOST info structure.
    Sums up the values found in the specified keys (e.g. N + P).
    """
    if not infos:
        return np.array([0.0])

    total_cost = 0.0
    
    for metric in cost_keys:
        if metric in infos:
            data_dict = infos[metric]
            for k, v in data_dict.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                total_cost += v
                
    if isinstance(total_cost, float):
        return np.array([total_cost])
        
    return total_cost

# ------------------------------------------------------------------------------
# 3. The CPO Agent
# ------------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CPO(nn.Module, Agent):
    def __init__(self, envs: gym.Env, state_fpath: str = None, **kwargs: dict):
        super().__init__()
        self.env = envs
        self.obs_shape = np.array(envs.single_observation_space.shape).prod()
        self.action_shape = envs.single_action_space.n

        # Actor (Policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_shape), std=0.01),
        )

        # Reward Critic (Value Function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Cost Critic (Safety Value Function)
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        if state_fpath is not None:
            try:
                self.load_state_dict(torch.load(state_fpath, weights_only=True))
            except:
                msg = f"Error loading state dictionary from {state_fpath}"
                raise Exception(msg)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        return Categorical(logits=logits).sample()

    def get_log_prob_entropy(self, x, action):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), dist

    def get_vals(self, x):
        return self.critic(x), self.cost_critic(x)

# ------------------------------------------------------------------------------
# 4. Training Logic
# ------------------------------------------------------------------------------
def train(kwargs: Namespace) -> None:
    args = kwargs.alg
    run_name = f"CPO/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)
    agent = CPO(envs).to(device)
    
    # Optimizers
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.vf_lr)
    optimizer_cost_critic = optim.Adam(agent.cost_critic.parameters(), lr=args.vf_lr)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Manual Episodic Return Tracking (since env is unwrapped)
    running_episodic_return = np.zeros(args.num_envs)
    running_episodic_length = np.zeros(args.num_envs)

    # --------------------------------------------------------------------------
    # Helper: FVP (Fisher Vector Product) for CPO
    # --------------------------------------------------------------------------
    def compute_fvp(vector, obs_b, act_b):
        """Computes Hessian-vector product Hx using the Pearlmutter trick."""
        # 1. Current policy (with gradients)
        _, _, dist = agent.get_log_prob_entropy(obs_b, act_b)
        
        # 2. Old policy (fixed/detached)
        with torch.no_grad():
             _, _, dist_old = agent.get_log_prob_entropy(obs_b, act_b)
        
        # 3. KL Divergence (dist_old || dist)
        kl = torch.distributions.kl.kl_divergence(dist_old, dist).mean()

        # 4. Gradients
        grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
        flat_grad_kl = flat_grad(grads, agent.actor.parameters(), detach=False)

        # 5. Pearlmutter trick
        kl_v = (flat_grad_kl * vector).sum()
        grads_v = torch.autograd.grad(kl_v, agent.actor.parameters())
        flat_grad_grad_kl = flat_grad(grads_v, agent.actor.parameters(), detach=True)

        return flat_grad_grad_kl + vector * args.damping

    # --------------------------------------------------------------------------
    # Main Loop
    # --------------------------------------------------------------------------
    for iteration in range(1, args.num_iterations + 1):
        
        # 1. Collect Data
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                val, c_val = agent.get_vals(next_obs)
                values[step] = val.flatten()
                cost_values[step] = c_val.flatten()
                action = agent.get_action(next_obs)
                logprob, _, _ = agent.get_log_prob_entropy(next_obs, action)
            
            actions[step] = action
            logprobs[step] = logprob

            # Step Env
            next_obs, reward, termin, trunc, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(termin, trunc)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # --- EXTRACT COST ---
            c_step = extract_cost(infos, cost_keys=["nitrogen", "phosphorous", "potassium"])
            costs[step] = torch.tensor(c_step).float().to(device).view(-1)

            # --- MANUAL LOGGING ---
            running_episodic_return += reward
            running_episodic_length += 1
            
            for i in range(args.num_envs):
                if next_done[i]:
                    print(f"global_step={global_step}, episodic_return={running_episodic_return[i]}")
                    writer.add_scalar("charts/episodic_return", running_episodic_return[i], global_step)
                    writer.add_scalar("charts/episodic_length", running_episodic_length[i], global_step)
                    running_episodic_return[i] = 0
                    running_episodic_length[i] = 0

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            if global_step % args.checkpoint_frequency == 0:
                 writer.add_scalar("charts/average_reward", eval_policy(agent, envs, kwargs, device), global_step)

        # 2. GAE Estimation (Reward & Cost)
        with torch.no_grad():
            next_val, next_c_val = agent.get_vals(next_obs)
            next_val = next_val.reshape(1, -1)
            next_c_val = next_c_val.reshape(1, -1)
            
            # Reward Advantages
            adv = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = adv + values
            
            # Cost Advantages (C_Adv)
            c_adv = torch.zeros_like(costs).to(device)
            lastgaelam_c = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues_c = next_c_val
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues_c = cost_values[t + 1]
                delta_c = costs[t] + args.gamma * nextvalues_c * nextnonterminal - cost_values[t]
                c_adv[t] = lastgaelam_c = delta_c + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_c
            c_returns = c_adv + cost_values

        # Flatten Batches
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_act = actions.reshape((-1,) + envs.single_action_space.shape).long()
        b_adv = adv.reshape(-1)
        b_c_adv = c_adv.reshape(-1)
        b_old_log_probs = logprobs.reshape(-1)

        # Normalize Advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_c_adv = (b_c_adv - b_c_adv.mean()) / (b_c_adv.std() + 1e-8)

        # ----------------------------------------------------------------------
        # 3. CPO Update Step (Achiam et al. 2017 Logic)
        # ----------------------------------------------------------------------
        curr_log_probs, dist_entropy, dist = agent.get_log_prob_entropy(b_obs, b_act)
        ratio = torch.exp(curr_log_probs - b_old_log_probs)
        
        # Surrogate Losses
        surr_loss = (ratio * b_adv).mean()
        cost_surr = (ratio * b_c_adv).mean()
        
        # Gradients
        grad_g = flat_grad(torch.autograd.grad(surr_loss, agent.actor.parameters(), retain_graph=True), agent.actor.parameters())
        grad_b = flat_grad(torch.autograd.grad(cost_surr, agent.actor.parameters(), retain_graph=True), agent.actor.parameters())
        
        # FVP & Conjugate Gradient
        def Hx(v):
            return compute_fvp(v, b_obs, b_act)

        Hg = conjugate_gradients(Hx, grad_g, args.cg_iters)
        Hb = conjugate_gradients(Hx, grad_b, args.cg_iters)

        # Quadratic forms (Achiam 2017)
        q = (grad_g * Hg).sum()
        r = (grad_g * Hb).sum()
        s = (grad_b * Hb).sum()
        
        # --- COST CONSTRAINT CALCULATION ---
        # Use absolute expected cost Jc, not advantage
        Jc = cost_values.mean().item() 
        cost_delta = Jc - args.cost_limit
        
        # Solve Dual Problem
        if cost_delta <= 0 and (cost_delta != 0 or r > 0): 
            # Feasible & Convex -> Maximize Reward
            A = q - r**2 / (s + 1e-8)
            B = 2 * args.target_kl - (cost_delta**2 / (s + 1e-8)) if cost_delta < 0 else 2 * args.target_kl
            
            if cost_delta < 0 and B < 0:
                 # Trust region too small -> Recovery
                 step_dir = -np.sqrt(2 * args.target_kl / (s + 1e-8)) * Hb
            else:
                lam = np.sqrt(abs(A) / B) if B != 0 else 0 
                nu = max(0, lam * r / (s + 1e-8) - cost_delta / (s + 1e-8))
                step_dir = (1 / (lam + 1e-8)) * (Hg - nu * Hb)
        else:
            # Infeasible / Violated -> Recovery (Minimize Cost)
            lam = np.sqrt(2 * args.target_kl / (s + 1e-8))
            step_dir = -lam * Hb 

        # 4. Line Search
        old_params = flat_params(agent.actor)
        
        def get_metrics(params):
            set_params(agent.actor, params)
            with torch.no_grad():
                logp, _, new_dist = agent.get_log_prob_entropy(b_obs, b_act)
                new_ratio = torch.exp(logp - b_old_log_probs)
                
                loss_pi = (new_ratio * b_adv).mean()
                
                # Check absolute cost estimate
                # J_new approx J_old + delta_surrogate
                surr_cost_new = (new_ratio * b_c_adv).mean().item()
                cost_pi = Jc + surr_cost_new
                
                kl_val = torch.distributions.kl.kl_divergence(dist, new_dist).mean()
            return loss_pi, cost_pi, kl_val

        for i in range(args.line_search_iters):
            step_frac = args.line_search_coeff ** i
            new_params = old_params + step_frac * step_dir
            
            loss_pi, cost_pi, kl_val = get_metrics(new_params)
            
            if kl_val > args.target_kl * 1.5:
                continue
            
            # Acceptance Logic
            if cost_delta <= 0:
                # Feasible: Ensure we stay feasible AND improve reward
                if cost_pi <= args.cost_limit and loss_pi > surr_loss:
                    break
            else:
                # Infeasible: Just ensure we decrease cost
                if cost_pi < Jc:
                    break
        else:
            # Revert if line search fails
            set_params(agent.actor, old_params)

        # 5. Value Function Updates
        b_returns = returns.reshape(-1)
        for _ in range(args.vf_iters):
            v_pred = agent.critic(b_obs).flatten()
            v_loss = ((v_pred - b_returns) ** 2).mean()
            optimizer_critic.zero_grad()
            v_loss.backward()
            optimizer_critic.step()
            
        b_c_returns = c_returns.reshape(-1)
        for _ in range(args.vf_iters):
            c_pred = agent.cost_critic(b_obs).flatten()
            c_loss = ((c_pred - b_c_returns) ** 2).mean()
            optimizer_cost_critic.zero_grad()
            c_loss.backward()
            optimizer_cost_critic.step()

        # Logging
        # Explicitly use Jc here, since current_cost is a local variable
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", c_loss.item(), global_step)
        writer.add_scalar("charts/cost_delta", cost_delta, global_step)
        writer.add_scalar("charts/avg_cost", Jc, global_step)

    envs.close()
    writer.close()