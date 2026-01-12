"""
Constrained Policy Optimization (CPO)
Achiam et al., 2017
Corrected Research-ready version for WOFOSTGym
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
    total_timesteps: int = 1_000_000
    num_envs: int = 1
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.97
    
    # CPO Specific Hyperparameters
    target_kl: float = 0.01
    cost_limit: float = 25.0 
    damping: float = 0.1
    cg_iters: int = 10
    line_search_iters: int = 10
    line_search_coeff: float = 0.8
    vf_lr: float = 1e-3
    vf_iters: int = 80
    checkpoint_frequency: int = 500
    
    # Computed at runtime
    batch_size: int = 0
    num_iterations: int = 0

# ------------------------------------------------------------------------------
# 2. Mathematical Helpers
# ------------------------------------------------------------------------------
def flat_grad(grads, params, detach=True):
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

def extract_cost(infos: dict, cost_keys: list = ["nitrogen", "phosphorous", "potassium"]) -> np.ndarray:
    if not infos:
        return np.array([0.0])
    total_cost = 0.0
    for metric in cost_keys:
        if metric in infos:
            data_dict = infos[metric]
            if isinstance(data_dict, dict):
                for k, v in data_dict.items():
                    if isinstance(k, str) and k.startswith("_"):
                        continue
                    total_cost += v
            else:
                total_cost += float(data_dict)
    return np.array([total_cost])

# ------------------------------------------------------------------------------
# 3. The CPO Agent
# ------------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CPO(nn.Module, Agent):
    def __init__(self, envs: gym.Env):
        super().__init__()
        self.obs_shape = np.array(envs.single_observation_space.shape).prod()
        self.action_shape = envs.single_action_space.n

        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_shape), std=0.01),
        )

        # Reward Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Cost Critic
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

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
    
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.vf_lr)
    optimizer_cost_critic = optim.Adam(agent.cost_critic.parameters(), lr=args.vf_lr)

    # Buffers
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Helper: Hessian-Vector Product
    def compute_fvp(vector, obs_b, act_b):
        _, _, dist = agent.get_log_prob_entropy(obs_b, act_b)
        with torch.no_grad():
             _, _, dist_old = agent.get_log_prob_entropy(obs_b, act_b)
        
        kl = torch.distributions.kl.kl_divergence(dist_old, dist).mean()
        grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
        flat_grad_kl = flat_grad(grads, agent.actor.parameters(), detach=False)
        kl_v = (flat_grad_kl * vector).sum()
        grads_v = torch.autograd.grad(kl_v, agent.actor.parameters())
        flat_grad_grad_kl = flat_grad(grads_v, agent.actor.parameters(), detach=True)
        return flat_grad_grad_kl + vector * args.damping

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

            next_obs, reward, termin, trunc, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(termin, trunc)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            c_step = extract_cost(infos)
            costs[step] = torch.tensor(c_step).float().to(device).view(-1)

            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        # 2. GAE Estimation
        with torch.no_grad():
            next_val, next_c_val = agent.get_vals(next_obs)
            next_val = next_val.reshape(1, -1)
            next_c_val = next_c_val.reshape(1, -1)
            
            # Reward GAE
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
            
            # Cost GAE
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

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_act = actions.reshape((-1,) + envs.single_action_space.shape).long()
        b_adv = adv.reshape(-1)
        b_c_adv = c_adv.reshape(-1)
        b_old_log_probs = logprobs.reshape(-1)

        # Normalize Reward Advantage
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ----------------------------------------------------------------------
        # 3. CPO Update Step
        # ----------------------------------------------------------------------
        curr_log_probs, _, dist = agent.get_log_prob_entropy(b_obs, b_act)
        ratio = torch.exp(curr_log_probs - b_old_log_probs)
        
        # Keep uncentered cost advantage for line search estimation
        b_c_adv_uncentered = b_c_adv.clone()
        # Center cost advantage for gradient direction stability
        b_c_adv = b_c_adv - b_c_adv.mean()

        surr_loss = (ratio * b_adv).mean()
        grad_g = flat_grad(torch.autograd.grad(surr_loss, agent.actor.parameters(), retain_graph=True), agent.actor.parameters())

        cost_loss = (ratio * b_c_adv).mean()
        grad_b = flat_grad(torch.autograd.grad(cost_loss, agent.actor.parameters(), retain_graph=True), agent.actor.parameters())
        
        current_cost = cost_values.mean().item() 
        cost_delta = current_cost - args.cost_limit
        
        # Conjugate Gradient
        step_dir_g = conjugate_gradients(lambda v: compute_fvp(v, b_obs, b_act), grad_g, args.cg_iters)
        step_dir_b = conjugate_gradients(lambda v: compute_fvp(v, b_obs, b_act), grad_b, args.cg_iters)

        q = (step_dir_g * compute_fvp(step_dir_g, b_obs, b_act)).sum()
        s = (step_dir_b * compute_fvp(step_dir_b, b_obs, b_act)).sum()
        r = (step_dir_g * compute_fvp(step_dir_b, b_obs, b_act)).sum()
        
        optim_case = 0
        if cost_delta > 0:
            # --- VIOLATION: Recovery (Minimize Cost) ---
            lam = torch.sqrt(2 * args.target_kl / (s + 1e-8))
            nu = 0
            final_step_dir = -lam * step_dir_b
            optim_case = 0
        else:
            # --- SAFE: Maximize Reward ---
            A = q - r**2 / (s + 1e-8)
            B = 2 * args.target_kl - cost_delta**2 / (s + 1e-8)
            
            # If feasible and TRPO is safe (or B < 0 degenerate)
            if cost_delta < 0 and B < 0:
                lam = torch.sqrt(2 * args.target_kl / (q + 1e-8))
                nu = 0
                final_step_dir = lam * step_dir_g
                optim_case = 2
            else:
                # CPO Standard Case
                lam = torch.sqrt(A / (B + 1e-8))
                nu = max(0, (lam * cost_delta - r) / (s + 1e-8))
                final_step_dir = (1 / (lam + 1e-8)) * (step_dir_g - nu * step_dir_b)
                optim_case = 1

        # 4. Line Search
        old_params = flat_params(agent.actor)
        
        def get_loss_and_kl():
            with torch.no_grad():
                new_log_prob, _, new_dist = agent.get_log_prob_entropy(b_obs, b_act)
                new_ratio = torch.exp(new_log_prob - b_old_log_probs)
                loss_pi = (new_ratio * b_adv).mean()
                kl_val = torch.distributions.kl.kl_divergence(dist, new_dist).mean()
                # Use uncentered adv for correct cost delta estimation
                cost_pi = (new_ratio * b_c_adv_uncentered).mean()
            return loss_pi, kl_val, cost_pi

        step_accepted = False
        for i in range(args.line_search_iters):
            step_frac = args.line_search_coeff ** i
            new_params = old_params + step_frac * final_step_dir
            set_params(agent.actor, new_params)
            
            loss, kl, cost_surr = get_loss_and_kl()
            
            if kl > args.target_kl * 1.5:
                continue
            
            if optim_case > 0:
                # Normal/Safe: Improve reward AND stay within limit
                # We need cost_new <= Limit => cost_old + cost_surr <= Limit
                # => cost_surr <= Limit - cost_old => cost_surr <= -cost_delta
                if loss > surr_loss and cost_surr <= -cost_delta:
                      step_accepted = True
                      break
            else:
                # Recovery: Reduce cost
                if cost_surr < 0:
                      step_accepted = True
                      break
        
        if not step_accepted:
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

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", c_loss.item(), global_step)
        writer.add_scalar("charts/cost_delta", cost_delta, global_step)
        writer.add_scalar("charts/avg_cost", current_cost, global_step)
        writer.add_scalar("charts/optim_case", optim_case, global_step)

    envs.close()
    writer.close()