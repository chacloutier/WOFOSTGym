"""
Code to train a CPO Agent (Constrained Policy Optimization)
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
    
    # --- UPDATED: Unified Cost Limit ---
    cost_limit: float = 1.0 
    """The maximum allowed cost ratio (1.0 = 100% of budget)"""
    
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
    max_n: float = 80.0
    """Maximum Nitrogen limit"""
    max_p: float = 80.0
    """Maximum Phosphorous limit"""
    max_k: float = 80.0
    """Maximum Potassium limit"""
    max_w: float = 40.0
    """Maximum Water limit"""
    
    # --- NEW: Evaluation & Logging ---
    checkpoint_frequency: int = 500
    """How often to save the agent and run eval_policy"""

    # Computed at runtime
    batch_size: int = 0
    num_iterations: int = 0

# ------------------------------------------------------------------------------
# 2. Mathematical Helpers (Conjugate Gradient & HVP)
# ------------------------------------------------------------------------------
def flat_grad(grads, params, detach=True):
    """
    Flatten gradients.
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
    p = b.clone()
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

def extract_cost(infos: dict, args: Args) -> np.ndarray:
    """
    Extracts a scalar cost by normalizing multiple constraints against their limits.
    Returns the maximum violation ratio (1.0 = exactly at limit).
    """
    if not infos:
        return np.array([0.0])

    # The 'infos' dict contains numpy arrays for the totals, e.g., 'track/total_p': array([8.])
    # We use .item() to extract the scalar float from the 1-element array.
    
    # Get Nitrogen Total
    n_arr = infos.get("track/total_n", np.array([0.0]))
    n_val = n_arr.item() if isinstance(n_arr, np.ndarray) else n_arr

    # Get Phosphorous Total
    p_arr = infos.get("track/total_p", np.array([0.0]))
    p_val = p_arr.item() if isinstance(p_arr, np.ndarray) else p_arr

    # Get Potassium Total
    k_arr = infos.get("track/total_k", np.array([0.0]))
    k_val = k_arr.item() if isinstance(k_arr, np.ndarray) else k_arr

    # Get Water Total
    w_arr = infos.get("track/total_w", np.array([0.0]))
    w_val = w_arr.item() if isinstance(w_arr, np.ndarray) else w_arr

    # Calculate ratios (Current / Limit)
    # Example: If P is 8.0 and Limit is 80.0, ratio is 0.1
    r_n = n_val / args.max_n
    r_p = p_val / args.max_p
    r_k = k_val / args.max_k
    r_w = w_val / args.max_w

    max_ratio = max(r_n, r_p, r_k, r_w)
    
    if max_ratio <= 1.0:
        return np.array([0.0])
    else:
        return np.array([1.0])

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

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        return Categorical(logits=logits).sample()

    def get_log_prob_entropy(self, x, action):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), dist

    def get_logits(self, x):
        return self.actor(x)

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
    
    # Optimizers for Value functions (Policy is optimized manually via CPO)
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

    # --------------------------------------------------------------------------
    # FVP Implementations
    # --------------------------------------------------------------------------

    def compute_fvp_direct(vector, obs_b, act_b):
        """
        Computes the product of the Fisher Information Matrix (Hessian of KL)
        """
        # 1. Compute KL
        _, _, dist = agent.get_log_prob_entropy(obs_b, act_b)
        with torch.no_grad():
             _, _, dist_old = agent.get_log_prob_entropy(obs_b, act_b)
        kl = torch.distributions.kl.kl_divergence(dist_old, dist).mean()
        
        # 2. Gradient of KL
        grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
        flat_grad_kl = flat_grad(grads, agent.actor.parameters(), detach=False)

        # 3. Hessian-Vector Product (Gradient of the dot product)
        kl_v = (flat_grad_kl * vector).sum()
        grads_v = torch.autograd.grad(kl_v, agent.actor.parameters())
        flat_grad_grad_kl = flat_grad(grads_v, agent.actor.parameters(), detach=True)

        return flat_grad_grad_kl + vector * args.damping

    # Main Loop
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
            c_step = extract_cost(infos, args)
            costs[step] = torch.tensor(c_step).float().to(device).view(-1)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # --- LOGGING: Episodic Returns ---
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # --- LOGGING: Evaluation & Checkpointing (Matches PPO) ---
            if global_step % args.checkpoint_frequency == 0:
                # 1. Save Agent
                torch.save(agent.state_dict(), f"{kwargs.save_folder}{run_name}/agent.pt")
                if kwargs.track:
                    wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")
                
                # 2. Run Eval Policy and Log Average Reward
                writer.add_scalar("charts/average_reward", eval_policy(agent, envs, kwargs, device), global_step)
                
                # 3. Log Detailed Constraints (If provided by wrapper)
                if "track/total_n" in infos:
                    writer.add_scalar("constraints/total_n", infos["track/total_n"], global_step)
                if "track/total_w" in infos:
                    writer.add_scalar("constraints/total_w", infos["track/total_w"], global_step)
                if "track/total_p" in infos:
                    writer.add_scalar("constraints/total_p", infos["track/total_p"], global_step)
                if "track/total_k" in infos:
                    writer.add_scalar("constraints/total_k", infos["track/total_k"], global_step)
                if "track/is_violating" in infos:
                    writer.add_scalar("constraints/violation_rate", infos["track/is_violating"], global_step)

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

        # # Normalize Advantages
        # b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        # b_c_adv = (b_c_adv - b_c_adv.mean()) / (b_c_adv.std() + 1e-8)

        # ----------------------------------------------------------------------
        # 3. CPO Update Step
        # ----------------------------------------------------------------------
        curr_log_probs, dist_entropy, dist = agent.get_log_prob_entropy(b_obs, b_act)
        ratio = torch.exp(curr_log_probs - b_old_log_probs)
        
        # Surrogate Loss (Maximize Reward)
        surr_loss = (ratio * b_adv).mean()
        grad_g = flat_grad(torch.autograd.grad(surr_loss, agent.actor.parameters(), retain_graph=True), agent.actor.parameters())

        # Cost Surrogate (Minimize Cost)
        cost_loss = (ratio * b_c_adv).mean()
        grad_b = flat_grad(torch.autograd.grad(cost_loss, agent.actor.parameters(), retain_graph=True), agent.actor.parameters())
        
        current_cost = cost_values.mean().item() 
        cost_delta = current_cost - args.cost_limit
        
        # Compute Search Direction (Conjugate Gradient)
        step_dir_g = conjugate_gradients(lambda v: compute_fvp_direct(v, b_obs, b_act), grad_g, args.cg_iters)
        step_dir_b = conjugate_gradients(lambda v: compute_fvp_direct(v, b_obs, b_act), grad_b, args.cg_iters)

        # Analytical Variables
        # q = g^T H^-1 g
        # s = b^T H^-1 b
        # r = g^T H^-1 b
        q = torch.dot(grad_g, step_dir_g)
        s = torch.dot(grad_b, step_dir_b)
        r = torch.dot(grad_g, step_dir_b)
    
        # --- DUAL OPTIMIZATION---
        
        # Helper functions for the dual objective
        def f_a_lambda(lam):
            return ((r**2) / s - q) / (2 * lam) + lam * ((cost_delta**2) / s - args.target_kl) / 2 - (r * cost_delta) / s

        def f_b_lambda(lam):
            return - (q / lam + lam * args.target_kl) / 2

        # 1. Check Feasibility
        # If the trust region is too small to correct the cost constraint (c^2/s > delta), 
        # and we are violating the constraint (cost_delta > 0), we are INFEASIBLE.
        is_feasible = True
        if cost_delta > 0 and (cost_delta**2) / (s + 1e-8) > args.target_kl:
            is_feasible = False
        
        optim_case = 0
        final_step_dir = torch.zeros_like(step_dir_g)

        if not is_feasible:
            # Step direction: - sqrt(2 * delta / s) * H^-1 b
            lam_rec = torch.sqrt(2 * args.target_kl / (s + 1e-8))
            final_step_dir = -lam_rec * step_dir_b
            optim_case = 0
        else:
            # --- FEASIBLE: SOLVE DUAL ---
            # We solve for optimal lambda (KL constraint) and nu (Cost constraint)
            
            # Coefficients for the quadratic equation of lambda (Active constraint case)
            # A corresponds to the sqrt term in the analytical solution
            radicand_A = (q - (r**2) / (s + 1e-8)) / (args.target_kl - (cost_delta**2) / (s + 1e-8))
            
            # Clamp to avoid NaNs if numerical noise makes it slightly negative
            radicand_A = torch.max(radicand_A, torch.tensor(0.0).to(device))
            A = torch.sqrt(radicand_A)
            
            # B corresponds to TRPO solution (Inactive constraint)
            B = torch.sqrt(q / args.target_kl)
            
            # Candidate lambdas
            if cost_delta > 0:
                # If we are violating, lambda must be large enough to reduce cost
                lam_a = torch.max(r / cost_delta, A)
                lam_b = torch.max(torch.tensor(0.0).to(device), torch.min(B, r / cost_delta))
            else:
                # If we are safe, lambda handles the trust region
                lam_b = torch.max(r / (cost_delta - 1e-8), B)
                lam_a = torch.max(torch.tensor(0.0).to(device), torch.min(A, r / (cost_delta - 1e-8)))

            # Compare Dual Objectives
            val_a = f_a_lambda(lam_a)
            val_b = f_b_lambda(lam_b)

            if val_a >= val_b:
                opt_lam = lam_a
                optim_case = 1 # Active Constraint
            else:
                opt_lam = lam_b
                optim_case = 2 # Inactive Constraint (TRPO)

            # Solve for Nu (Cost Lagrange Multiplier)
            # nu = (lambda * c - r) / s
            nu = (opt_lam * cost_delta - r) / (s + 1e-8)
            opt_nu = torch.max(nu, torch.tensor(0.0).to(device))
            
            # Calculate Final Direction
            # d = (1/lambda) * (H^-1 g - nu * H^-1 b)
            final_step_dir = (1.0 / (opt_lam + 1e-8)) * (step_dir_g - opt_nu * step_dir_b)

        # 4. Line Search
        old_params = flat_params(agent.actor)
        def get_loss_and_kl():
            with torch.no_grad():
                new_log_prob, _, new_dist = agent.get_log_prob_entropy(b_obs, b_act)
                new_ratio = torch.exp(new_log_prob - b_old_log_probs)
                loss_pi = (new_ratio * b_adv).mean()
                kl_val = torch.distributions.kl.kl_divergence(dist, new_dist).mean()
                cost_pi = (new_ratio * b_c_adv).mean()
            return loss_pi, kl_val, cost_pi

        for i in range(args.line_search_iters):
            step_frac = args.line_search_coeff ** i
            new_params = old_params + step_frac * final_step_dir
            set_params(agent.actor, new_params)
            loss, kl, cost_surr = get_loss_and_kl()
            
            # Check KL Constraint
            if kl > args.target_kl * 1.5:
                continue
            
            # Check Improvement & Safety
            if optim_case == 0:
                # Recovery: We only care that cost decreases
                if cost_surr < cost_loss:
                    break
            else:
                cost_change = cost_surr - cost_loss
                if loss > surr_loss and (cost_change + cost_delta <= 1e-8):
                    break
        else:
            # Line search failed: Revert to old params
            set_params(agent.actor, old_params)

        # 5. Value Function Updates
        b_returns = returns.reshape(-1)
        for _ in range(args.vf_iters):
            v_pred = agent.critic(b_obs).flatten()
            v_loss = ((v_pred - b_returns) ** 2).mean()
            optimizer_critic.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
            optimizer_critic.step()
            
        b_c_returns = c_returns.reshape(-1)
        for _ in range(args.vf_iters):
            c_pred = agent.cost_critic(b_obs).flatten()
            c_loss = ((c_pred - b_c_returns) ** 2).mean()
            optimizer_cost_critic.zero_grad()
            c_loss.backward()
            optimizer_cost_critic.step()

        # Logging
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", c_loss.item(), global_step)
        writer.add_scalar("charts/cost_delta", cost_delta, global_step)
        writer.add_scalar("charts/avg_cost", current_cost, global_step)
        writer.add_scalar("charts/optim_case", optim_case, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()