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
    """Learning rate for the critic optimizer"""
    num_envs: int = 1
    num_steps: int = 2048
    """TRPO usually requires larger batch sizes for stable curvature estimates"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    """Used for the critic update"""
    update_epochs: int = 10
    """Used for the critic update"""
    norm_adv: bool = True
    max_grad_norm: float = 0.5
    checkpoint_frequency: int = 500
    
    # TRPO Specific Hyperparameters
    max_kl: float = 0.01
    """The maximum KL divergence constraint (delta)"""
    cg_iters: int = 10
    """Number of iterations for Conjugate Gradient"""
    line_search_backtrack_ratio: float = 0.8
    line_search_max_steps: int = 10
    damping: float = 0.1
    """Damping factor for the Fisher Information Matrix (Tikhonov regularization)"""

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def layer_init(layer: nn.Module, std: np.ndarray = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class TRPO_Agent(nn.Module, Agent):
    def __init__(self, envs: gym.Env):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

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
    run_name = f"TRPO/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)
    agent = TRPO_Agent(envs).to(device)
    
    # Only the critic uses Adam
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.learning_rate)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # 1. Collect Rollouts
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        # 2. GAE Calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_logprobs = logprobs.reshape(-1)

        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 3. TRPO Actor Update
        
        # FIX: Pre-compute old logits/probs specifically for the line search and KL calculation
        with torch.no_grad():
            old_logits = agent.actor(b_obs)
            old_probs = Categorical(logits=old_logits)

        # Surrogate loss: E[ (pi / pi_old) * Adv ]
        def get_loss():
            new_logits = agent.actor(b_obs)
            new_probs = Categorical(logits=new_logits)
            ratio = torch.exp(new_probs.log_prob(b_actions) - b_logprobs)
            return (ratio * b_advantages).mean()

        # Fisher Information Matrix Vector Product
        def fvp(v):
            new_logits = agent.actor(b_obs)
            new_probs = Categorical(logits=new_logits)
            
            # KL divergence for curvature (approx Hessian)
            # We differentiate KL(old || new)
            kl = torch.distributions.kl.kl_divergence(old_probs, new_probs).mean()
            
            grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            
            kl_v = (flat_grad_kl * v).sum()
            grads_v = torch.autograd.grad(kl_v, agent.actor.parameters())
            flat_grad_grad_kl = torch.cat([g.view(-1) for g in grads_v]).detach()
            
            return flat_grad_grad_kl + v * args.damping

        # Search direction
        loss = get_loss()
        grads = torch.autograd.grad(loss, agent.actor.parameters())
        loss_grad = torch.cat([g.view(-1) for g in grads]).detach()
        
        step_dir = conjugate_gradient(fvp, loss_grad, iters=args.cg_iters)
        
        # Max step size
        shs = 0.5 * (step_dir * fvp(step_dir)).sum(0, keepdim=True)
        
        # FIX: Safety check for negative curvature (common in neural net approximations)
        if shs > 0:
            lm = torch.sqrt(shs / args.max_kl)
            full_step = step_dir / lm

            # Backtracking Line Search
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
                    
                # Improvement Check:
                # 1. new_loss > loss (Gradient Ascent)
                # 2. KL constraint satisfied
                if kl <= args.max_kl and new_loss > loss:
                    # Accepted
                    break
                
                if i == args.line_search_max_steps - 1:
                    # Failed to find a step, revert params
                    set_flat_params(agent.actor, old_params)
        else:
            # If shs <= 0, the Hessian is not positive definite; skip update
            pass

        # 4. Critic Update
        for epoch in range(args.update_epochs):
            inds = np.arange(args.batch_size)
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = inds[start:end]
                
                v_loss = 0.5 * ((agent.get_value(b_obs[mb_inds]).view(-1) - b_returns[mb_inds]) ** 2).mean()
                optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()