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
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 650
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    checkpoint_frequency: int = 500
    """How often to save the agent during training"""

    # --- PPO-Lagrangian Specific Args ---
    cost_limit: float = 25.0
    """The maximum allowed expected cost per episode"""
    lagrangian_learning_rate: float = 5e-2
    """Learning rate for the Lagrange multiplier"""
    initial_lambda: float = 1.0
    """Initial value for the Lagrange multiplier"""

    # --- Constraint Thresholds (From CPO) ---
    max_n: float = 80.0
    """Maximum Nitrogen limit"""
    max_p: float = 80.0
    """Maximum Phosphorous limit"""
    max_k: float = 80.0
    """Maximum Potassium limit"""
    max_w: float = 40.0
    """Maximum Water limit"""
    terminate_on_violation: bool = False
    """Toggle whether the episode terminates immediately upon violating any constraint limit"""

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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
            
            if n > args.max_n:
                costs[i] += (n - args.max_n) / args.max_n
            if k > args.max_k:
                costs[i] += (k - args.max_k) / args.max_k
            if p > args.max_p:
                costs[i] += (p - args.max_p) / args.max_p
            if w > args.max_w:
                costs[i] += (w - args.max_w) / args.max_w
                
    return costs


class PPOLag(nn.Module, Agent):
    def __init__(self, envs: gym.Env, args: Args, state_fpath: str = None) -> None:
        super().__init__()
        self.env = envs
        self.args = args

        # 1. Reward Critic (Standard V_R)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # 2. Cost Critic (V_C) - Estimates expected cost
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # 3. Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        # 4. Lagrange Multiplier (Learnable Parameter)
        self.log_lagrange_multiplier = nn.Parameter(
            torch.log(torch.tensor(max(1e-5, args.initial_lambda))), 
            requires_grad=True
        )

        if state_fpath is not None:
            assert isinstance(
                state_fpath, str
            ), f"`state_fpath` must be of type `str` but is of type `{type(state_fpath)}`"
            try:
                self.load_state_dict(torch.load(state_fpath, weights_only=True))
            except:
                msg = f"Error loading state dictionary from {state_fpath}"
                raise Exception(msg)

    def get_lagrange_multiplier(self):
        """Returns the current value of lambda (strictly positive)"""
        return torch.exp(self.log_lagrange_multiplier)

    def get_action(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Helper for inference"""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Reward Value, Cost Value)"""
        return self.critic(x), self.cost_critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # Returns: action, log_prob, entropy, reward_val, cost_val
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), self.cost_critic(x)


def train(kwargs: Namespace) -> None:
    """
    PPO-Lagrangian Training Function
    """

    args = kwargs.alg
    run_name = f"PPOLag/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)

    agent = PPOLag(envs, args).to(device)
    
    # Optimizer for Policy and Both Value Networks
    optimizer = optim.Adam(
        list(agent.actor.parameters()) + list(agent.critic.parameters()) + list(agent.cost_critic.parameters()),
        lr=args.learning_rate, 
        eps=1e-5
    )

    # Separate Optimizer for the Lagrange Multiplier
    lagrange_optimizer = optim.Adam([agent.log_lagrange_multiplier], lr=args.lagrangian_learning_rate)

    # Storage buffers
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device) # Store binary costs
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs)).to(device) 

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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
                # Get action and BOTH values (reward and cost)
                action, logprob, _, value, cost_value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                cost_values[step] = cost_value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            # --- EARLY TERMINATION ON VIOLATION ---
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
                    if violated[i] and not terminations[i] and not truncations[i]:
                        terminations[i] = True
                        if args.num_envs == 1:
                            next_obs, _ = envs.reset(seed=args.seed)
                        else:
                            print("WARNING: Early termination triggered, but manual reset for num_envs > 1 requires a custom wrapper.")

            next_done = np.logical_or(terminations, truncations)
            
            # --- UPDATED: Specific Cost Calculation ---
            step_cost = extract_step_cost(infos, args, args.num_envs)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            costs[step] = torch.tensor(step_cost).to(device).view(-1)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            
            # --- Logging specific constraints AND Average Reward ---
            if global_step % args.checkpoint_frequency == 0:
                # 1. Log Average Reward (using eval_policy)
                writer.add_scalar("charts/average_reward", eval_policy(agent, envs, kwargs, device), global_step)

                # 2. Log Constraint Details
                if "track/total_n" in infos:
                    writer.add_scalar("constraints/total_n", infos["track/total_n"][0], global_step)
                    writer.add_scalar("constraints/total_p", infos["track/total_p"][0], global_step)
                    writer.add_scalar("constraints/total_k", infos["track/total_k"][0], global_step)
                    writer.add_scalar("constraints/total_w", infos["track/total_w"][0], global_step)
                    writer.add_scalar("constraints/violation_rate", infos["track/is_violating"], global_step)

        # --- GAE Calculation (Double GAE) ---
        with torch.no_grad():
            next_value, next_cost_value = agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1)
            next_cost_value = next_cost_value.reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            cost_advantages = torch.zeros_like(costs).to(device)
            
            lastgaelam = 0
            lastgaelam_cost = 0
            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    nextcostvalues = next_cost_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    nextcostvalues = cost_values[t + 1]
                
                # 1. Reward GAE
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                
                # 2. Cost GAE
                delta_cost = costs[t] + args.gamma * nextcostvalues * nextnonterminal - cost_values[t]
                cost_advantages[t] = lastgaelam_cost = delta_cost + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_cost

            returns = advantages + values
            cost_returns = cost_advantages + cost_values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_cost_advantages = cost_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_cost_returns = cost_returns.reshape(-1)
        b_values = values.reshape(-1)
        b_cost_values = cost_values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        epoch_entropy = []
        
        # --- Update Phase ---
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newcostvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Store entropy for accurate logging
                epoch_entropy.append(entropy.mean().item())

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # --- Lagrangian Advantage ---
                cur_lambda = agent.get_lagrange_multiplier().item()
                
                mb_advantages = b_advantages[mb_inds]
                mb_cost_advantages = b_cost_advantages[mb_inds]

                # 1. COMBINE FIRST to preserve the true mathematical ratio
                combined_advantages = mb_advantages - cur_lambda * mb_cost_advantages
                
                # 2. THEN NORMALIZE the combined signal
                if args.norm_adv:
                    combined_advantages = (combined_advantages - combined_advantages.mean()) / (combined_advantages.std() + 1e-8)

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

                # Cost Value Loss
                newcostvalue = newcostvalue.view(-1)
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
                
                # Total Loss
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef + cv_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # --- Lagrange Multiplier Update ---
        mean_cost = b_cost_returns.mean()
        violation = mean_cost - args.cost_limit
        
        lagrange_optimizer.zero_grad()
        lambda_loss = -agent.get_lagrange_multiplier() * violation.detach()
        lambda_loss.backward()
        lagrange_optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/lambda", agent.get_lagrange_multiplier().item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", cv_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", np.mean(epoch_entropy), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("debug/adv_mean", b_advantages.mean(), global_step)
        writer.add_scalar("debug/adv_std", b_advantages.std(), global_step)
        writer.add_scalar("debug/pg_loss", pg_loss.item(), global_step)
        writer.add_scalar("debug/v_loss", v_loss.item(), global_step)
        
    envs.close()
    writer.close()