"""File for training a Deep RL Agent
Agents Supported:
 - PPO
 - RPPO
 - SAC
 - DQN
 - BCQ
 - AIRL
 - GAIL
 - BC

Written by: Will Solow, 2024

To run: python3 train_agent.py --agent-type <Agent Type> --save-folder <logs>
"""

import tyro
from dataclasses import dataclass, field

import utils
from typing import Union, Annotated, Optional
from argparse import Namespace
from rl_algs.PPO import Args as PPOArgs
from rl_algs.CPO import Args as CPOArgs
from rl_algs.DQN import Args as DQNArgs
from rl_algs.SAC import Args as SACArgs
from rl_algs.TRPO import Args as TRPOArgs
from rl_algs.PPO_Lag import Args as PPOLagArgs
from rl_algs.PPO_Lag_rate import Args as PPOLagRateArgs
from rl_algs.TRPO_Lag import Args as TRPOLagArgs
from rl_algs.BASELINE import Args as BASELINEArgs

@dataclass
class AgentArgs(utils.Args):
    """Agent Args configurations"""
    alg: Union[
        Annotated[PPOArgs, tyro.conf.subcommand(name="PPO")],
        Annotated[CPOArgs, tyro.conf.subcommand(name="CPO")],
        Annotated[DQNArgs, tyro.conf.subcommand(name="DQN")],
        Annotated[SACArgs, tyro.conf.subcommand(name="SAC")],
        Annotated[TRPOArgs, tyro.conf.subcommand(name="TRPO")],
        Annotated[PPOLagArgs, tyro.conf.subcommand(name="PPO_Lag")],
        Annotated[PPOLagRateArgs, tyro.conf.subcommand(name="PPO_Lag_rate")],
        Annotated[TRPOLagArgs, tyro.conf.subcommand(name="TRPO_Lag")],
        Annotated[BASELINEArgs, tyro.conf.subcommand(name="BASELINE")],
    ] = field(default_factory=lambda: PPOArgs())


    """Agent Type: RPPO | PPO | DQN | SAC| BCQ | etc"""
    agent_type: Optional[str] = None

    """Tracking Flag, if True will Track using Weights and Biases"""
    track: bool = False

    """Render mode, default to None for no rendering"""
    render_mode: Optional[str] = None

if __name__ == "__main__":
    tyro_args = tyro.cli(AgentArgs)

    if isinstance(tyro_args.alg, PPOArgs):
        agent_key = "PPO"
    elif isinstance(tyro_args.alg, CPOArgs):
        agent_key = "CPO"
    elif isinstance(tyro_args.alg, DQNArgs):
        agent_key = "DQN"
    elif isinstance(tyro_args.alg, SACArgs):
        agent_key = "SAC"
    elif isinstance(tyro_args.alg, TRPOArgs):
        agent_key = "TRPO"
    elif isinstance(tyro_args.alg, PPOLagArgs):
        agent_key = "PPO_Lag"
    elif isinstance(tyro_args.alg, PPOLagRateArgs):
        agent_key = "PPO_Lag_rate"
    elif isinstance(tyro_args.alg, TRPOLagArgs):
        agent_key = "TRPO_Lag"
    elif isinstance(tyro_args.alg, BASELINEArgs):
        agent_key = "BASELINE"
    else:
        raise ValueError(f"Unknown agent configuration selected: {type(tyro_args.alg)}")

    try:
        trainers, _ = utils.get_valid_trainers()
        ag_trainer = trainers[agent_key]
    except KeyError:
        msg = f"Error: Trainer for {agent_key} not found in rl_algs/. Check utils.get_valid_trainers()."
        raise Exception(msg)
    
    tyro_args.agent_type = agent_key

    # 5. Run training
    ag_trainer(tyro_args)