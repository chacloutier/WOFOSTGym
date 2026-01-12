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
# from rl_algs.dqn import Args as DQNArgs  # Add others as you implement them

@dataclass
class AgentArgs(utils.Args):
    """Agent Args configurations"""
    alg: Union[
        Annotated[PPOArgs, tyro.conf.subcommand(name="PPO")],
        Annotated[CPOArgs, tyro.conf.subcommand(name="CPO")],
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