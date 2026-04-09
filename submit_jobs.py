import os
import subprocess
import time

# --- Configuration ---
# seeds = [1, 2, 3, 4]  # The 4 seeds you want to run
seeds = [1]
project_dir = os.getcwd() 
base_save_folder = os.path.join(os.getenv('SCRATCH', '.'), 'runs') 

# Define the 3 configurations based on your VS Code launch.json
jobs = [
    {
        "name": "PPO_Pear",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "SimpleRewardMachineWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:PPO",
            "--alg.num-steps", "2048",
        ]
    },
    {
        "name": "SAC_Pear",
        "agent_type": "SAC",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "SimpleRewardMachineWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:SAC",
        ]
    },
    {
        "name": "DQN_Pear",
        "agent_type": "DQN",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "SimpleRewardMachineWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:DQN",
        ]
    },
    {
        "name": "BASELINE_Pear",
        "agent_type": "BASELINE",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env-reward", "SimpleRewardMachineWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:BASELINE",
        ],
    },
]

jobs = [
    {
        "name": "CPO_Pear",
        "agent_type": "CPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "40.0",
            "--env-reward", "RewardScalingWrapper",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:CPO",
            "--alg.target-kl", "0.01"
        ]
    },
    {
        "name": "PPO_Lag_Pear",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.05",
            "--alg.cost-limit", "0.1",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0"
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0"
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Rate_Pear_no_norm",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0"
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0"
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Pear",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.1",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0"
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Rate_Pear",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.001",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Rate_Pear",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Rate_Pear",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Rate_Pear",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardScalingWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Pear_Smooth_Constraint",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseSmoothConstraintRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Pear_Smooth_Constraint",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseSmoothConstraintRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    }
]

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.fixed-lambda",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.fixed-lambda",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.fixed-lambda",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.fixed-lambda",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    }
]

jobs = [
    {
        "name": "PPO_Pear_Smooth_Constraint",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseSmoothConstraintRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Pear_Smooth_Constraint",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseSmoothConstraintRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.fixed-lambda",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.fixed-lambda",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Pear_Smooth_Constraint",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseSmoothConstraintRewardWrapper",
            "--max_n", "80.0",
            "--max_p", "80.0",
            "--max_k", "80.0",
            "--max_w", "40.0",
            "--npk.max_n", "80.0",
            "--npk.max_p", "80.0",
            "--npk.max_k", "80.0",
            "--npk.max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
         {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0",
            "--alg.ent-coef", "0.05",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0",
            "--alg.ent-coef", "0.05",
        ]
    },
    {
        "name": "PPO_Pear_Smooth_Constraint",
        "agent_type": "PPO",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseSmoothConstraintRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
        {
        "name": "PPO_Lag_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0",
            "--alg.ent-coef", "0.05",
            "--alg.learning-rate", "0.001",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0",
            "--alg.ent-coef", "0.05",
            "--alg.learning-rate", "0.001",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward_Fixed_Lambda",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.fixed-lambda",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0",
            "--alg.ent-coef", "0.05",
        ]
    },
    {
        "name": "PPO_Lag_Rate_Pear_Dense_Reward_Fixed_Lambda",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max-n", "40.0",
            "--max-p", "40.0",
            "--max-k", "40.0",
            "--max-w", "20.0",
            "--npk.max-n", "40.0",
            "--npk.max-p", "40.0",
            "--npk.max-k", "40.0",
            "--npk.max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.fixed-lambda",
            "--alg.max-n", "40.0",
            "--alg.max-p", "40.0",
            "--alg.max-k", "40.0",
            "--alg.max-w", "20.0",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "1.0",
            "--alg.ent-coef", "0.05",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.2",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.3",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.03",
        ]
    },
    {
        "name": "PPO_Lag_Pear_Dense_Reward_higher_ent_coef",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.05",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.2",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.3",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.2",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.3",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "BASELINE_Pear_Dense_Reward",
        "agent_type": "BASELINE",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            
            # Global Env Limits
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            
            # Agro Settings
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            
            # Logging
            "--track",
            "--track-resources", "True",
            
            # Baseline Specific Args
            "alg:BASELINE",
            "--alg.max_n", "40.0",
            "--alg.max_p", "40.0",
            "--alg.max_k", "40.0",
            "--alg.max_w", "20.0"
        ]
    }
]

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.01",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Pear",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.05",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.05",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_rate_Pear_Dense_Reward",
        "agent_type": "PPO_Lag_rate",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag_rate",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.lagrangian-learning-rate", "0.03",
            "--alg.pid-kp", "0.3",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]


# 1. Define the agents and any agent-specific hyperparameters
agents = [
    ("PPO", ["--alg.no-norm-adv", "--alg.num-steps", "4096", "--alg.ent-coef", "0.01"]),
    ("PPO", ["--alg.num-steps", "4096", "--alg.ent-coef", "0.01"]),
    ("SAC", []),
    ("DQN", []),
    ("BASELINE", [])
]

# 2. Define the four upgraded Dense wrappers
wrappers = [
    "DenseSmoothConstraintRewardWrapper",
    "DenseRewardMachineWrapper",         
    "DenseFertilizationThresholdWrapper", 
    "DenseThresholdRespectingWrapper", 
    "RewardFertilizationThresholdWrapper",
]

agents = [
    ("CPO", ["--alg.num-steps", "4096", "--alg.target-kl", "0.01", "--alg.cost-limit", "0.05"]),
]

agents = [
    ("SAC", []),
    ("DQN", []),
]

# 2. Define the four upgraded Dense wrappers
wrappers = [
    "DenseLagrangianRewardWrapper",
]

agents = [
    ("PPO", ["--alg.no-norm-adv", "--alg.num-steps", "4096", "--alg.ent-coef", "0.01"]),
    ("PPO", ["--alg.num-steps", "4096", "--alg.ent-coef", "0.01"]),
    ("SAC", []),
    ("DQN", []),
    ("BASELINE", [])
]

# 2. Define the four upgraded Dense wrappers
wrappers = [
    "DenseSmoothConstraintRewardWrapper",
    "DenseRewardMachineWrapper",         
    "DenseFertilizationThresholdWrapper", 
    "DenseThresholdRespectingWrapper", 
]

agents = [
    ("SAC", []),
    ("BASELINE", [])
]

# 2. Define the four upgraded Dense wrappers
wrappers = [
    "RewardFertilizationThresholdWrapper",
]

agents = [
    ("CPO", ["--alg.num-steps", "4096", "--alg.target-kl", "0.05", "--alg.cost-limit", "0.1"]),
    ("CPO", ["--alg.num-steps", "4096", "--alg.target-kl", "0.01", "--alg.cost-limit", "0.1"]),
]

# 2. Define the four upgraded Dense wrappers
wrappers = [
    "DenseLagrangianRewardWrapper",
]

jobs = []

# 3. Automatically generate the 25 job configurations
for agent_name, agent_args in agents:
    for wrapper in wrappers:
        
        # Create a shorter, cleaner name for the WandB logs (e.g., "PPO_SmoothConstraint")
        short_wrapper = wrapper.replace("Dense", "").replace("RewardWrapper", "").replace("Wrapper", "")
        job_name = f"{agent_name}_{short_wrapper}"
        
        job = {
            "name": job_name,
            "agent_type": agent_name,
            "args": [
                "--env-id", "perennial-lnpkw-v0",
                "--agro-file", "pear_agro.yaml",
                "--env_reward", wrapper,
                
                # --- Global Environment Limits ---
                "--max-n", "40.0",
                "--max-p", "40.0",
                "--max-k", "40.0",
                "--max-w", "20.0",
                "--npk.max-n", "40.0",
                "--npk.max-p", "40.0",
                "--npk.max-k", "40.0",
                "--npk.max-w", "20.0",
                
                # --- Agro Settings ---
                "--npk.ag.crop-name", "pear",
                "--npk.ag.crop-variety", "pear_1",
                "--npk.intvn_interval", "14",
                
                # --- Logging ---
                "--track",
                "--track-resources", "True",
                
                # --- Algorithm-Specific Section ---
                f"alg:{agent_name}",
                "--alg.max-n", "40.0",
                "--alg.max-p", "40.0",
                "--alg.max-k", "40.0",
                "--alg.max-w", "20.0",
            ] + agent_args  # Appends the specific args for PPO, SAC, etc.
        }
        jobs.append(job)

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.01",
        ]
    },
]

jobs = [
    {
        "name": "PPO_Lag_Pear_Dense_Reward_higher_lr",
        "agent_type": "PPO_Lag",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "DenseLagrangianRewardWrapper",
            "--max_n", "40.0",
            "--max_p", "40.0",
            "--max_k", "40.0",
            "--max_w", "20.0",
            "--npk.max_n", "40.0",
            "--npk.max_p", "40.0",
            "--npk.max_k", "40.0",
            "--npk.max_w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "--track-resources", "True",
            "alg:PPO_Lag",
            "--alg.no-norm-adv",
            "--alg.num-steps", "4096",
            "--alg.cost-limit", "0.05",
            "--alg.lagrangian-learning-rate", "0.1",
            "--alg.initial-lambda", "0.05",
            "--alg.ent-coef", "0.05",
        ]
    },
]

# SLURM Template for Compute Canada
slurm_template = """#!/bin/bash
#SBATCH --account=def-mcrowley_gpu
#SBATCH --time=0-23:00:00  
#SBATCH --mem=32G             
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x-%j.out

# Activate Virtual Environment
source $HOME/env/wofost_env/bin/activate 

# Offline W&B Configuration
export WANDB_MODE=online
export WANDB_DIR={save_folder} 
export WANDB_PROJECT="WOFOST-RL"
export WANDB_ENTITY="chacloutier-4B"
if [ -f $HOME/.wandb_key ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
fi

# wandb login --relogin

# 4. Run Command
echo "Starting job on $(hostname)"
{command}
"""

os.makedirs("generated_scripts", exist_ok=True)

# --- Submission Loop ---
for seed in seeds:
    for job in jobs:
        job_name = f"{job['name']}_s{seed}"
        
        save_folder = os.path.join(
            base_save_folder, 
            f"{job['agent_type'].lower()}_local", 
            "pear", 
            f"seed_{seed}/"
        )
        os.makedirs(save_folder, exist_ok=True)

        # --- FIX: Split args based on where "alg:XXX" is located ---
        raw_args = job['args']
        algo_token = f"alg:{job['agent_type']}"
        
        try:
            split_index = raw_args.index(algo_token)
            # Args before "alg:XXX" (Global args)
            pre_args = raw_args[:split_index]
            # Args starting from "alg:XXX" (Subcommand args)
            post_args = raw_args[split_index:]
        except ValueError:
            print(f"Error: Could not find '{algo_token}' in args for {job_name}")
            continue

        # Construct new command list
        cmd_parts = ["python", "train_agent.py"]
        
        # 1. Add Global Args + Save Folder
        cmd_parts.extend(pre_args)
        cmd_parts.extend(["--save-folder", save_folder])
        
        # 2. Add Algo Subcommand + Algo Args + Seed
        cmd_parts.extend(post_args)
        # Change --seed to --alg.seed per error message
        cmd_parts.extend(["--alg.seed", str(seed)]) 
        
        full_command = " ".join(cmd_parts)

        script_content = slurm_template.format(
            job_name=job_name,
            log_dir=save_folder,
            save_folder=save_folder,
            command=full_command
        )

        script_filename = f"generated_scripts/submit_{job_name}.sh"
        with open(script_filename, "w") as f:
            f.write(script_content)

        print(f"Submitting {job_name} (Seed {seed})...")
        subprocess.run(["sbatch", script_filename])