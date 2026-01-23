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
            "--env_reward", "RewardFertilizationThresholdWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:PPO",
            "--alg.num-steps", "2048",
            "--alg.checkpoint-frequency", "5000",
        ]
    },
    {
        "name": "SAC_Pear",
        "agent_type": "SAC",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardFertilizationThresholdWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:SAC",
            "--alg.checkpoint-frequency", "5000",
        ]
    },
    {
        "name": "DQN_Pear",
        "agent_type": "DQN",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env_reward", "RewardFertilizationThresholdWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:DQN",
            "--alg.checkpoint-frequency", "5000",
        ]
    },
    {
        "name": "BASELINE_Pear",
        "agent_type": "BASELINE",
        "args": [
            "--env-id", "perennial-lnpkw-v0",
            "--agro-file", "pear_agro.yaml",
            "--env-reward", "RewardFertilizationThresholdWrapper",
            "--max-n", "80.0",
            "--max-p", "80.0",
            "--max-k", "80.0",
            "--max-w", "20.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--npk.intvn_interval", "14",
            "--track",
            "alg:BASELINE",
            "--alg.checkpoint-frequency", "5000",
        ],
    },
]

# SLURM Template for Compute Canada
# Adjust --account, --time, and --mem as needed
slurm_template = """#!/bin/bash
#SBATCH --account=def-mcrowley_gpu
#SBATCH --time=0-5:00:00        
#SBATCH --mem=8G             
#SBATCH --cpus-per-task=2
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

wandb login --relogin

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