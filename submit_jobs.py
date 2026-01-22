import os
import subprocess
import time

# --- Configuration ---
seeds = [1, 2, 3, 4]  # The 4 seeds you want to run
project_dir = os.getcwd() # Assumes you run this script from the root of your workspace
# pointing to scratch is critical on Compute Canada to avoid quota issues
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
            "--max_n", "80.0",
            "--max_p", "inf",
            "--max_k", "inf",
            "--max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
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
            "--max_n", "80.0",
            "--max_p", "1000.0",
            "--max_k", "1000.0",
            "--max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
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
            "--max_n", "80.0",
            "--max_p", "1000.0",
            "--max_k", "1000.0",
            "--max_w", "40.0",
            "--npk.ag.crop-name", "pear",
            "--npk.ag.crop-variety", "pear_1",
            "--track",
            "alg:DQN",
            "--alg.checkpoint-frequency", "5000",
        ]
    }
]

# SLURM Template for Compute Canada
# Adjust --account, --time, and --mem as needed
slurm_template = """#!/bin/bash
#SBATCH --account=def-mcrowley_gpu
#SBATCH --time=3:00:00        
#SBATCH --mem=16G             
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x-%j.out

# Activate Virtual Environment
source $HOME/env_wofost/bin/activate 

# Offline W&B Configuration
export WANDB_MODE=offline
export WANDB_DIR={save_folder} 

# 4. Run Command
echo "Starting job on $(hostname)"
{command}
"""

os.makedirs("generated_scripts", exist_ok=True)

# --- Submission Loop ---
for seed in seeds:
    for job in jobs:
        job_name = f"{job['name']}_s{seed}"
        
        # Construct the save folder path on scratch
        # Structure: scratch/runs/ppo_local/pear/seed_X
        save_folder = os.path.join(
            base_save_folder, 
            f"{job['agent_type'].lower()}_local", 
            "pear", 
            f"seed_{seed}"
        )
        
        # Ensure the directory exists so W&B can write to it immediately
        os.makedirs(save_folder, exist_ok=True)

        # Build the python command
        # We inject --seed and --save-folder dynamically
        cmd_args = ["python", "train_agent.py"] + job['args']
        cmd_args.extend(["--seed", str(seed)])
        cmd_args.extend(["--save-folder", save_folder])
        
        full_command = " ".join(cmd_args)

        # Create the SLURM script content
        script_content = slurm_template.format(
            job_name=job_name,
            log_dir=save_folder,
            save_folder=save_folder,
            command=full_command
        )

        # Write to file
        script_filename = f"generated_scripts/submit_{job_name}.sh"
        with open(script_filename, "w") as f:
            f.write(script_content)

        # Submit
        print(f"Submitting {job_name} (Seed {seed})...")
        subprocess.run(["sbatch", script_filename]) 
        # Uncomment the line above to actually submit. 
        # Leaving it commented so you can inspect generated scripts first.