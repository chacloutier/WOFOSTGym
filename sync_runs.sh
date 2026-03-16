#!/bin/bash

source $HOME/env/wofost_env/bin/activate

echo "🔍 Searching for W&B offline runs from March 16, 2026..."

# Define the exact target directories using brace expansion 
# (Note: Bash requires NO spaces after the commas inside the braces!)
TARGET_DIRS=~/scratch/runs/{baseline_local,dqn_local,ppo_local,sac_local}/pear/seed_1/wandb/

# Find all directories within those targets that match today's run prefix
find $TARGET_DIRS -type d \( -name "run-20260316*" -o -name "offline-run-20260316*" -o -name "offline-run-20260317*" -o -name "run-20260317*" \) 2>/dev/null | while read -r run_dir; do
    echo "---------------------------------------------------"
    echo "🚀 Syncing: $run_dir"
    wandb sync "$run_dir"
done

echo "---------------------------------------------------"
echo "✅ All matching runs have been synced!"