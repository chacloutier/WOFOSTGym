#!/bin/bash

source $HOME/env/wofost_env/bin/activate

echo "🔍 Searching for W&B offline runs from March 16 and 17, 2026..."

# Put the paths directly into the find command so Bash expands them correctly!
# (Note: Added /wandb/ to the end of the path based on your ls output)
find /scratch/cloutcha/runs/{baseline_local,dqn_local,ppo_local,sac_local,cpo_local}/pear/seed_1/wandb/ -type d \( -name "run-20260316*" -o -name "offline-run-20260316*" -o -name "run-20260317*" -o -name "offline-run-20260317*" \) 2>/dev/null | while read -r run_dir; do
    echo "---------------------------------------------------"
    echo "🚀 Syncing: $run_dir"
    wandb sync "$run_dir"
done

echo "---------------------------------------------------"
echo "✅ All matching runs have been synced!"