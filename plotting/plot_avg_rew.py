import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Initialize API
api = wandb.Api()

ENTITY = "chacloutier-4B"   
PROJECT = "WOFOST-RL"

# 2. Specify your runs and their CUSTOM NAMES here
# Format: "wandb_path": "Custom Graph Label"
RUNS_TO_PLOT = {
    "chacloutier-4B/WOFOST-RL/awcxfins": "PPO-Lagrangian",  
    "chacloutier-4B/WOFOST-RL/0y70rbij": "Config 1", 
    "chacloutier-4B/WOFOST-RL/5mybr8ij": "Config 2",  
    "chacloutier-4B/WOFOST-RL/5bvrob8i": "Config 3", 
    "chacloutier-4B/WOFOST-RL/egsbe7f6": "Config 4", 
    "chacloutier-4B/WOFOST-RL/kivw5if1": "Config 5", 
}

# run_paths = [
#     "chacloutier-4B/WOFOST-RL/awcxfins", # PPO-Lagrangian
#     "chacloutier-4B/WOFOST-RL/0y70rbij",
#     "chacloutier-4B/WOFOST-RL/5mybr8ij",
#     "chacloutier-4B/WOFOST-RL/5bvrob8i",
#     "chacloutier-4B/WOFOST-RL/egsbe7f6",
#     "chacloutier-4B/WOFOST-RL/kivw5if1",
# ]

# 2. Specify your runs and their CUSTOM NAMES here
# Format: "wandb_path": "Custom Graph Label"
RUNS_TO_PLOT = {
    "0.95": "0q4kco8d",
    "0.85": "hpj157ix",
    "0.80": "gicurq2f",
    "0.75": "gyn1s1w5"
}

# --- METRIC KEYS ---
# Change the constraint_key if you want to plot a specific resource like "constraints/total_n"
reward_key = "charts/average_reward"
constraint_key = "constraints/violation_rate" 

names = []
means_reward, stds_reward = [], []
means_constraint, stds_constraint = [], []

# 3. Fetch and process data for each run
for custom_name, run_id in RUNS_TO_PLOT.items():
    path = f"{ENTITY}/{PROJECT}/{run_id}"
    print(f"Fetching data for: {custom_name} ({path})...")
    run = api.run(path)
    
    # 1. Fetch EVERYTHING without specifying keys to bypass the WandB API bug
    history = run.history(samples=100000)
    
    if history.empty:
        print(f"  -> Warning: API returned empty history for {custom_name}. Skipping.")
        continue

    # 2. Safety Net: Inject missing columns so the plots don't break
    if reward_key not in history.columns:
        print(f"  -> Missing reward for {custom_name}. Defaulting to 0.")
        history[reward_key] = 0.0
        
    if constraint_key not in history.columns:
        print(f"  -> Missing constraints for {custom_name}. Defaulting to 0.")
        history[constraint_key] = 0.0
        
    if "global_step" not in history.columns:
        # Fallback just in case global_step is missing
        history["global_step"] = history["_step"]

    # 3. Clean the data locally! 
    # Forward fill gaps, then drop remaining NaNs
    history_clean = history.sort_values("global_step").ffill().dropna(subset=[reward_key, constraint_key])
    history_clean = history.sort_values("global_step")
    
    if history_clean.empty:
         print(f"  -> Warning: Data was all NaNs for {custom_name} after cleaning. Skipping.")
         continue

    # Find the maximum step this run reached
    max_step = history_clean["global_step"].max()
    
    # Filter to the last 10,000 steps (Note: your code says 100k, variable name says 100k, but math was 100,000. Kept as is!)
    last_100k_data = history_clean[history_clean["global_step"] >= (max_step - 100000)]
    
    # Calculate final metrics
    means_reward.append(last_100k_data[reward_key].mean())
    # stds_reward.append(last_100k_data[reward_key].std())
    stds_reward.append(last_100k_data[reward_key].std() / np.sqrt(len(last_100k_data)))

    test = last_100k_data[constraint_key].sum()
    print("sum = " + str(test))
    
    means_constraint.append(last_100k_data[constraint_key].mean())
    # stds_constraint.append(last_100k_data[constraint_key].std())
    stds_constraint.append(last_100k_data[constraint_key].std() / np.sqrt(len(last_100k_data)))
    
    # Use the custom name for the graph labels instead of run.name
    names.append(custom_name)

# 4. Plotting (Side-by-Side Subplots)
# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

x_positions = np.arange(len(names))

# # --- SUBPLOT 1: REWARD ---
# ax1.bar(
#     x_positions, means_reward, yerr=stds_reward, 
#     align='center', alpha=0.85, ecolor='black', capsize=6, color='#2ca02c' # Green
# )
# # ax1.set_ylim(8.75, 9.75) # Your custom zoom limit
# ax1.set_ylabel('Average Yield Reward (Last 100k Steps)', fontsize=12)
# ax1.set_title('Yield Performance (Last 100k Steps)', fontsize=14)
# ax1.set_xticks(x_positions)
# ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=11)
# ax1.grid(axis='y', linestyle='--', alpha=0.5)
# ax1.set_xlabel("Target Resource Utilization")

ax1.errorbar(
    x_positions, means_reward, yerr=stds_reward, 
    fmt='-o',          # '-' means solid line, 'o' means circle marker
    color='#2ca02c',   # Green
    ecolor='black',    # Black error bars
    capsize=6, 
    markersize=8,      # Make the dots a bit bigger
    linewidth=2        # Thicken the connecting line
)

# You can safely zoom back in now!
ax1.set_ylim(9.1, 9.3) 

ax1.set_ylabel('Average Yield Reward (Last 100k Steps)', fontsize=12)
ax1.set_title('Yield Performance', fontsize=14)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_xlabel("Target Resource Utilization", fontsize=12)
ax1.set_xlim(-0.2, 3.2)

# --- SUBPLOT 2: CONSTRAINTS ---
ax2.bar(
    x_positions, means_constraint, yerr=stds_constraint,
    align='center', alpha=0.85, ecolor='black', capsize=6, color='#d62728' # Red
)
ax2.set_ylabel('Constraint Violation', fontsize=12)
ax2.set_title('Average Resource Constraint Violation (Last 100k Steps)', fontsize=14)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_xlabel("Target Resource Utilization", fontsize=12)

# Formatting to prevent overlap
plt.tight_layout()

plt.show()