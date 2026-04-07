# import wandb
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # 1. Initialize API
# api = wandb.Api()

# # 2. Specify your list of runs
# run_paths = [
#     "chacloutier-4B/WOFOST-RL/awcxfins",  # PPO-Lagrangian
#     "chacloutier-4B/WOFOST-RL/0y70rbij", 
#     "chacloutier-4B/WOFOST-RL/5mybr8ij",  
#     "chacloutier-4B/WOFOST-RL/5bvrob8i", 
#     "chacloutier-4B/WOFOST-RL/egsbe7f6", 
#     "chacloutier-4B/WOFOST-RL/kivw5if1", 
# ]

# names = []
# means = []
# stds = []

# # 3. Fetch and process data for each run
# for path in run_paths:
#     print(f"Fetching data for: {path}...")
#     run = api.run(path)
    
#     # We explicitly request only the two columns we need. 
#     # Setting samples=100000 prevents WandB from downsampling your data.
#     history = run.history(samples=100000000, keys=["global_step", "charts/average_reward"])
    
#     # Clean the data
#     history = history.dropna(subset=["charts/average_reward", "global_step"])
    
#     if history.empty:
#         print(f"  -> Warning: No reward data found for {run.name}. Skipping.")
#         continue

#     # Find the maximum step this run reached (e.g., 1,000,000)
#     max_step = history["global_step"].max()
    
#     # Filter the dataframe to ONLY include the last 10,000 steps
#     last_10k_data = history[history["global_step"] >= (max_step - 10000)]
    
#     # Calculate our final metrics
#     avg_reward = last_10k_data["charts/average_reward"].mean()
#     std_reward = last_10k_data["charts/average_reward"].std()
    
#     names.append(run.name)
#     means.append(avg_reward)
#     stds.append(std_reward)

# # 4. Plot the Bar Chart
# plt.figure(figsize=(10, 6))

# plt.ylim(8, 10)

# x_positions = np.arange(len(names))

# # Create bars. 'yerr' automatically draws the standard deviation lines.
# # 'capsize' gives the error bars those nice horizontal caps.
# bars = plt.bar(
#     x_positions, 
#     means, 
#     yerr=stds, 
#     align='center', 
#     alpha=0.85, 
#     ecolor='black', 
#     capsize=8,
#     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Add more colors if you have >4 runs
# )

# # Formatting
# plt.ylabel('Average Yield Reward', fontsize=12)
# plt.title('Final Performance Comparison (Last 10k Steps: Mean ± Std)', fontsize=14)

# # Set the x-axis labels to the run names and tilt them so they don't overlap
# plt.xticks(x_positions, names, rotation=25, ha='right', fontsize=11)

# # Add a subtle grid behind the bars for easier reading
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # This ensures the tilted labels don't get cut off when saving
# plt.tight_layout()

# plt.show()
# # plt.savefig("final_performance_bar_chart.png", dpi=300)

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Initialize API
api = wandb.Api()

# 2. Specify your list of runs
run_paths = [
    "chacloutier-4B/WOFOST-RL/awcxfins",  # PPO-Lagrangian
    "chacloutier-4B/WOFOST-RL/0y70rbij", 
    "chacloutier-4B/WOFOST-RL/5mybr8ij",  
    "chacloutier-4B/WOFOST-RL/5bvrob8i", 
    "chacloutier-4B/WOFOST-RL/egsbe7f6", 
    "chacloutier-4B/WOFOST-RL/kivw5if1", 
]

# --- METRIC KEYS ---
# Change the constraint_key if you want to plot a specific resource like "constraints/total_n"
reward_key = "charts/average_reward"
constraint_key = "constraints/violation_rate" 

names = []
means_reward, stds_reward = [], []
means_constraint, stds_constraint = [], []

# 3. Fetch and process data for each run
for path in run_paths:
    print(f"Fetching data for: {path}...")
    run = api.run(path)
    
    # 1. Fetch EVERYTHING without specifying keys to bypass the WandB API bug
    history = run.history(samples=100000)
    
    if history.empty:
        print(f"  -> Warning: API returned empty history for {run.name}. Skipping.")
        continue

    # 2. Safety Net: Inject missing columns so the plots don't break
    if reward_key not in history.columns:
        print(f"  -> Missing reward for {run.name}. Defaulting to 0.")
        history[reward_key] = 0.0
        
    if constraint_key not in history.columns:
        print(f"  -> Missing constraints for {run.name}. Defaulting to 0.")
        history[constraint_key] = 0.0
        
    if "global_step" not in history.columns:
        # Fallback just in case global_step is missing
        history["global_step"] = history["_step"]

    # 3. Clean the data locally! 
    # Forward fill gaps, then drop remaining NaNs
    history_clean = history.sort_values("global_step").ffill().dropna(subset=[reward_key, constraint_key])
    history_clean = history.sort_values("global_step")
    
    if history_clean.empty:
         print(f"  -> Warning: Data was all NaNs for {run.name} after cleaning. Skipping.")
         continue

    # Find the maximum step this run reached
    max_step = history_clean["global_step"].max()
    
    # Filter to the last 10,000 steps
    last_100k_data = history_clean[history_clean["global_step"] >= (max_step - 100000)]
    
    # Calculate final metrics
    means_reward.append(last_100k_data[reward_key].mean())
    stds_reward.append(last_100k_data[reward_key].std())

    test = last_100k_data[constraint_key].sum()
    print("sum = " + str(test))
    
    means_constraint.append(last_100k_data[constraint_key].mean())
    stds_constraint.append(last_100k_data[constraint_key].std())
    
    names.append(run.name)

# 4. Plotting (Side-by-Side Subplots)
# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x_positions = np.arange(len(names))

# --- SUBPLOT 1: REWARD ---
ax1.bar(
    x_positions, means_reward, yerr=stds_reward, 
    align='center', alpha=0.85, ecolor='black', capsize=6, color='#2ca02c' # Green
)
ax1.set_ylim(8.75, 9.75) # Your custom zoom limit
ax1.set_ylabel('Average Yield Reward (Last 100k Steps)', fontsize=12)
ax1.set_title('Yield Performance (Last 100k Steps)', fontsize=14)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- SUBPLOT 2: CONSTRAINTS ---
# ax2.bar(
#     x_positions, means_constraint, yerr=stds_constraint, 
#     align='center', alpha=0.85, ecolor='black', capsize=6, color='#d62728' # Red
# )
ax2.bar(
    x_positions, means_constraint, 
    align='center', alpha=0.85, ecolor='black', capsize=6, color='#d62728' # Red
)
ax2.set_ylabel('Average Constraint Violation', fontsize=12)
ax2.set_title('Safety / Constraint Compliance', fontsize=14)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Optional: Add a dashed line on the constraint graph to show your budget limit (e.g., 0.05)
# ax2.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label="Budget Limit")
# ax2.legend()

# Formatting to prevent overlap
plt.tight_layout()

plt.show()
# plt.savefig("final_performance_comparison.png", dpi=300)