import wandb
import matplotlib.pyplot as plt
import pandas as pd

# 1. Initialize the API
api = wandb.Api()

# 2. Specify your list of run paths
run_paths = [
    "chacloutier-4B/WOFOST-RL/awcxfins",
    "chacloutier-4B/WOFOST-RL/insert_run_2_here", # Replace with your other run IDs
    "chacloutier-4B/WOFOST-RL/insert_run_3_here"
]

# 3. Setup the Matplotlib figure before the loop
plt.figure(figsize=(10, 5))

# 4. Loop through each run and plot its data
for path in run_paths:
    print(f"Fetching data for: {path}...")
    run = api.run(path)
    
    # Download history
    history = run.history(samples=10000000)
    
    # Define columns to pull (this prevents total deletion from dropna like we saw earlier)
    plot_cols = ["global_step", "charts/average_reward", "constraints/mean_episodic_cost"]
    
    # Check which columns actually exist in this run to avoid KeyError
    available_cols = [col for col in plot_cols if col in history.columns]
    
    # Clean the data: Forward fill gaps, then drop remaining NaNs
    history_clean = history[available_cols].ffill().dropna()

    # Use run.name to automatically label the legend with the wandb display name
    run_label = run.name 

    # --- Plot Yield ---
    if "charts/average_reward" in history_clean.columns:
        plt.plot(history_clean["global_step"], history_clean["charts/average_reward"], 
                 label=f"Yield: {run_label}", alpha=0.8)

    # --- Plot Constraints (Uncomment if you want them on the same graph!) ---
    # if "constraints/mean_episodic_cost" in history_clean.columns:
    #     plt.plot(history_clean["global_step"], history_clean["constraints/mean_episodic_cost"], 
    #              label=f"Cost: {run_label}", linestyle=':', alpha=0.7)


# 5. Global Plot Formatting
# Add a threshold line to show your 0.05 budget limit
plt.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label="Cost Limit (0.05)")

# Formatting for your thesis
plt.title("PPO-Lagrangian: Run Comparisons over Time")
plt.xlabel("Global Step")
plt.ylabel("Value")

# Move legend outside the plot if it gets too crowded, or keep it 'best'
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid(True, alpha=0.3)

# Adjust layout so the legend doesn't get cut off when saving
plt.tight_layout()

plt.show()
# plt.savefig("ppo_lag_multi_run_results.png", dpi=300)