import wandb
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. WandB Configuration
# ==========================================
# Replace these with your actual WandB details
ENTITY = "chacloutier-4B"   
PROJECT = "WOFOST-RL"

# The exact name of the metric as logged in your code
# Based on your earlier script, this was likely:
METRIC_NAME = "constraints/mean_episodic_cost" 

# Map your plot labels to their specific WandB Run IDs
# Find the Run ID at the end of the URL for each run
RUNS = {
    "Config 1 ($LR=0.1, K_p=0.1$)": "hb2vkbrj",
    "Config 2 ($LR=0.01, K_p=0.1$)": "ic48h636",
    "Config 3 ($LR=0.05, K_p=0.3$)": "u1qm3t15",
    "Config 4 ($LR=0.03, K_p=0.3$)": "0q4kco8d"
}

RUNS = {
    "Target Resource Utilization = 0.95": "0q4kco8d",
    "Target Resource Utilization = 0.85": "hpj157ix",
    "Target Resource Utilization = 0.80": "gicurq2f",
    "Target Resource Utilization = 0.75": "gyn1s1w5"
}

# ==========================================
# 2. Fetch Data via WandB API
# ==========================================
api = wandb.Api()
run_data = {}

print("Fetching data from WandB...")
for label, run_id in RUNS.items():
    # Construct the full path to the run
    run_path = f"{ENTITY}/{PROJECT}/{run_id}"
    run = api.run(run_path)
    
    # Fetch the history. 
    # 'samples=1000' downsamples the data evenly so you aren't trying 
    # to plot millions of individual steps, which keeps the graph clean and fast.
    history = run.history(keys=["global_step", METRIC_NAME], samples=1000000)
    
    # Drop NaNs just in case wandb returned empty rows for the metric
    history = history.dropna(subset=[METRIC_NAME])
    
    run_data[label] = {
        "global_steps": history["global_step"],
        "utilization": history[METRIC_NAME]
    }
print("Data fetched successfully!")

# ==========================================
# 3. Plotting
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
alphas = [0.7, 0.7, 0.7, 0.7]
# linewidths = [1.5, 2, 1.5, 2.5]
linewidths = [1, 1, 1, 1]

# Plot each fetched run
for idx, (label, data) in enumerate(run_data.items()):
    ax.plot(data["global_steps"], data["utilization"], 
            label=label, 
            color=colors[idx], 
            alpha=alphas[idx], 
            linewidth=linewidths[idx])

# Plot the target objective line
ax.axhline(y=0.95, color='tab:red', linestyle='--', linewidth=1)
ax.axhline(y=0.85, color='tab:orange', linestyle='--', linewidth=1)
ax.axhline(y=0.80, color='tab:blue', linestyle='--', linewidth=1)
ax.axhline(y=0.75, color='tab:green', linestyle='--', linewidth=1)

# Formatting the axes
ax.set_title('Mean Resource Utilization over Training Steps', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Mean Resource Utilization', fontsize=12)

# Format the x-axis ticks to show 'K' or 'M' for thousands/millions
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x < 1000000 else f'{x/1000000:g}M'))

# Set dynamic or static limits
# ax.set_xlim(0, 1_000_000)  # Uncomment to lock x-axis to exactly 1M steps
ax.set_ylim(0.6, 1.4)

# Add legend
ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)

# Layout adjustments and display
plt.tight_layout()
plt.show()

# Optional: Save the figure
# plt.savefig('tuning_results.png', dpi=300, bbox_inches='tight')