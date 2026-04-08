import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. WandB Configuration
# ==========================================
# Replace these with your actual WandB details
ENTITY = "chacloutier-4B"   
PROJECT = "WOFOST-RL"

# The metric logged in your training script
METRIC_NAME = "charts/average_reward"

# Map your plot labels to their specific WandB Run IDs
# (Find the Run ID at the end of the URL for each run)
RUNS = {
    "Target Resource Utilization = 0.95": "0q4kco8d",
    "Target Resource Utilization = 0.85": "hpj157ix",
    "Target Resource Utilization = 0.80": "gicurq2f",
    "Target Resource Utilization = 0.75": "gyn1s1w5"
}

# ==========================================
# 2. Fetch & Scale Data via WandB API
# ==========================================
api = wandb.Api()
run_data = {}

print("Fetching data from WandB...")
for label, run_id in RUNS.items():
    run_path = f"{ENTITY}/{PROJECT}/{run_id}"
    try:
        run = api.run(run_path)
        
        # Fetch the history, downsampling to 1000 points for a clean plot
        history = run.history(keys=["global_step", METRIC_NAME], samples=1000)
        history = history.dropna(subset=[METRIC_NAME])
        
        # --- CRITICAL STEP: Scale the reward by 1000 ---
        # This converts the 1e-3 scaled yield back to kg/ha 
        # and scales the RM bonuses to massive spikes.
        scaled_score = history[METRIC_NAME] * 1000.0
        
        run_data[label] = {
            "steps": history["global_step"],
            "score": scaled_score
        }
        print(f"  Successfully fetched: {label}")
    except Exception as e:
        print(f"  Error fetching {label} ({run_id}): {e}")

# ==========================================
# 3. Plotting
# ==========================================
# Set a clean, professional style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
alphas = [0.7, 0.8, 0.7, 0.9]
linewidths = [1.5, 2, 1.5, 2.5]

# Plot each fetched and scaled run
for idx, (label, data) in enumerate(run_data.items()):
    ax.plot(data["steps"], data["score"], 
            label=label, 
            color=colors[idx], 
            alpha=alphas[idx], 
            linewidth=linewidths[idx])

# Formatting the axes
ax.set_title('Average Reward (kg/ha Equivalent Score)', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('kg/ha Equivalent Score\n(Yield + RM Bonuses)', fontsize=12)

# Format the x-axis ticks to show 'K' or 'M' for thousands/millions
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x < 1000000 else f'{x/1000000:g}M'))

# Set x-axis limit based on your training run length
ax.set_xlim(0, 1_000_000) 

# Add legend
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

# Layout adjustments and display
plt.tight_layout()
plt.show()

# Optional: Save the figure
# plt.savefig('scaled_reward_results.png', dpi=300, bbox_inches='tight')