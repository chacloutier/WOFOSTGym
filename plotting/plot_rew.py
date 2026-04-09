import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional: Use a cleaner built-in matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')

def plot_specific_runs(runs_config, metric_name="charts/average_reward", max_steps=1000000, smoothing_weight=0.95):
    api = wandb.Api()
    
    plt.figure(figsize=(12, 7))
    plotted_runs = 0
    
    for run_info in runs_config:
        path = run_info["path"]
        custom_label = run_info.get("label", path) 
        custom_color = run_info.get("color", None) 
        
        try:
            print(f"Fetching run: {path}...")
            run = api.run(path)
            
            history = run.history(keys=["global_step", metric_name], samples=100000)
            
            if metric_name in history.columns:
                
                clean_history = history[["global_step", metric_name]].dropna()
                clean_history = clean_history.sort_values("global_step").reset_index(drop=True)
                
                if not clean_history.empty:
                    # Find last value and extend if it stopped early
                    non_zero_history = clean_history[clean_history[metric_name] != 0]
                    last_valid_value = non_zero_history.iloc[-1][metric_name] if not non_zero_history.empty else clean_history.iloc[-1][metric_name]
                    current_max_step = clean_history["global_step"].max()
                    
                    if current_max_step < max_steps:
                        extension_row = pd.DataFrame({"global_step": [max_steps], metric_name: [last_valid_value]})
                        clean_history = pd.concat([clean_history, extension_row], ignore_index=True)
                    
                    clean_history = clean_history[clean_history["global_step"] <= max_steps]

                    # 1. Apply custom symlog transformation
                    y_raw = clean_history[metric_name]
                    y_symlog = np.sign(y_raw) * np.log10(np.abs(y_raw) + 1)
                    
                    # 2. Calculate Exponential Moving Average (EMA)
                    # A higher smoothing_weight (e.g. 0.99) means smoother curves
                    span = int(1 / (1 - smoothing_weight))
                    y_smoothed = y_symlog.ewm(span=span, adjust=False).mean()

                    # 3. Plot the raw data faded in the background
                    plt.plot(
                        clean_history["global_step"], 
                        y_symlog, 
                        color=custom_color,
                        alpha=0.25,          # Very faded!
                        linewidth=0.5,       # Very thin!
                        zorder=1             # Push to back
                    )

                    # 4. Plot the smoothed line bolded on top
                    plt.plot(
                        clean_history["global_step"], 
                        y_smoothed, 
                        label=custom_label, 
                        color=custom_color,
                        alpha=1.0,           # Full solid color
                        linewidth=2.0,       # Thicker line
                        zorder=2             # Pull to front
                    )
                    plotted_runs += 1
                else:
                    print(f"  -> Warning: Run {run.name} had no valid data.")
            else:
                print(f"  -> Warning: Metric '{metric_name}' not found in run {run.name}")
                
        except wandb.errors.CommError:
            print(f"  -> Error: Could not find run at path '{path}'.")

    if plotted_runs == 0:
        print(f"\nNo data could be plotted.")
        return

    # Formatting the plot
    plt.xlabel("Global Step", fontsize=15, fontweight='bold')
    plt.ylabel("Average Reward (Symmetric Log Scale)", fontsize=15, fontweight='bold') 
    plt.title("Agent Training Performance", fontsize=20, fontweight='bold', pad=15)
    
    plt.xlim(0, max_steps)
    
    # Format X-axis to show "200k, 400k" instead of scientific notation "2e5"
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x != 0 else '0'))
    
    # Clean up the legend
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=15, frameon=True, shadow=True)
    legend.get_frame().set_edgecolor('gray')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    METRIC = "charts/average_reward" 
    
    RUNS_CONFIG = [
        {
            "path": "chacloutier-4B/WOFOST-RL/myqiethm", 
            "label": "DQN", 
            "color": "#E69F00", # Using colorblind-friendly hex codes!
        },
        {
            "path": "chacloutier-4B/WOFOST-RL/7amjp9bm", 
            "label": "PPO", 
            "color": "#0072B2",
        },
        # {
        #     "path": "chacloutier-4B/WOFOST-RL/6w0s07zf", 
        #     "label": "SAC", 
        #     "color": "#CC79A7",
        # },
        {
            "path": "chacloutier-4B/WOFOST-RL/rfd9jmae", 
            "label": "Baseline", 
            "color": "#009E73",
        },
    ]

    plot_specific_runs(RUNS_CONFIG, METRIC, max_steps=1000000, smoothing_weight=0.98)