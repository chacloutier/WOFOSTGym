import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def plot_lambdas(runs_config, metric_name="charts/lambda", max_steps=1000000, smoothing_weight=0.95):
    """
    Plots the evolution of Lagrange multipliers for a specific list of W&B runs.
    """
    api = wandb.Api()
    
    plt.figure(figsize=(10, 6))
    plotted_runs = 0
    
    for run_info in runs_config:
        path = run_info["path"]
        custom_label = run_info.get("label", path) 
        custom_color = run_info.get("color", None) 
        
        try:
            print(f"Fetching run: {path}...")
            run = api.run(path)
            
            # Fetch the metric history
            history = run.history(keys=["global_step", metric_name], samples=100000)
            
            if metric_name in history.columns:
                
                # Drop NaNs and sort chronologically
                clean_history = history[["global_step", metric_name]].dropna()
                clean_history = clean_history.sort_values("global_step").reset_index(drop=True)
                
                if not clean_history.empty:
                    # Extend line if run stopped early
                    last_valid_value = clean_history.iloc[-1][metric_name] 
                    current_max_step = clean_history["global_step"].max()
                    
                    if current_max_step < max_steps:
                        extension_row = pd.DataFrame({"global_step": [max_steps], metric_name: [last_valid_value]})
                        clean_history = pd.concat([clean_history, extension_row], ignore_index=True)
                    
                    clean_history = clean_history[clean_history["global_step"] <= max_steps]

                    y_raw = clean_history[metric_name]

                    plt.plot(
                        clean_history["global_step"], 
                        y_raw, 
                        color=custom_color,         
                        linewidth=1,        
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
    plt.xlabel("Global Step", fontsize=12, fontweight='bold')
    
    # Dynamically label the Y-axis based on the metric provided
    clean_metric_name = metric_name.split('/')[-1].replace('_', ' ').title()
    plt.ylabel(f"{clean_metric_name} Penalty Weight", fontsize=12, fontweight='bold') 
    
    plt.title(f"Evolution of {clean_metric_name} During Training", fontsize=14, fontweight='bold', pad=15)
    
    plt.xlim(0, max_steps)
    
    # Format X-axis to show "200k, 400k"
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x != 0 else '0'))
    
    # Force Y-axis to start at 0, since lambda cannot be negative
    plt.ylim(bottom=0)
    
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True)
    legend.get_frame().set_edgecolor('gray')
    
    plt.tight_layout()
    plt.savefig("lambda_evolution.png", dpi=300)
    print("\nPlot saved as 'lambda_evolution.png'")
    plt.show()

if __name__ == "__main__":
    # Change this to "charts/lambda_n", "charts/lambda_w", etc. to isolate a resource
    METRIC = "charts/lambda" 
    
    # Example configuration using a few distinct hex colors
    RUNS_CONFIG = [
        {
            "path": "chacloutier-4B/WOFOST-RL/awcxfins", 
            "label": "Config 1", 
            "color": "#E69F00", 
        },
        {
            "path": "chacloutier-4B/WOFOST-RL/762482g0", 
            "label": "Config 2", 
            "color": "#0072B2",
        },
        {
            "path": "chacloutier-4B/WOFOST-RL/0y70rbij", 
            "label": "Config 3", 
            "color": "#CC79A7",
        },
        {
            "path": "chacloutier-4B/WOFOST-RL/5bvrob8i", 
            "label": "Config 4", 
            "color": "#009E73",
        }
    ]

    plot_lambdas(RUNS_CONFIG, METRIC, max_steps=1000000, smoothing_weight=0.95)