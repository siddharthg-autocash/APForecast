# src/apforecast/reporting/visuals.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.apforecast.core.constants import REPORTS_DIR

def plot_model_curves(models_dict, run_date_str):
    """
    Generates a graph for every model (Specific Vendors & Global Cohorts).
    Saves them in reports/plots/<date>/
    """
    plot_dir = f"{REPORTS_DIR}/plots/{run_date_str}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Combine both dictionaries for easy looping
    all_models = {**models_dict['GLOBAL'], **models_dict['SPECIFIC']}
    
    print(f"Generating reference graphs for {len(all_models)} models...")

    for name, model in all_models.items():
        if model.n < 2: continue # Skip if not enough data to plot nicely
        
        # 1. Setup Data
        data = model.sorted_data # The actual historical days to settle
        
        # Create a figure with a secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # --- COLOR UPDATE ---
        # Matches your reference: Burnt Orange bars, Dark Blue line
        BAR_COLOR = '#E67E22'  # Burnt Orange
        LINE_COLOR = '#154360' # Dark Slate Blue
        
        # 2. Histogram (Orange Bars) - Frequency
        sns.histplot(data, bins=range(0, int(max(data))+5), color=BAR_COLOR, alpha=0.8, ax=ax1, stat='count', edgecolor=None)
        ax1.set_ylabel('Frequency (Count of Checks)', color=BAR_COLOR, fontweight='bold')
        ax1.set_xlabel('Days to Settle', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=BAR_COLOR)
        
        # 3. CDF Curve (Dark Blue Line) - Probability
        ax2 = ax1.twinx()
        
        # Calculate CDF for plotting
        x_vals = np.linspace(0, max(data)+5, 100)
        y_vals = [model.cdf(x) for x in x_vals]
        
        ax2.plot(x_vals, y_vals, color=LINE_COLOR, linewidth=3, label='Cumulative Probability')
        ax2.set_ylabel('Probability (0-100%)', color=LINE_COLOR, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelcolor=LINE_COLOR)
        
        # 4. Styling
        plt.title(f"Behavior Reference: {name}\n(Based on {model.n} historical checks)", fontsize=12)
        plt.grid(True, alpha=0.2, color='gray', linestyle='--')
        
        # Save
        safe_name = str(name).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{plot_dir}/{safe_name}.png", dpi=100)
        plt.close()
        
    print(f"Graphs saved in {plot_dir}")