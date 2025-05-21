import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # For np.arange
import os

def generate_mean_std_comparison_plot(
    csv_file_path, 
    model_group1_name, 
    model_group2_name, 
    data_to_plot = 'chip690',
    metrics_to_plot=["Accuracy", "F1", "MCC", "Precision", "Recall"],
    output_directory=".", 
    output_filename="model_mean_std_comparison.png"
):
    """
    Generates and saves a bar plot comparing the mean performance (with std error bars)
    of two model groups across different runs for specified metrics.

    Args:
        csv_file_path (str): Path to the input CSV file.
        model_group1_name (str): Name of the first Model Group.
        model_group2_name (str): Name of the second Model Group.
        metrics_to_plot (list, optional): Metrics to plot.
        output_directory (str, optional): Directory to save the plot.
        output_filename (str, optional): Filename for the saved plot.
    """
    df = pd.read_csv(csv_file_path)
    df = df[df.data==data_to_plot]


    # Validate required columns
    required_cols = ["Model Group"] + metrics_to_plot # "data" and "random seed" are implicit in the runs
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: CSV missing required columns: {', '.join(missing)}")
        return
    
    # Ensure metrics are numeric
    for metric in metrics_to_plot:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Filter for selected models
    df_models = df[df["Model Group"].isin([model_group1_name, model_group2_name])]
    if df_models.empty:
        print(f"No data found for model groups: {model_group1_name}, {model_group2_name}")
        return
    if len(df_models["Model Group"].unique()) < 2 :
        print(f"Could not find data for both specified model groups. Found: {df_models['Model Group'].unique()}")
        return


    # Calculate mean and std for each model group and metric
    summary_stats = df_models.groupby("Model Group")[metrics_to_plot].agg(['mean', 'std']).reset_index()
    
    # Flatten MultiIndex columns (e.g., ('Accuracy', 'mean') -> 'Accuracy_mean')
    summary_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary_stats.columns.values]
    
    model1_means = []
    model1_stds = []
    model2_means = []
    model2_stds = []

    for metric in metrics_to_plot:
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"

        m1_row = summary_stats[summary_stats["Model Group"] == model_group1_name]
        m2_row = summary_stats[summary_stats["Model Group"] == model_group2_name]

        if not m1_row.empty:
            model1_means.append(m1_row[metric_mean_col].iloc[0] if metric_mean_col in m1_row else 0)
            model1_stds.append(m1_row[metric_std_col].iloc[0] if metric_std_col in m1_row else 0)
        else:
            print(f"Warning: No data for {model_group1_name} to calculate mean/std for {metric}.")
            model1_means.append(0)
            model1_stds.append(0)

        if not m2_row.empty:
            model2_means.append(m2_row[metric_mean_col].iloc[0] if metric_mean_col in m2_row else 0)
            model2_stds.append(m2_row[metric_std_col].iloc[0] if metric_std_col in m2_row else 0)
        else:
            print(f"Warning: No data for {model_group2_name} to calculate mean/std for {metric}.")
            model2_means.append(0)
            model2_stds.append(0)

    # Plotting
    n_metrics = len(metrics_to_plot)
    index = np.arange(n_metrics) # x locations for the groups
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4)) # Adjust width based on number of metrics
    palette = sns.color_palette("viridis", 2)

    rects1 = ax.bar(index - bar_width/2, model1_means, bar_width, yerr=model1_stds,
                    label=model_group1_name, color=palette[0], capsize=5, ecolor='gray')
    rects2 = ax.bar(index + bar_width/2, model2_means, bar_width, yerr=model2_stds,
                    label=model_group2_name, color=palette[1], capsize=5, ecolor='gray')

    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title(f'Mean Performance Comparison: {model_group1_name} vs {model_group2_name}\n(Error bars: +/- 1 SD)', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(metrics_to_plot, rotation=0, ha="center", fontsize=10)
    ax.legend(title="Model Group", loc="upper right")
    ax.grid(axis='y', linestyle='--')

    # Move legend to the right panel
    # Shrink current axis's width to make space for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # Shrink width to 80%
    ax.legend(title="Model Group", loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add bar labels for mean values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if pd.notna(height):
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3 if height >=0 else -12),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom' if height >=0 else 'top', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    os.makedirs(output_directory, exist_ok=True)
    plot_path = os.path.join(output_directory, output_filename)
    try:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # plt.show() # Uncomment to display the plot interactively

if __name__ == '__main__':
    # --- Example Usage ---
    # Replace with your actual file path and model names
    csv_file = './benchmark/benchmark_v3.csv'  
    model1 = 'gemini_gemini-2.0-flash'   
    model2 = 'DPI'                 
    
    # Define metrics you want to plot (optional, defaults are provided in function)
    custom_metrics = ["Accuracy", "F1", "MCC", "Precision", "Recall"] 
    
    # Call the function
    generate_mean_std_comparison_plot(
        csv_file_path=csv_file,
        model_group1_name=model1,
        model_group2_name=model2,
        metrics_to_plot=custom_metrics, 
        data_to_plot='chip690',
        output_directory="./benchmark",
        output_filename=f"{model1}_vs_{model2}_mean_std_comparison_chip690.png"
    )

    # Example with default metrics if you omit metrics_to_plot
    # generate_mean_std_comparison_plot(
    #     csv_file_path=csv_file,
    #     model_group1_name=model1,
    #     model_group2_name=model2
    # )