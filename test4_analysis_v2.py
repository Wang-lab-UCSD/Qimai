import pandas as pd
import numpy as np
import json
import glob
import os
import re
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score, # Still needed for calculation
    average_precision_score, # Still needed for calculation
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    # roc_curve, # REMOVED
    # precision_recall_curve, # REMOVED
    # RocCurveDisplay, # REMOVED
    # PrecisionRecallDisplay, # REMOVED
)
# from sklearn.calibration import calibration_curve, CalibrationDisplay # REMOVED
import warnings
import traceback
from typing import Union, List

# constants
n_bins = 5

# --- Helper function to check explanation/label consistency ---
# (Copied from agent script - ensure it's identical if you modify it there)
def check_explanation_conclusion_sentiment(explanation: str) -> Union[int, None]:
    """Performs a simple keyword check on the explanation's likely conclusion."""
    if not explanation or not isinstance(explanation, str):
        return None
    conclusion_text = explanation[-150:].lower() # Check near the end
    positive_keywords = ["likely interaction", "high probability", "strong evidence for binding", "interaction detected", "does interact", "binds", "predict an interaction","an interaction occurs","interacts with"]
    negative_keywords = ["unlikely interaction", "low probability", "no significant evidence", "no interaction", "does not interact", "minimal interaction", "absence of hits"]
    found_positive = any(kw in conclusion_text for kw in positive_keywords)
    found_negative = any(kw in conclusion_text for kw in negative_keywords)
    if found_positive and not found_negative: return 1
    elif found_negative and not found_positive: return 0
    else: return None # Ambiguous
# -----------------------------------------------------------------------

# --- NEW: Function to plot metrics vs. confidence bins ---
def plot_metric_by_confidence_bins(
    y_true: np.ndarray,
    y_pred_labels_from_llm: np.ndarray,
    y_prob_confidence: np.ndarray,
    metric_to_plot: str,
    n_bins: int,
    group_name: str,
    plot_path_prefix: str
):
    """
    Plots a specified metric (e.g., accuracy) across confidence bins.

    Args:
        y_true (np.array): True labels.
        y_pred_labels_from_llm (np.array): Predicted labels (0 or 1) from the LLM.
        y_prob_confidence (np.array): Predicted probabilities/confidence scores for the positive class, used for binning.
        metric_to_plot (str): Name of the metric to plot (e.g., 'Accuracy', 'Precision').
        n_bins (int): Number of confidence bins.
        group_name (str): Name of the model group for title.
        plot_path_prefix (str): Prefix for saving the plot file (e.g., "output_dir/model_style").
    """
    print(f"  Generating {metric_to_plot} vs. Confidence plot for {group_name}...")
    try:
        bin_limits = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_limits[:-1] + bin_limits[1:]) / 2
        metric_values = []
        bin_counts = []

        for i in range(n_bins):
            lower_bound = bin_limits[i]
            upper_bound = bin_limits[i+1]
            
            if i == n_bins - 1: # Ensure the last bin includes 1.0
                bin_mask = (y_prob_confidence >= lower_bound) & (y_prob_confidence <= upper_bound)
            else:
                bin_mask = (y_prob_confidence >= lower_bound) & (y_prob_confidence < upper_bound)

            y_true_bin = y_true[bin_mask]
            y_pred_bin = y_pred_labels_from_llm[bin_mask]

            bin_counts.append(len(y_true_bin))

            if len(y_true_bin) > 5: # Min samples in bin for stable metric
                score = np.nan
                if metric_to_plot.lower() == 'accuracy':
                    score = accuracy_score(y_true_bin, y_pred_bin)
                elif metric_to_plot.lower() == 'precision':
                    score = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                # Add other metrics like F1, Recall here if needed
                elif metric_to_plot.lower() == 'f1':
                    score = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                # --- START ADDITION FOR MCC ---
                elif metric_to_plot.lower() == 'mcc':
                    try:
                        # MCC can warn if predictions are all one class or ground truth is all one class in a bin
                        # This can happen with small bins.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore undefined MCC warnings for a bin
                            score = matthews_corrcoef(y_true_bin, y_pred_bin)
                    except Exception as e_mcc_bin:
                        print(f"    WARN: Could not calculate MCC for bin {i} in {group_name}: {e_mcc_bin}")
                        score = np.nan # Set to NaN if calculation fails
                                        
                else:
                    print(f"  WARN: Metric '{metric_to_plot}' not supported for confidence bin plot. Skipping plot generation.")
                    return
                metric_values.append(score)
            else:
                metric_values.append(np.nan)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        ax1.plot(bin_centers, metric_values, marker='o', linestyle='-', label=metric_to_plot, color='blue')
        ax1.set_xlabel('Mean Predicted Confidence (Positive Class Bins)')
        ax1.set_ylabel(metric_to_plot, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xticks(bin_centers)
        ax1.set_xticklabels([f'{c:.2f}' for c in bin_centers], rotation=45, ha="right")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        ax1.set_title(f'{metric_to_plot} vs. Confidence Bins - {group_name}')

        ax2 = ax1.twinx()
        ax2.bar(bin_centers, bin_counts, width=(bin_limits[1]-bin_limits[0])*0.8, alpha=0.3, color='gray', label='Sample Count')
        ax2.set_ylabel('Number of Samples', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(bottom=0, top=max(10, max(bin_counts)*1.1 if any(c > 0 for c in bin_counts) else 10))

        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        
        all_handles = []
        all_labels = []
        if lines and labels:
            all_handles.extend(lines)
            all_labels.extend(labels)
        if bars and bar_labels:
            all_handles.extend(bars)
            all_labels.extend(bar_labels)

        if all_handles: # Only add legend if there's something to show
            # Place legend below the plot
            fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=max(1, len(all_handles)))
        
        fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for legend and title
        
        plot_filename = f"{plot_path_prefix}_{metric_to_plot.lower().replace(' ', '_')}_vs_confidence.png"
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"  {metric_to_plot} vs. Confidence plot saved to: {plot_filename}")

    except Exception as e_plot:
        print(f"  ERROR generating {metric_to_plot} vs. Confidence plot for {group_name}: {e_plot}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
# ----------------------------------------------------

# --- NEW: Function to create comparison bar plots (Metrics on X-axis) ---
def plot_comparison_bars_metrics_on_x(summary_df: pd.DataFrame, metrics_to_plot: List[str], output_dir: str):
    """
    Generates grouped bar plots with metrics on the x-axis and models as legends.

    Args:
        summary_df (pd.DataFrame): DataFrame containing the calculated metrics summary.
                                 Expects 'Model Group' and 'Prompt Style' columns.
        metrics_to_plot (list): List of metric column names to display on the x-axis.
        output_dir (str): Directory to save the plot.
    """
    print(f"\n--- Generating Comparison Bar Plot (Metrics on X-axis) for: {', '.join(metrics_to_plot)} ---")
    try:
        df_plot = summary_df.copy()
        df_plot['Model & Style'] = df_plot['Model Group'] + "\n(" + df_plot['Prompt Style'] + ")"
        df_filtered = df_plot[['Model & Style'] + metrics_to_plot]
        df_pivot = df_filtered.set_index('Model & Style')
        df_to_plot = df_pivot[metrics_to_plot].T

        if df_to_plot.empty or df_to_plot.isnull().all().all():
            print("  WARN: No data available for plotting comparison bars (metrics on x-axis).")
            return

        num_metrics_on_x = len(df_to_plot.index)
        num_model_styles = len(df_to_plot.columns)

        if num_model_styles == 0:
            print("  WARN: No model/style combinations to plot.")
            return

        bar_width = 0.8 / num_model_styles
        index = np.arange(num_metrics_on_x)

        fig_width = max(10, num_metrics_on_x * num_model_styles * 0.3) # Dynamic width
        fig_height = 7 
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for i, model_style_combo in enumerate(df_to_plot.columns):
            offset = (i - (num_model_styles - 1) / 2) * bar_width
            metric_values = df_to_plot[model_style_combo]
            
            valid_idx = metric_values.notna()
            if valid_idx.any():
                bars = ax.bar(index[valid_idx] + offset, metric_values[valid_idx], bar_width, label=model_style_combo)
                # ax.bar_label(bars, fmt='%.2f', padding=3, fontsize='xx-small', rotation=90) 

        ax.set_ylabel('Metric Score')
        ax.set_xlabel('Metrics')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(index)
        ax.set_xticklabels(df_to_plot.index, rotation=45, ha="right", fontsize='medium')
        ax.legend(title="Model & Style", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize='small')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        all_vals_flat = df_to_plot.values.flatten()
        all_vals_flat = all_vals_flat[~np.isnan(all_vals_flat)]
        
        min_val = 0.0
        if len(all_vals_flat) > 0:
            min_val = np.min(all_vals_flat)
        
        bottom_ylim = min(0.0, min_val - 0.05) if min_val <= 0 else 0.0
        ax.set_ylim(bottom=bottom_ylim, top=1.05)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plot_path = os.path.join(output_dir, "comparison_bar_plot_metrics_on_x.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Comparison bar plot (metrics on x-axis) saved to: {plot_path}")

    except Exception as e_plot:
        print(f"  ERROR generating comparison bar plot (metrics on x-axis): {e_plot}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
# -----------------------------------------------------------------------

def analyze_results(input_dir, analysis_output_dir):
    print(f"Starting analysis...")
    print(f"Input directory: {input_dir}")
    print(f"Analysis output directory: {analysis_output_dir}")

    plot_output_dir = os.path.join(analysis_output_dir, "plots")
    try:
        os.makedirs(analysis_output_dir, exist_ok=True)
        os.makedirs(plot_output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create output directories: {e}", file=sys.stderr)
        return

    json_files = glob.glob(os.path.join(input_dir, "*_prompt_*_aggregated_simple_results.json"))
    if not json_files:
        print(f"ERROR: No '*_prompt_*_aggregated_simple_results.json' files found directly in '{input_dir}'.", file=sys.stderr)
        print("       Ensure evaluation script ran and saved aggregated results with the correct naming pattern.")
        return

    print(f"Found {len(json_files)} aggregated result files.")
    all_results_dfs = []
    file_pattern = re.compile(r"^(.*?)_prompt_(verbose|concise|transformer-priority|motif-priority|auto)_aggregated_simple_results\.json$")

    for f_path in json_files:
        try:
            filename = os.path.basename(f_path)
            match = file_pattern.match(filename)
            if not match:
                print(f"WARN: Skipping file with unexpected filename format: {filename}")
                continue
            model_name, prompt_style = match.group(1), match.group(2)
            print(f"  Loading: {filename} (Model Group: {model_name}, Prompt: {prompt_style})")

            with open(f_path, 'r') as f: data = json.load(f)
            if not data: print(f"  WARN: File is empty: {f_path}"); continue

            df = pd.DataFrame(data)
            df['model_name_group'], df['prompt_style'] = model_name, prompt_style
            all_results_dfs.append(df)
        except json.JSONDecodeError: print(f"WARN: Skipping invalid JSON file: {f_path}", file=sys.stderr)
        except Exception as e: print(f"WARN: Error processing file {f_path}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)

    if not all_results_dfs: print("ERROR: No valid data loaded.", file=sys.stderr); return
    results_df = pd.concat(all_results_dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(results_df)}")

    print("--- Cleaning and Preparing Data ---")
    initial_count = len(results_df)

    results_df = results_df[results_df['error'].isna()].copy()
    print(f"Filtered {initial_count - len(results_df)} records with agent errors.")
    
    pred_na_count = results_df['predicted_label'].isna().sum()
    if pred_na_count > 0: print(f"Filtered {pred_na_count} records with null 'predicted_label'.")
    results_df = results_df[results_df['predicted_label'].notna()].copy()

    gt_na_count = results_df['ground_truth_label'].isna().sum()
    if gt_na_count > 0: print(f"Filtered {gt_na_count} records with null 'ground_truth_label'.")
    results_df = results_df[results_df['ground_truth_label'].notna()].copy()

    if not results_df.empty:
        try:
            results_df['ground_truth_label'] = results_df['ground_truth_label'].astype(int)
            results_df['predicted_label'] = results_df['predicted_label'].astype(int)
        except ValueError as e: print(f"ERROR: Converting labels to int: {e}", file=sys.stderr); return

    if 'confidence_score' in results_df.columns:
        results_df['confidence_score'] = results_df['confidence_score'].fillna(0.5)
        results_df['confidence_score'] = results_df['confidence_score'].replace([np.inf, -np.inf], 0.5)
        results_df['confidence_score'] = results_df['confidence_score'].astype(float)
        results_df['confidence_score'] = np.clip(results_df['confidence_score'], 0.0, 1.0)
        if results_df['confidence_score'].isna().any():
            print("WARN: NaNs found in confidence scores after cleaning. Filling with 0.5.")
            results_df['confidence_score'] = results_df['confidence_score'].fillna(0.5)
    else:
        print("WARN: 'confidence_score' column missing. Defaulting to 0.5. Some metrics might be affected.")
        results_df['confidence_score'] = 0.5

    if 'llm_explanation' in results_df.columns:
        print("Calculating explanation consistency...")
        results_df['explanation_consistent'] = results_df.apply(
            lambda r: (check_explanation_conclusion_sentiment(r['llm_explanation']) == r['predicted_label'])
                       if pd.notna(r['predicted_label']) and check_explanation_conclusion_sentiment(r['llm_explanation']) is not None else None, axis=1)
    else:
        print("WARN: 'llm_explanation' missing. Consistency metrics skipped.")
        results_df['explanation_consistent'] = None
    
    final_valid_count = len(results_df)
    print(f"Total valid records for analysis: {final_valid_count} (out of {initial_count})")
    if final_valid_count == 0: print("ERROR: No valid records remaining.", file=sys.stderr); return

    metrics_summary = []
    grouped = results_df.groupby(['model_name_group', 'prompt_style'])
    print("\n--- Calculating Metrics & Generating Plots per Group ---")

    for (model_group, style), group_df in grouped:
        group_name = f"{model_group} ({style})"
        print(f"\nAnalyzing Group: {group_name}")
        group_metrics = {'Model Group': model_group, 'Prompt Style': style, 'Samples': len(group_df)}

        y_true = group_df['ground_truth_label'].values
        y_pred = group_df['predicted_label'].values
        y_prob_raw_confidence = group_df['confidence_score'].values # For new plot binning
        y_prob_metric_calc = np.clip(y_prob_raw_confidence, 1e-15, 1 - 1e-15) # For Brier, AUCs

        if len(y_true) == 0: print("  No valid samples. Skipping."); continue
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        has_both_classes = len(unique_true) == 2

        group_metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        group_metrics['F1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        group_metrics['Precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        group_metrics['Recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4: tn, fp, fn, tp = cm.ravel(); group_metrics['Specificity (Pos=0)'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else: group_metrics['Specificity (Pos=0)'] = np.nan
        
        try: group_metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        except Exception: group_metrics['MCC'] = np.nan; print(f"  WARN: MCC calculation error for {group_name}.")

        if has_both_classes:
            group_metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob_metric_calc)
            group_metrics['AUC-PR (AvgPrec)'] = average_precision_score(y_true, y_prob_metric_calc, pos_label=1)
        else:
            g_warn = f"  WARN: Only class {unique_true[0]} (n={counts_true[0]}) present in ground truth for {group_name}. "
            if len(y_pred) > 0:
                unique_pred = np.unique(y_pred)
                g_warn += f"Predictions have classes: {unique_pred}. "
            g_warn += "AUC/AP metrics undefined."
            print(g_warn)
            group_metrics['AUC-ROC'], group_metrics['AUC-PR (AvgPrec)'] = np.nan, np.nan
        
        group_metrics['Brier Score'] = brier_score_loss(y_true, y_prob_metric_calc, pos_label=1)

        if 'explanation_consistent' in group_df.columns and group_df['explanation_consistent'].notna().any():
            valid_checks = group_df['explanation_consistent'].notna().sum()
            consistent_count = group_df['explanation_consistent'].sum() # NaNs are treated as False in sum if not careful; ensure boolean or handle
            group_metrics['Explanation Consistency %'] = (consistent_count / valid_checks * 100) if valid_checks > 0 else np.nan
            group_metrics['Inconsistent Expl Count'] = valid_checks - consistent_count
        else:
            group_metrics['Explanation Consistency %'], group_metrics['Inconsistent Expl Count'] = np.nan, np.nan
        
        metrics_summary.append(group_metrics)
        print(f"  Metrics calculated for {group_name}.")

        plot_prefix = os.path.join(plot_output_dir, re.sub(r'[^\w\-_\.]', '_', group_name.replace(' ', '_')))
        try:
            plot_metric_by_confidence_bins(
                y_true=y_true, y_pred_labels_from_llm=y_pred,
                y_prob_confidence=y_prob_raw_confidence,
                metric_to_plot='Accuracy', n_bins=n_bins,
                group_name=group_name, plot_path_prefix=plot_prefix
            )

            plot_metric_by_confidence_bins(
                y_true=y_true, y_pred_labels_from_llm=y_pred,
                y_prob_confidence=y_prob_raw_confidence,
                metric_to_plot='MCC', n_bins=n_bins,
                group_name=group_name, plot_path_prefix=plot_prefix
            )
            plot_metric_by_confidence_bins(
                y_true=y_true, y_pred_labels_from_llm=y_pred,
                y_prob_confidence=y_prob_raw_confidence,
                metric_to_plot="F1", n_bins=n_bins,
                group_name=group_name, plot_path_prefix=plot_prefix
            )      
            plot_metric_by_confidence_bins(
                y_true=y_true, y_pred_labels_from_llm=y_pred,
                y_prob_confidence=y_prob_raw_confidence,
                metric_to_plot="Precision", n_bins=n_bins,
                group_name=group_name, plot_path_prefix=plot_prefix
            )                      
        except Exception as e_plot: print(f"  ERROR generating plots for {group_name}: {e_plot}", file=sys.stderr)

    print("\nIndividual group plotting complete. Combined ROC/PR plots were removed.")

    if not metrics_summary: print("\nERROR: No metrics calculated for any group.", file=sys.stderr); return
    summary_df = pd.DataFrame(metrics_summary)
    cols_order = ['Model Group', 'Prompt Style', 'Samples', 'Accuracy', 'F1', 'MCC',
                  'Precision', 'Recall', 'Specificity (Pos=0)',
                  'AUC-ROC', 'AUC-PR (AvgPrec)', 'Brier Score',
                  'Explanation Consistency %', 'Inconsistent Expl Count']
    summary_df = summary_df.reindex(columns=[col for col in cols_order if col in summary_df.columns]).round(4)

    print("\n--- Overall Performance Summary ---")
    print(summary_df.to_string())
    summary_csv_path = os.path.join(analysis_output_dir, "performance_summary.csv")
    try: summary_df.to_csv(summary_csv_path, index=False); print(f"\nSummary table saved to: {summary_csv_path}")
    except Exception as e: print(f"ERROR saving summary CSV: {e}", file=sys.stderr)

    metrics_for_bar_plot = ['Accuracy','Precision','Recall','F1', 'MCC']
    valid_metrics = [m for m in metrics_for_bar_plot if m in summary_df.columns and summary_df[m].notna().any()]
    if valid_metrics: plot_comparison_bars_metrics_on_x(summary_df, valid_metrics, analysis_output_dir)
    else: print("\nSkipping comparison bar plot: no valid data for selected metrics.")

    print("\nAnalysis script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze DPI-Agent evaluation results from multiple runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="Input directory containing the aggregated simple result JSON files. " \
             "Files must be directly in this directory (e.g., './aggregated_outputs/') and " \
             "follow the naming pattern 'MODEL_prompt_STYLE_aggregated_simple_results.json'."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./analysis_results",
        help="Directory where analysis plots and summary CSV will be saved."
    )
    args = parser.parse_args()
    analyze_results(args.input_dir, args.output_dir)
