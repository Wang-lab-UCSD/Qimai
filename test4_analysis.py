# analyze results generated from test4_eval.py

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
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
import warnings
import traceback
from typing import Union

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

# --- NEW: Function to create comparison bar plots ---
def plot_comparison_bars(summary_df, metrics_to_plot, output_dir):
    """
    Generates grouped bar plots comparing specified metrics across models/styles.

    Args:
        summary_df (pd.DataFrame): DataFrame containing the calculated metrics summary.
        metrics_to_plot (list): List of metric column names to include in the plot.
        output_dir (str): Directory to save the plot.
    """
    print(f"\n--- Generating Comparison Bar Plot for metrics: {', '.join(metrics_to_plot)} ---")
    try:
        # Prepare data for plotting
        df_plot = summary_df.copy()
        # Create a combined label for the x-axis
        df_plot['Model & Style'] = df_plot['Model Group'] + "\n(" + df_plot['Prompt Style'] + ")"
        df_plot = df_plot.set_index('Model & Style')

        # Select only the metrics we want to plot
        df_metrics = df_plot[metrics_to_plot]

        if df_metrics.empty:
             print("  WARN: No data available for plotting comparison bars.")
             return

        num_models = len(df_metrics)
        num_metrics = len(metrics_to_plot)
        bar_width = 0.8 / num_metrics # Adjust bar width based on number of metrics
        index = np.arange(num_models) # Positions for model groups

        fig, ax = plt.subplots(figsize=(max(8, num_models * num_metrics * 0.6), 6)) # Dynamic width

        # Plot bars for each metric
        for i, metric in enumerate(metrics_to_plot):
            # Calculate offset for each metric's bars within a group
            offset = (i - (num_metrics - 1) / 2) * bar_width
            # Handle potential NaN values - plot where not NaN
            valid_idx = df_metrics[metric].notna()
            if valid_idx.any(): # Only plot if there's valid data
                 bars = ax.bar(index[valid_idx] + offset, df_metrics[metric][valid_idx], bar_width, label=metric)
                 # Optional: Add labels on top of bars (can get crowded)
                 # ax.bar_label(bars, fmt='%.3f', padding=3, fontsize='x-small')

        # --- Formatting ---
        ax.set_ylabel('Metric Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(index)
        ax.set_xticklabels(df_metrics.index, rotation=45, ha="right", fontsize='small') # Rotate labels if many models
        ax.legend(title="Metrics", bbox_to_anchor=(1.04, 1), loc="upper left") # Legend outside plot
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=min(0, df_metrics.min().min() - 0.05), top=1.05) # Adjust y-lim slightly beyond 0-1

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plot_path = os.path.join(output_dir, "comparison_bar_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Comparison bar plot saved to: {plot_path}")

    except Exception as e_plot:
        print(f"  ERROR generating comparison bar plot: {e_plot}", file=sys.stderr)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)


def analyze_results(input_dir, analysis_output_dir):
    """
    Analyzes aggregated simple results JSON files from multiple evaluation runs,
    assuming files are directly in input_dir and named like:
    'MODEL_prompt_STYLE_aggregated_simple_results.json'.

    Args:
        input_dir (str): Directory containing the aggregated JSON result files.
        analysis_output_dir (str): Directory where analysis results (plots, summary CSV)
                                   will be saved.
    """
    print(f"Starting analysis...")
    print(f"Input directory: {input_dir}")
    print(f"Analysis output directory: {analysis_output_dir}")

    # --- Create output directories ---
    plot_output_dir = os.path.join(analysis_output_dir, "plots")
    try:
        os.makedirs(analysis_output_dir, exist_ok=True)
        os.makedirs(plot_output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create output directories: {e}", file=sys.stderr)
        return

    # --- Find and Load Data ---
    # Find files matching the pattern directly in input_dir
    json_files = glob.glob(os.path.join(input_dir, "*_prompt_*_aggregated_simple_results.json"))
    if not json_files:
        print(f"ERROR: No '*_prompt_*_aggregated_simple_results.json' files found directly in '{input_dir}'.", file=sys.stderr)
        print("       Ensure evaluation script ran and saved aggregated results with the correct naming pattern.")
        return

    print(f"Found {len(json_files)} aggregated result files.")

    all_results_dfs = []
    # --- Regex to extract info from FILENAME ---
    # Example filename: hf_mistral7b_verbose_aggregated_simple_results.json
    file_pattern = re.compile(r"^(.*?)_prompt_(verbose|concise|transformer-priority)_aggregated_simple_results\.json$")
    # -------------------------------------------

    for f_path in json_files:
        try:
            filename = os.path.basename(f_path)
            match = file_pattern.match(filename)
            if not match:
                print(f"WARN: Skipping file with unexpected filename format: {filename}")
                continue
            model_name = match.group(1) # Extracted from filename
            prompt_style = match.group(2) # Extracted from filename
            print(f"  Loading: {filename} (Model Group: {model_name}, Prompt: {prompt_style})")

            with open(f_path, 'r') as f:
                data = json.load(f)

            if not data:
                print(f"  WARN: File is empty: {f_path}")
                continue

            df = pd.DataFrame(data)
            df['model_name_group'] = model_name # Use consistent column name
            df['prompt_style'] = prompt_style
            all_results_dfs.append(df)

        except json.JSONDecodeError:
            print(f"WARN: Skipping invalid JSON file: {f_path}", file=sys.stderr)
        except Exception as e:
            print(f"WARN: Error processing file {f_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_results_dfs:
        print("ERROR: No valid data loaded from any result files.", file=sys.stderr)
        return

    # Concatenate all data
    results_df = pd.concat(all_results_dfs, ignore_index=True)
    print(f"\nTotal records loaded across all files: {len(results_df)}")

    # --- Data Cleaning & Preparation ---
    print("--- Cleaning and Preparing Data ---")
    initial_count = len(results_df)

    # 1. Filter out runs with agent errors
    results_df_errors = results_df[results_df['error'].notna()]
    if not results_df_errors.empty:
        print(f"Filtered out {len(results_df_errors)} records with agent errors ('error' column not null).")
        # Optional: Save errors for inspection
        # error_log_path = os.path.join(analysis_output_dir, "agent_errors.csv")
        # results_df_errors.to_csv(error_log_path, index=False)
        # print(f"Agent error details saved to: {error_log_path}")
    results_df = results_df[results_df['error'].isna()].copy()

    # 2. Filter out runs where LLM prediction is null (parsing failed or error during LLM call)
    results_df_no_pred = results_df[results_df['predicted_label'].isna()]
    if not results_df_no_pred.empty:
         print(f"Filtered out {len(results_df_no_pred)} records with null 'predicted_label'.")
    results_df = results_df[results_df['predicted_label'].notna()].copy()

    # 3. Filter out runs where Ground Truth is null (shouldn't happen if input file is good)
    results_df_no_gt = results_df[results_df['ground_truth_label'].isna()]
    if not results_df_no_gt.empty:
         print(f"Filtered out {len(results_df_no_gt)} records with null 'ground_truth_label'.")
    results_df = results_df[results_df['ground_truth_label'].notna()].copy()

    # Convert labels to integers AFTER filtering NaNs
    if not results_df.empty:
        try:
            results_df['ground_truth_label'] = results_df['ground_truth_label'].astype(int)
            results_df['predicted_label'] = results_df['predicted_label'].astype(int)
        except ValueError as e:
            print(f"ERROR: Could not convert labels to integers after filtering NaNs: {e}", file=sys.stderr)
            # Identify problematic rows if needed
            # print(results_df[pd.to_numeric(results_df['ground_truth_label'], errors='coerce').isna()])
            # print(results_df[pd.to_numeric(results_df['predicted_label'], errors='coerce').isna()])
            return

    # 4. Handle confidence scores
    if 'confidence_score' in results_df.columns:
        invalid_confidence = results_df[results_df['confidence_score'].isna() | ~np.isfinite(results_df['confidence_score'])]
        if not invalid_confidence.empty:
            print(f"WARN: Found {len(invalid_confidence)} records with invalid confidence scores. Filling with 0.5.")
            results_df['confidence_score'] = results_df['confidence_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5)
        results_df['confidence_score'] = results_df['confidence_score'].astype(float)
    else:
        print("WARN: 'confidence_score' column not found. Calibration metrics will be skipped.")
        results_df['confidence_score'] = 0.5 # Assign default if missing

    # 5. Calculate Explanation Consistency
    if 'llm_explanation' in results_df.columns:
        print("Calculating explanation consistency...")
        results_df['explanation_consistent'] = results_df.apply(
            lambda row: (check_explanation_conclusion_sentiment(row['llm_explanation']) == row['predicted_label'])
                         if pd.notna(row['predicted_label']) and check_explanation_conclusion_sentiment(row['llm_explanation']) is not None
                         else None, # None if sentiment is ambiguous or prediction missing
            axis=1
        )
        print("Consistency calculation complete.")
    else:
        print("WARN: 'llm_explanation' column not found. Consistency metrics will be skipped.")
        results_df['explanation_consistent'] = None


    final_valid_count = len(results_df)
    print(f"Total valid records for analysis: {final_valid_count} (out of {initial_count} initially loaded)")

    if final_valid_count == 0:
        print("ERROR: No valid records remaining after filtering. Cannot proceed with analysis.", file=sys.stderr)
        return

    # --- Metric Calculation & Plotting ---
    metrics_summary = []
    # Setup combined plots
    fig_roc_all, ax_roc_all = plt.subplots(figsize=(8, 8))
    fig_pr_all, ax_pr_all = plt.subplots(figsize=(8, 8))

    grouped = results_df.groupby(['model_name_group', 'prompt_style'])

    print("\n--- Calculating Metrics & Generating Plots per Group ---")
    for (model_group, style), group_df in grouped:
        group_name = f"{model_group} ({style})"
        print(f"\nAnalyzing Group: {group_name}")
        group_metrics = {'Model Group': model_group, 'Prompt Style': style, 'Samples': len(group_df)}

        y_true = group_df['ground_truth_label'].values
        y_pred = group_df['predicted_label'].values
        y_prob = np.clip(group_df['confidence_score'].values, 1e-15, 1 - 1e-15) # Use confidence as prob

        # Check if valid data for metrics
        if len(y_true) == 0: print("  No valid samples. Skipping metrics."); continue
        unique_true = np.unique(y_true)
        has_both_classes = len(unique_true) == 2

        # --- Basic Classification Metrics ---
        group_metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        group_metrics['F1 (Pos=1)'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        group_metrics['Precision (Pos=1)'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        group_metrics['Recall (Pos=1)'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            group_metrics['Specificity (Pos=0)'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        except ValueError: group_metrics['Specificity (Pos=0)'] = np.nan # Handle case where only one class predicted
        try:
            group_metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        except Exception as e_mcc: # Catch potential math errors if predictions are constant etc.
            print(f"  WARN: Could not calculate MCC for {group_name}: {e_mcc}")
            group_metrics['MCC'] = np.nan

        # --- Ranking / Probability Metrics ---
        if has_both_classes:
            group_metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
            group_metrics['AUC-PR (AvgPrec)'] = average_precision_score(y_true, y_prob, pos_label=1)
        else:
             print(f"  WARN: Only one class ({unique_true}) present in ground truth. AUC/AP metrics are undefined.")
             group_metrics['AUC-ROC'], group_metrics['AUC-PR (AvgPrec)'] = np.nan, np.nan

        # --- Calibration Metrics ---
        group_metrics['Brier Score'] = brier_score_loss(y_true, y_prob, pos_label=1)
        # ECE calculation needs bins, can be added if desired

        # --- Explanation Consistency ---
        if 'explanation_consistent' in group_df.columns:
            consistent_count = group_df['explanation_consistent'].sum()
            valid_checks = group_df['explanation_consistent'].notna().sum()
            group_metrics['Explanation Consistency %'] = (consistent_count / valid_checks * 100) if valid_checks > 0 else np.nan
            group_metrics['Inconsistent Expl Count'] = valid_checks - consistent_count
        else:
            group_metrics['Explanation Consistency %'], group_metrics['Inconsistent Expl Count'] = np.nan, np.nan

        metrics_summary.append(group_metrics)
        print(f"  Metrics calculated.")

        # --- Plotting for this group ---
        safe_group_name = re.sub(r'[^\w\-_\.]', '_', group_name.replace(' ', '_'))
        plot_prefix = os.path.join(plot_output_dir, safe_group_name)

        try:
            # Add curves to combined plots
            if has_both_classes:
                RocCurveDisplay.from_predictions(y_true, y_prob, name=group_name, ax=ax_roc_all, pos_label=1)
                PrecisionRecallDisplay.from_predictions(y_true, y_prob, name=group_name, ax=ax_pr_all, pos_label=1)

            # Individual Calibration Plot
            fig_cal, ax_cal = plt.subplots(figsize=(6, 6))
            CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, name=group_name, ax=ax_cal, strategy='uniform')
            ax_cal.set_title(f'Calibration Curve - {group_name}')
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_calibration_curve.png")
            plt.close(fig_cal)
            print(f"  Calibration plot saved.")

        except Exception as e_plot:
            print(f"  ERROR generating plots for {group_name}: {e_plot}", file=sys.stderr)
            if 'fig_cal' in locals() and plt.fignum_exists(fig_cal.number): plt.close(fig_cal)

    # --- Finalize and Save Combined Plots ---
    try:
        if ax_roc_all.has_data(): # Check if any curves were added
            ax_roc_all.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--') # Add random chance line
            ax_roc_all.set_title('ROC Curves (All Models)')
            ax_roc_all.legend(loc='lower right', fontsize='small')
            plt.figure(fig_roc_all.number) # Set current figure
            plt.tight_layout()
            plt.savefig(os.path.join(plot_output_dir, "combined_roc_curves.png"))
        plt.close(fig_roc_all)

        if ax_pr_all.has_data():
            # You might want to add iso-F1 lines or baseline for PR curve depending on class balance
            ax_pr_all.set_title('Precision-Recall Curves (All Models)')
            ax_pr_all.legend(loc='lower left', fontsize='small')
            plt.figure(fig_pr_all.number) # Set current figure
            plt.tight_layout()
            plt.savefig(os.path.join(plot_output_dir, "combined_pr_curves.png"))
        plt.close(fig_pr_all)
        print(f"Combined ROC and PR plots saved.")
    except Exception as e_comb_plot:
         print(f"ERROR generating combined plots: {e_comb_plot}", file=sys.stderr)
         if 'fig_roc_all' in locals() and plt.fignum_exists(fig_roc_all.number): plt.close(fig_roc_all)
         if 'fig_pr_all' in locals() and plt.fignum_exists(fig_pr_all.number): plt.close(fig_pr_all)

    # --- Generate and Save Summary Table ---
    if not metrics_summary:
        print("\nERROR: No metrics were calculated for any group.", file=sys.stderr)
        return

    summary_df = pd.DataFrame(metrics_summary)
    # Reorder columns for readability
    cols_order = ['Model Group', 'Prompt Style', 'Samples', 'Accuracy', 'F1 (Pos=1)', 'MCC',
                  'Precision (Pos=1)', 'Recall (Pos=1)', 'Specificity (Pos=0)',
                  'AUC-ROC', 'AUC-PR (AvgPrec)', 'Brier Score',
                  'Explanation Consistency %', 'Inconsistent Expl Count']
    # Ensure all expected columns exist before reordering
    summary_df = summary_df.reindex(columns=[col for col in cols_order if col in summary_df.columns])

    summary_df = summary_df.round(4)

    print("\n--- Overall Performance Summary ---")
    print(summary_df.to_string())

    summary_csv_path = os.path.join(analysis_output_dir, "performance_summary.csv")
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary table saved to: {summary_csv_path}")
    except Exception as e:
        print(f"ERROR saving summary CSV: {e}", file=sys.stderr)

    # --- <<< NEW: Call the Bar Plot Function >>> ---
    metrics_for_bar_plot = ['Accuracy', 'F1 (Pos=1)', 'AUC-PR (AvgPrec)', 'MCC'] # Select key metrics
    # Ensure only valid metrics are passed if some were NaN for all models
    valid_metrics_for_plot = [m for m in metrics_for_bar_plot if m in summary_df.columns and summary_df[m].notna().any()]
    if valid_metrics_for_plot:
         plot_comparison_bars(summary_df, valid_metrics_for_plot, analysis_output_dir)
    else:
         print("\nSkipping comparison bar plot as no valid data found for selected metrics.")
    # ------------------------------------------

    print("\nAnalysis script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze DPI-Agent evaluation results from multiple runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="Input directory containing the evaluation run subdirectories "\
             "(e.g., './dpi_agent_eval_outputs'). Each subdir must contain one "\
             "'*_aggregated_simple_results.json' file and follow naming pattern 'MODEL_prompt_STYLE'."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./analysis_results",
        help="Directory where analysis plots and summary CSV will be saved."
    )
    args = parser.parse_args()

    analyze_results(args.input_dir, args.output_dir)