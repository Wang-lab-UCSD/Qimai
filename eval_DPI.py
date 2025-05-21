#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import json
import os
import glob
import traceback
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Transformer model using ground truth labels and prediction probabilities from a folder of JSON files (one sample per file).")

    parser.add_argument("--input_data_folder", type=str, required=True,
                        help="Path to the folder containing input JSON files. Each file should represent one sample.")
    parser.add_argument("--label_key", type=str, default="ground_truth_label",
                        help="JSON key in each file for the true label.")
    parser.add_argument("--probability_key", type=str, default="transformer_prob",
                        help="JSON key in each file for the transformer's prediction probability.")
    parser.add_argument("--prediction_threshold", type=float, default=0.5,
                        help="Threshold to convert probabilities to binary predictions (0 or 1).")
    
    parser.add_argument("--model_name", type=str, default="DPI",
                        help="Name of the model, used for output filenames and plot titles.")
    
    parser.add_argument("--output_dir", type=str, default="./transformer_evaluation_results",
                        help="Directory to save evaluation results (metrics CSV and plot).")
    parser.add_argument("--metrics_csv_filename", type=str, default="transformer_metrics_summary.csv",
                        help="Filename for the output metrics CSV file.")
    parser.add_argument("--plot_filename", type=str, default="transformer_performance_plot.png",
                        help="Filename for the output performance plot.")

    parser.add_argument("--max_files_to_process", type=int, default=None,
                        help="Maximum number of JSON files to process from the folder. If not set, all files are processed. Files are chosen randomly if this limit is applied.")
    parser.add_argument("--random_seed_for_file_selection", type=int, default=13,
                        help="Random seed for selecting files if --max_files_to_process is set.")
    
    return parser.parse_args()

def load_and_prepare_data_from_folder(data_folder_path, label_key, prob_key, 
                                      max_files, random_seed):
    print(f"Loading data from folder: {data_folder_path}")
    
    json_files = glob.glob(os.path.join(data_folder_path, "*comp.json"))
    if not json_files:
        print(f"No JSON files found in {data_folder_path}.")
        return None, 0, 0

    print(f"Found {len(json_files)} JSON files.")

    if max_files is not None and len(json_files) > max_files:
        print(f"Randomly selecting {max_files} files to process (seed: {random_seed})...")
        random.seed(random_seed)
        selected_files = random.sample(json_files, max_files)
    else:
        selected_files = json_files
        if max_files is not None:
             print(f"Processing all {len(json_files)} found files (max_files_to_process >= found files).")
        else:
             print(f"Processing all {len(json_files)} found files.")


    labels_list = []
    probabilities_list = []
    successfully_processed_files = 0
    skipped_files_count = 0

    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            label = data.get(label_key)
            probability = data.get(prob_key)

            # Validate label
            if label is None:
                # print(f"Warning: Missing label_key '{label_key}' in {file_path}. Skipping file.")
                skipped_files_count += 1
                continue
            try:
                label = int(float(label)) # Allow float like "0.0" then convert to int
                if label not in [0, 1]:
                    # print(f"Warning: Invalid label value '{label}' (must be 0 or 1) in {file_path}. Skipping file.")
                    skipped_files_count += 1
                    continue
            except (ValueError, TypeError):
                # print(f"Warning: Non-numeric label '{label}' in {file_path}. Skipping file.")
                skipped_files_count += 1
                continue
            
            # Validate probability (allow None, handle non-numeric as None for now)
            if probability is not None:
                try:
                    probability = float(probability)
                except (ValueError, TypeError):
                    # print(f"Warning: Non-numeric probability '{probability}' in {file_path}. Treating as missing (None).")
                    probability = None 
            
            labels_list.append(label)
            probabilities_list.append(probability)
            successfully_processed_files += 1

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Skipping file.")
            skipped_files_count += 1
        except Exception as e:
            print(f"Warning: Error processing file {file_path}: {e}. Skipping file.")
            traceback.print_exc()
            skipped_files_count += 1
            
    print(f"Successfully processed data from {successfully_processed_files} files.")
    if skipped_files_count > 0:
        print(f"Skipped {skipped_files_count} files due to missing keys, invalid data, or read errors.")

    if not labels_list:
        print("No valid data could be extracted from the files.")
        return None, 0, skipped_files_count

    # Create DataFrame using the actual key names for columns
    df = pd.DataFrame({
        label_key: labels_list,
        prob_key: probabilities_list
    })
    
    # Note: Label validation already done per file. Probability NaN/type handling done in evaluate_transformer_from_dataframe.
    return df, len(df), skipped_files_count


def evaluate_transformer_from_dataframe(eval_df, model_name_desc, label_col_name, prob_col_name, threshold):
    """
    Calculates metrics for Transformer predictions based on specified columns in the DataFrame.
    label_col_name and prob_col_name are the actual names of the columns in eval_df.
    """
    print(f"\n--- Evaluating: {model_name_desc} (from DataFrame probabilities) ---")
    num_expected_samples_in_df = len(eval_df)
    metrics = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "MCC": 0.0, 
               "samples_eval": 0, "parsing_failures": num_expected_samples_in_df}

    if num_expected_samples_in_df == 0:
        print("DataFrame for evaluation is empty.")
        return metrics

    true_labels_list = eval_df[label_col_name].astype(int).tolist()

    df_processed = eval_df[[prob_col_name]].copy() # Work with a copy

    initial_nan_prob = df_processed[prob_col_name].isna().sum()
    if initial_nan_prob > 0:
        print(f"Found {initial_nan_prob} missing transformer probabilities in the evaluation set. Treating as prediction=0 (prob=0.0).")
        df_processed[prob_col_name].fillna(0.0, inplace=True)

    try:
        df_processed[prob_col_name] = pd.to_numeric(df_processed[prob_col_name], errors='coerce')
        coercion_nan_prob = df_processed[prob_col_name].isna().sum()
        if coercion_nan_prob > 0: # These were non-numeric that weren't None initially
            print(f"Found {coercion_nan_prob} non-numeric probabilities after coercion (originally not NaN). Treating as prediction=0 (prob=0.0).")
            df_processed[prob_col_name].fillna(0.0, inplace=True)

        predicted_labels = (df_processed[prob_col_name] >= threshold).astype(int).tolist()
    except Exception as e:
        print(f"Error converting transformer probabilities to labels: {e}")
        traceback.print_exc()
        return metrics

    if len(predicted_labels) == num_expected_samples_in_df:
        accuracy = accuracy_score(true_labels_list, predicted_labels)
        precision = precision_score(true_labels_list, predicted_labels, zero_division=0)
        recall = recall_score(true_labels_list, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels_list, predicted_labels, zero_division=0)
        mcc = matthews_corrcoef(true_labels_list, predicted_labels)
        
        metrics = {
            "Accuracy": float(accuracy), "Precision": float(precision), 
            "Recall": float(recall), "F1": float(f1), "MCC": float(mcc), 
            "samples_eval": len(predicted_labels), "parsing_failures": 0
        }
        print(f"Evaluated on: {len(predicted_labels)} samples")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1 Score: {metrics['F1']:.4f}")
        print(f"  MCC: {metrics['MCC']:.4f}")
    else:
        print(f"Error: Number of generated Transformer predictions ({len(predicted_labels)}) doesn't match expected ({num_expected_samples_in_df}).")
    return metrics

def save_metrics_to_csv(metrics_dict, model_name, output_csv_path, skipped_file_count_at_load):
    summary_data = {
        "Model": [model_name],
        "files_skipped_at_load": [skipped_file_count_at_load], # New info
        "samples_eval": [metrics_dict.get("samples_eval", 0)],
        "parsing_failures_in_eval_df": [metrics_dict.get("parsing_failures", 0)], # Failures within the successfully loaded data
        "Accuracy": [metrics_dict.get("Accuracy", 0.0)],
        "F1": [metrics_dict.get("F1", 0.0)],
        "MCC": [metrics_dict.get("MCC", 0.0)],
        "Precision": [metrics_dict.get("Precision", 0.0)],
        "Recall": [metrics_dict.get("Recall", 0.0)]

    }
    summary_df = pd.DataFrame(summary_data)
    
    try:
        summary_df.to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"\nMetrics summary saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving metrics summary CSV: {e}")

def generate_metrics_plot(metrics_dict, model_name, plot_path, threshold_used):
    print("\n--- Generating Performance Plot ---")
    
    plot_metric_keys = ["Accuracy", "F1", "MCC", "Precision", "Recall"]
    metric_values = [metrics_dict.get(k, 0.0) for k in plot_metric_keys]

    if metrics_dict.get("samples_eval", 0) == 0 : 
        print("No samples evaluated. Skipping plot generation.")
        return

    df_plot = pd.DataFrame({
        "Metric": plot_metric_keys,
        "Score": metric_values,
        "Model": model_name 
    })

    try:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df_plot, x="Metric", y="Score", color="steelblue")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)
            
        plt.title(f'{model_name} Performance (Threshold: {threshold_used})', fontsize=16)
        plt.xlabel('Metric', fontsize=12); plt.ylabel('Score', fontsize=12)
        
        max_score = df_plot['Score'].max()
        y_upper_limit = max(1.05, (max_score * 1.1) if not math.isnan(max_score) and max_score > 0 else 1.05)
        plt.ylim(0, y_upper_limit)
        
        plt.xticks(rotation=0, ha='center', fontsize=11); plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error generating or saving plot: {e}"); traceback.print_exc()

if __name__ == "__main__":
    args = parse_args()

    print("Starting Transformer Evaluation Script (from folder of JSONs)")
    print(f"Input Data Folder: {args.input_data_folder}")
    print(f"Label JSON Key: {args.label_key}")
    print(f"Probability JSON Key: {args.probability_key}")
    print(f"Prediction Threshold: {args.prediction_threshold}")
    print(f"Model Name: {args.model_name}")
    print(f"Output Directory: {args.output_dir}")
    if args.max_files_to_process:
        print(f"Max Files to Process: {args.max_files_to_process}, File Selection Seed: {args.random_seed_for_file_selection}")
    else:
        print("Processing all found JSON files.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data from folder, creating a DataFrame
    eval_df, num_df_rows, skipped_file_count = load_and_prepare_data_from_folder(
        args.input_data_folder, 
        args.label_key, 
        args.probability_key,
        args.max_files_to_process, 
        args.random_seed_for_file_selection
    )

    if eval_df is None or num_df_rows == 0:
        print("Exiting: No data loaded into DataFrame or DataFrame is empty.")
        # Optionally, still save a metrics CSV indicating 0 samples evaluated
        empty_metrics = {"samples_eval": 0, "parsing_failures": 0} # parsing_failures here would be 0 as no df rows to parse
        metrics_csv_path = os.path.join(args.output_dir, args.metrics_csv_filename)
        save_metrics_to_csv(empty_metrics, args.model_name, metrics_csv_path, skipped_file_count)
        exit(1)

    # Evaluate using the DataFrame; pass the actual column names from args
    transformer_metrics = evaluate_transformer_from_dataframe(
        eval_df,
        args.model_name,
        args.label_key,       # Pass the actual column name used in the DataFrame
        args.probability_key, # Pass the actual column name used in the DataFrame
        args.prediction_threshold
    )

    metrics_csv_path = os.path.join(args.output_dir, args.metrics_csv_filename)
    save_metrics_to_csv(transformer_metrics, args.model_name, metrics_csv_path, skipped_file_count)

    plot_path = os.path.join(args.output_dir, args.plot_filename)
    generate_metrics_plot(transformer_metrics, args.model_name, plot_path, args.prediction_threshold)
    
    print("\nEvaluation script finished.")