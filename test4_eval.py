# --- batch evluation version of test4.py ---
# prepare json files for llm model performance benchmark
# --- Core Python & LangGraph ---
import time
import json
import operator
import os
import random
import re
import time
import sys
import pickle
import traceback
import duckdb
import gc
import math
import subprocess # For running external tools like FIMO
import tempfile   # For creating temporary files for FIMO
import argparse   # For command-line arguments
from functools import partial # To pass loaded resources to nodes
from typing import TypedDict, Annotated, List, Union, Dict, Tuple
from collections import OrderedDict # For transformer state_dict
import warnings

# --- Unsloth (IMPORT FIRST if using) ---
try:
    import unsloth # <<<--- IMPORT UNSLOTH FIRST
    UNSLOTH_AVAILABLE = True
    print("INFO: Unsloth library found and imported first.")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("INFO: Unsloth library not found.")

# --- LangGraph & LLM Backends ---
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import requests # For Ollama API

# --- Dependencies from Hybrid Script & HF Models ---
import torch
import torch.cuda
import numpy as np
import pandas as pd
from Bio.Seq import Seq

# --- Hugging Face Transformers ---
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, BitsAndBytesConfig, BertModel, BertConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    if not UNSLOTH_AVAILABLE: print("WARN: `transformers` libraries not found. HF models unavailable.")


# --- Import Model Components (Make sure these are accessible) ---
try:
    from dpi_v0.main_singletask import Predictor
    from dpi_v0.model_eval_single_v2 import unify_dim_embedding as eval_unify_dim_embedding
    from dpi_v0.model_eval_single_v2 import get_config as eval_get_config
    print("Successfully imported Predictor and helper functions.")
except ImportError as e:
    print(f"ERROR: Could not import required components from source scripts: {e}", file=sys.stderr)
    print("Ensure main_singletask.py and model_eval_single_v2.py are in the Python path.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
     print(f"ERROR during import: {e}", file=sys.stderr); sys.exit(1)


# --- Configuration & Constants ---
# --- API Key Setup ---
GOOGLE_API_KEY_CONFIGURED = False
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        GOOGLE_API_KEY_CONFIGURED = True
        print("Google API Key configured.")
    else:
        print("Google API Key environment variable (GOOGLE_API_KEY) not set. Google models will not be available.")
except Exception as e: print(f"ERROR configuring Google Generative AI: {e}", file=sys.stderr)

# --- Constants ---
DEFAULT_GEMINI_MODEL = 'gemini-1.5-flash-latest'
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_HF_MODEL = "unsloth/mistral-7b-bnb-4bit" # Example default HF model

# --- DNABERT Configuration ---
DNABERT_KMER = 6 # Set the k-mer value used by your DNABERT model
DNABERT_MAX_LEN = 512 # Max length for DNABERT tokenization

FINETUNED_LOCAL_MODEL_PATH_ID = "/new-stg/home/cong/DPI/scripts/deepseek/stage2_rat_70b_2500_steps"

# --- Domain-Specific Constants ---
DEFAULT_P_VALUE_THRESHOLD = 1e-4
TARGET_SPECIES = "Homo_sapiens"
MAX_MOTIFS_PER_TF_TO_PROCESS = 3
STRING_API_URL = "https://string-db.org/api/"
STRING_OUTPUT_FORMAT = "json"
STRING_MIN_INTERACTION_SCORE = 700
MAX_STRING_INTERACTORS_TO_CHECK = 50
TARGET_SPECIES_TAXONOMY_ID = 9606
MAX_INTERACTING_TFS_TO_ANALYZE = 5

# Confidence Score Parameters (TUNABLE)
CONFIDENCE_BASELINE = 0.50
DIRECT_MATCH_BOOST = 0.25         # Boost if direct hit matches LLM vote=1
DIRECT_RELIABLE_MATCH_EXTRA = 0.10 # Extra boost if reliable direct hit matches LLM vote=1
DIRECT_MISMATCH_PENALTY = -0.30   # Penalty if direct hit contradicts LLM vote=0
DIRECT_RELIABLE_MISMATCH_EXTRA = -0.15 # Extra penalty if reliable direct hit contradicts LLM vote=0
DIRECT_MISSING_PENALTY = -0.15    # Penalty if LLM vote=1 but no direct hit found (and search was ok)
DIRECT_ABSENCE_BOOST = 0.10       # Boost if LLM vote=0 and no direct hit found (consistent)

INDIRECT_MATCH_BOOST = 0.10       # Smaller boosts/penalties for indirect
INDIRECT_MISMATCH_PENALTY = -0.15
INDIRECT_MISSING_PENALTY = -0.05
INDIRECT_ABSENCE_BOOST = 0.05

GENERAL_NUM_HITS_BONUS_MAX = 0.10          # Max bonus from number of hits
GENERAL_NUM_HITS_SCALE_FACTOR = 0.02       # Bonus per hit, e.g., 0.02 * min(num_hits, 5)
GENERAL_MAX_HITS_FOR_BONUS_CAP = 5         # Cap for bonus scaling
GENERAL_LOW_PVALUE_THRESHOLD = 1e-7        # P-value below this gets a bonus
GENERAL_LOW_PVALUE_BONUS = 0.07            # Bonus for a significant p-value

TRANSFORMER_MATCH_BOOST = 0.15    # Boost if transformer prob aligns with LLM vote
TRANSFORMER_CONFIDENT_EXTRA = 0.10 # Extra if transformer is confident (e.g. <0.1 or >0.9)
TRANSFORMER_CONFIDENT_HIGH = 0.8
TRANSFORMER_CONFIDENT_LOW = 0.2
TRANSFORMER_MISMATCH_PENALTY = -0.20
TRANSFORMER_CONFIDENT_MISMATCH_EXTRA = -0.10
TRANSFORMER_NOVELTY_FACTOR = 0.6 # Factor to reduce transformer impact for novel proteins

SCAN_FAIL_FACTOR = 0.3 # Factor to reduce motif impact if scan failed

# --- Confidence Score Parameters (FOR TRANSFORMER-PRIORITY PROMPT) ---
# These are STARTING POINTS and will likely need tuning.
TP_CONFIDENCE_BASELINE = 0.45  # Start a bit lower, as alignment will boost significantly

# Transformer Evidence (given higher importance)
TP_DIRECT_MATCH_BOOST = 0.10         # Motif agreement is a smaller, secondary boost
TP_DIRECT_RELIABLE_MATCH_EXTRA = 0.05
TP_DIRECT_MISMATCH_PENALTY = -0.20   # Penalty if direct hit contradicts (LLM+Transformer)
TP_DIRECT_RELIABLE_MISMATCH_EXTRA = -0.05
TP_DIRECT_MISSING_PENALTY = -0.10    # Penalty if (LLM+Transformer)=1 but no direct hit
TP_DIRECT_ABSENCE_BOOST = 0.05       # Boost if (LLM+Transformer)=0 and no direct hit (consistent)

TP_INDIRECT_MATCH_BOOST = 0.10       
TP_INDIRECT_MISMATCH_PENALTY = -0.15
TP_INDIRECT_MISSING_PENALTY = -0.05
TP_INDIRECT_ABSENCE_BOOST = 0.05

# Transformer Evidence (Primary Driver when LLM aligns with it)
TP_TRANSFORMER_MATCH_BOOST = 0.30    # SIGNIFICANT boost if LLM vote aligns with transformer
TP_TRANSFORMER_CONFIDENT_EXTRA = 0.10 # Extra if transformer is confident AND aligns

# What if LLM DISAGREES with transformer, despite instruction? This should be penalized.
TP_TRANSFORMER_MISMATCH_PENALTY = -0.25 
TP_TRANSFORMER_CONFIDENT_MISMATCH_EXTRA = -0.15
TP_TRANSFORMER_NOVELTY_FACTOR = 0.5 

# --- Confidence Score Parameters (FOR MOTIF-PRIORITY PROMPT) ---
# These are STARTING POINTS and will likely need tuning.
MP_CONFIDENCE_BASELINE = 0.50  # Start similar to default, motif evidence will drive it

# Motif Evidence (Primary Driver)
MP_DIRECT_MATCH_BOOST = 0.30         # Significantly higher boost for direct motif match
MP_DIRECT_RELIABLE_MATCH_EXTRA = 0.10
MP_DIRECT_MISMATCH_PENALTY = -0.35   # Stronger penalty if direct hit contradicts LLM vote=0
MP_DIRECT_RELIABLE_MISMATCH_EXTRA = -0.15
MP_DIRECT_MISSING_PENALTY = -0.20    # Penalty if LLM vote=1 but no direct hit
MP_DIRECT_ABSENCE_BOOST = 0.15       # Boost if LLM vote=0 and no direct hit

MP_INDIRECT_MATCH_BOOST = 0.20       # Higher boost for indirect as well
MP_INDIRECT_MISMATCH_PENALTY = -0.25
MP_INDIRECT_MISSING_PENALTY = -0.10
MP_INDIRECT_ABSENCE_BOOST = 0.10

# New: Motif Quality Factors
MP_NUM_HITS_BONUS_MAX = 0.15          # Max bonus achievable from number of hits
MP_NUM_HITS_SCALE_FACTOR = 0.03       # Bonus per hit, e.g., 0.03 * min(num_hits, 5)
MP_MAX_HITS_FOR_BONUS_CAP = 5         # Consider up to this many hits for bonus scaling
MP_LOW_PVALUE_THRESHOLD = 1e-8        # P-value below this threshold gets a bonus
MP_LOW_PVALUE_BONUS = 0.10            # Bonus for a very significant p-value

# Transformer Evidence (Secondary, especially if conflicting with strong motifs)
MP_TRANSFORMER_MATCH_BOOST = 0.10     # Smaller boost if transformer aligns
MP_TRANSFORMER_CONFIDENT_EXTRA = 0.05
MP_TRANSFORMER_MISMATCH_PENALTY = -0.15 # Smaller penalty for mismatch
MP_TRANSFORMER_CONFIDENT_MISMATCH_EXTRA = -0.05
MP_TRANSFORMER_NOVELTY_FACTOR = 0.7   # Transformer is less discounted for novel if motifs are primary


# --- Agent State Definition (Unchanged from test3.py) ---
class AgentState(TypedDict):
    protein_name: str; dna_sequence: str
    ground_truth_label: Union[int, None]
    direct_motif_fetch_results: Union[dict, None]; direct_motif_scan_results: Union[dict, None]
    direct_motif_analysis_summary: Union[dict, None]; string_interactors: Union[List[dict], None]
    tf_interactors_cisbp: Union[List[dict], None]; tfs_analyzed_indirectly: Union[List[dict], None]
    indirect_motif_fetch_results: Union[dict, None]; indirect_motif_scan_results: Union[dict, None]
    indirect_motif_analysis_summary: Union[dict, None]; transformer_prob: Union[float, None]
    is_known_protein: bool; llm_prompt: Union[str, None]; llm_vote: Union[int, None]
    llm_explanation: Union[str, None]
    raw_llm_response: Union[str, None]  
    final_confidence: Union[float, None] 
    error: Union[str, None]

def _parse_cisbp_pwm(pwm_filepath: str) -> Union[Dict[str, List[float]], None]:
    # ... (Implementation unchanged) ...
    if not os.path.exists(pwm_filepath): return None
    counts = {'A': [], 'C': [], 'G': [], 'T': []}; background = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
    try:
        with open(pwm_filepath, 'r') as f: lines = f.readlines()
        if not lines: raise ValueError("PWM file is empty")
        data_lines = [line for line in lines if not line.strip().startswith('>') and not line.strip().startswith('Pos')]
        if not data_lines:
            header_line = lines[0].strip()
            if header_line.upper().startswith("POS"): data_lines = lines[1:]
            else: data_lines = lines
            if not data_lines: raise ValueError("No data lines found after skipping headers")
        if len(data_lines[0].split()) == 4: # Format: A C G T
            for line in data_lines:
                freqs = list(map(float, line.split()))
                if len(freqs) != 4: raise ValueError(f"Expected 4 values per line, got {len(freqs)}")
                counts['A'].append(freqs[0]); counts['C'].append(freqs[1]); counts['G'].append(freqs[2]); counts['T'].append(freqs[3])
        elif len(data_lines[0].split()) > 4: # Format: Pos A C G T
             for line in data_lines:
                 parts = line.split();
                 if len(parts) < 5: continue
                 freqs = list(map(float, parts[1:5]))
                 counts['A'].append(freqs[0]); counts['C'].append(freqs[1]); counts['G'].append(freqs[2]); counts['T'].append(freqs[3])
        else: raise ValueError(f"Unrecognized PWM format. First data line: {data_lines[0].strip()}")
        if not all(len(lst) == len(counts['A']) for lst in counts.values()) or len(counts['A']) == 0: raise ValueError("Inconsistent lengths or zero length motif")
        for i in range(len(counts['A'])):
            pos_sum = counts['A'][i] + counts['C'][i] + counts['G'][i] + counts['T'][i]
            if abs(pos_sum - 1.0) > 1e-3 and pos_sum > 1e-6:
                if pos_sum == 0: continue
                counts['A'][i] /= pos_sum; counts['C'][i] /= pos_sum; counts['G'][i] /= pos_sum; counts['T'][i] /= pos_sum
            elif abs(pos_sum - 1.0) <= 1e-3: pass
            elif pos_sum == 0: counts['A'][i]=0.25; counts['C'][i]=0.25; counts['G'][i]=0.25; counts['T'][i]=0.25;
        return counts
    except ValueError as ve: print(f"  ERROR parsing PWM {os.path.basename(pwm_filepath)}: {ve}", file=sys.stderr); return None
    except Exception as e: print(f"  ERROR reading/parsing PWM {os.path.basename(pwm_filepath)}: {e}", file=sys.stderr); return None

def _calculate_cisbp_motif_reliability_for_prompt(motifs_metadata: List[Dict]) -> str:
    # ... (Implementation unchanged) ...
    if not motifs_metadata: return "Not Found"
    best_reliability_category = "Unknown"; has_direct = False; has_inferred = False; has_direct_high_source = False; has_inferred_high_source = False
    HIGH_CONFIDENCE_SOURCES_FOR_PROMPT = ["JASPAR", "HOCOMOCO"]
    for meta in motifs_metadata:
        status = meta.get('TF_Status'); source = meta.get('MSource_Identifier', '')
        is_high_conf_source = any(hs.lower() in source.lower() for hs in HIGH_CONFIDENCE_SOURCES_FOR_PROMPT if hs)
        if status == 'D': has_direct = True;
        if is_high_conf_source: has_direct_high_source = True
        elif status == 'I': has_inferred = True;
        if is_high_conf_source: has_inferred_high_source = True
    if has_direct_high_source: best_reliability_category = "Direct_HighConfidenceSource"
    elif has_direct: best_reliability_category = "Direct_Standard"
    elif has_inferred_high_source: best_reliability_category = "Inferred_HighConfidenceSource"
    elif has_inferred: best_reliability_category = "Inferred_Standard"
    CISBP_RELIABILITY_DISPLAY_FOR_PROMPT = {"Direct_HighConfidenceSource": "High (Direct, Reputable Source)", "Direct_Standard": "High (Direct)", "Inferred_HighConfidenceSource": "Moderate (Inferred, Good Source)", "Inferred_Standard": "Moderate (Inferred)", "Unknown": "Unknown", "Not_Found": "Not Found"}
    return CISBP_RELIABILITY_DISPLAY_FOR_PROMPT.get(best_reliability_category, "Unknown")

def _calculate_motif_priority(row: pd.Series) -> int:
    # ... (Implementation unchanged) ...
    HIGH_CONFIDENCE_SOURCES_FOR_PRIORITY = ["JASPAR", "HOCOMOCO"]
    status = row.get('TF_Status'); source = row.get('MSource_Identifier', '')
    is_high_conf_source = any(hs.lower() in source.lower() for hs in HIGH_CONFIDENCE_SOURCES_FOR_PRIORITY if hs)
    if status == 'D': return 1 if is_high_conf_source else 2
    elif status == 'I': return 3 if is_high_conf_source else 4
    else: return 5

def convert_to_meme_format(motifs_with_metadata: List[tuple]) -> str:
    # ... (Implementation unchanged) ...
    meme_lines = ["MEME version 5", "\nALPHABET= ACGT", "\nstrands: + -", "\n"]
    background = "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n"; meme_lines.append(background)
    for motif_data, metadata in motifs_with_metadata:
        motif_id = metadata.get('Motif_ID', 'UnknownMotif')
        if not isinstance(motif_data, dict) or not all(k in motif_data for k in ['A', 'C', 'G', 'T']): print(f"  WARN: Skipping invalid motif data for {motif_id} in MEME conversion.", file=sys.stderr); continue
        counts = motif_data; length = len(counts['A'])
        if length == 0: print(f"  WARN: Skipping zero-length motif {motif_id} in MEME conversion.", file=sys.stderr); continue
        meme_lines.append(f"MOTIF {motif_id} {metadata.get('TF_Name', '')}\n")
        meme_lines.append(f"letter-probability matrix: alength= 4 w= {length} nsites= 20 E= 0\n")
        for i in range(length):
            a, c, g, t = counts['A'][i], counts['C'][i], counts['G'][i], counts['T'][i]
            total = a + c + g + t
            if abs(total - 1.0) > 1e-3 and total > 1e-6: a /= total; c /= total; g /= total; t /= total
            elif total <= 1e-6: a=0.25; c=0.25; g=0.25; t=0.25;
            meme_lines.append(f" {a:>8.6f} {c:>8.6f} {g:>8.6f} {t:>8.6f}\n")
        meme_lines.append("\n")
    return "".join(meme_lines)

# --- Implemented Tool Functions ---
# ... (Keep implementations of run_cisbp_fetch, run_motif_scan, run_string_query, run_transformer_prediction exactly as in test3.py) ...
def run_cisbp_fetch(protein_name: str, cisbp_tf_info_df: pd.DataFrame, cisbp_pwm_dir: str) -> Dict:
    print(f"\n--- TOOL: Fetching CisBP motifs for '{protein_name}' ---")
    results = {'motifs_metadata': [], 'status': 'not_found_cisbp', 'query_name': protein_name, 'num_total_found': 0, 'num_selected': 0, 'num_direct_selected': 0, 'num_inferred_selected': 0, 'sources_selected': set(), 'filtering_applied': False}
    if cisbp_tf_info_df is None or cisbp_tf_info_df.empty or not cisbp_pwm_dir:
        results['status'] = 'error_cisbp_data_not_loaded_or_empty'; print(f"  WARN: CisBP DataFrame empty or PWM dir missing. Cannot fetch for {protein_name}."); results['sources_selected'] = list(results['sources_selected']); return results
    try:
        protein_matches = cisbp_tf_info_df[cisbp_tf_info_df['TF_Name'] == protein_name].copy(); results['num_total_found'] = len(protein_matches)
        if protein_matches.empty: print(f"  No entries found for TF_Name '{protein_name}' in loaded CisBP data."); results['sources_selected'] = list(results['sources_selected']); return results
        selected_matches_df = protein_matches
        if results['num_total_found'] > MAX_MOTIFS_PER_TF_TO_PROCESS:
            print(f"  Applying filtering: Selecting top {MAX_MOTIFS_PER_TF_TO_PROCESS} motifs based on quality."); results['filtering_applied'] = True
            protein_matches['priority'] = protein_matches.apply(_calculate_motif_priority, axis=1); selected_matches_df = protein_matches.nsmallest(MAX_MOTIFS_PER_TF_TO_PROCESS, 'priority', keep='first'); print(f"  Selected {len(selected_matches_df)} motifs after filtering.")
        results['num_selected'] = len(selected_matches_df); loaded_motifs_count = 0; processed_motif_ids = set()
        for index, row in selected_matches_df.iterrows():
            motif_id = row['Motif_ID'];
            if motif_id in processed_motif_ids: continue
            pwm_filepath = os.path.join(cisbp_pwm_dir, f"{motif_id}.txt"); freq_counts_dict = _parse_cisbp_pwm(pwm_filepath)
            if freq_counts_dict:
                try:
                    motif_data = freq_counts_dict; metadata = {k: row.get(k) for k in ['Motif_ID', 'TF_Name', 'TF_Status', 'Family_Name', 'MSource_Identifier', 'Motif_Type', 'MSource_Author', 'MSource_Year', 'PMID']}; metadata['priority_score'] = row.get('priority', None)
                    results['motifs_metadata'].append((motif_data, metadata)); results['sources_selected'].add(metadata['MSource_Identifier'])
                    if metadata['TF_Status'] == 'D': results['num_direct_selected'] += 1
                    elif metadata['TF_Status'] == 'I': results['num_inferred_selected'] += 1
                    loaded_motifs_count += 1; processed_motif_ids.add(motif_id)
                except Exception as e_create: print(f"  ERROR processing data for {motif_id}: {e_create}", file=sys.stderr)
        if loaded_motifs_count > 0: results['status'] = 'success_cisbp'
        elif results['num_selected'] > 0: results['status'] = 'error_cisbp_pwm_loading'
        else: results['status'] = 'error_cisbp_filtering'
        source_str = ", ".join(list(results['sources_selected'])[:5]) + ('...' if len(results['sources_selected']) > 5 else ''); filter_msg = f"(Filtered from {results['num_total_found']})" if results['filtering_applied'] else ""
        print(f"  Successfully loaded {loaded_motifs_count} motif data structures {filter_msg}. Status: {results['status']}. Sources: {source_str}")
    except KeyError as ke: print(f"ERROR: Missing expected column during CisBP fetch: {ke}", file=sys.stderr); results['status'] = 'error_cisbp_processing'
    except Exception as e: print(f"ERROR during CisBP motif fetching for '{protein_name}': {e}", file=sys.stderr); results['status'] = 'error_cisbp_processing'; traceback.print_exc()
    results['sources_selected'] = list(results['sources_selected']); return results

def run_motif_scan(dna_sequence: str, motifs_with_metadata: List[tuple], pvalue_threshold: float) -> Dict:
    print(f"\n--- TOOL: Scanning DNA for {len(motifs_with_metadata)} motifs via FIMO ---")
    if not motifs_with_metadata: return {'status': 'success', 'hits': []}
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as dna_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.meme', delete=False) as motif_file, \
             tempfile.TemporaryDirectory() as output_dir:
            dna_filepath = dna_file.name; motif_filepath = motif_file.name; fimo_output_dir = output_dir
            dna_file.write(f">input_dna\n{dna_sequence}\n"); dna_file.flush()
            meme_content = convert_to_meme_format(motifs_with_metadata)
            if not meme_content or "MOTIF" not in meme_content: print("  WARN: No valid motifs to write to MEME file. Skipping FIMO.", file=sys.stderr); return {'status': 'error_no_valid_motifs_for_fimo', 'hits': []}
            motif_file.write(meme_content); motif_file.flush()
            fimo_command = ['fimo', '--oc', fimo_output_dir, '--verbosity', '1', '--thresh', str(pvalue_threshold), motif_filepath, dna_filepath]; print(f"  Running FIMO command: {' '.join(fimo_command)}")
            try: result = subprocess.run(fimo_command, capture_output=True, text=True, check=True, timeout=300)
            except FileNotFoundError: print("ERROR: 'fimo' command not found. Ensure MEME Suite is installed and in PATH.", file=sys.stderr); return {'status': 'error_fimo_not_found', 'hits': []}
            except subprocess.TimeoutExpired: print(f"ERROR: FIMO command timed out after 300 seconds.", file=sys.stderr); return {'status': 'error_fimo_timeout', 'hits': []}
            except subprocess.CalledProcessError as e: print(f"ERROR: FIMO execution failed with return code {e.returncode}.", file=sys.stderr); print(f"  FIMO stderr:\n{e.stderr}"); return {'status': 'error_fimo_execution', 'message': e.stderr, 'hits': []}
            fimo_output_path = os.path.join(fimo_output_dir, 'fimo.tsv'); parsed_hits = []; fimo_status = 'success'; fimo_message = None
            if os.path.exists(fimo_output_path):
                try:
                    if os.path.getsize(fimo_output_path) < 50: print("  FIMO output file is very small, likely contains no hits.")
                    else:
                        fimo_df = pd.read_csv(fimo_output_path, sep='\t', comment='#', on_bad_lines='warn')
                        if fimo_df.empty: print("  FIMO output file contained no data rows after parsing comments.")
                        else:
                            metadata_map = {meta['Motif_ID']: meta for _, meta in motifs_with_metadata if meta.get('Motif_ID')}
                            for _, row in fimo_df.iterrows():
                                required_fimo_cols = ['motif_id', 'start', 'score', 'p-value', 'strand', 'matched_sequence']
                                if not all(col in row for col in required_fimo_cols): print(f"  WARN: Skipping FIMO row due to missing columns: {row.to_dict()}", file=sys.stderr); continue
                                motif_id = row['motif_id']; meta = metadata_map.get(motif_id, {})
                                parsed_hits.append({'motif_id': motif_id, 'motif_name': meta.get('TF_Name', row.get('motif_alt_id')), 'position': row['start'] - 1, 'score': row['score'], 'pvalue': row['p-value'], 'qvalue': row.get('q-value'), 'pvalue_threshold': pvalue_threshold, 'strand': row['strand'], 'sequence': row['matched_sequence'], 'tf_status': meta.get('TF_Status'), 'msource_id': meta.get('MSource_Identifier'), 'interacting_tf': meta.get('interacting_tf_name'), 'interacting_tf_string_score': meta.get('interacting_tf_string_score')})
                            parsed_hits.sort(key=lambda x: x['pvalue']); print(f"  Parsed {len(parsed_hits)} significant hits from FIMO output.")
                except pd.errors.EmptyDataError: print("  FIMO output file was empty or unparsable (EmptyDataError)."); fimo_status = 'success_empty_output'
                except Exception as e_parse: print(f"ERROR parsing FIMO output '{fimo_output_path}': {e_parse}", file=sys.stderr); traceback.print_exc(); fimo_status = 'error_fimo_parsing'; fimo_message = str(e_parse)
            else: print("  FIMO output file (fimo.tsv) not found. Assuming no hits."); fimo_status = 'success_output_missing'
            return {'status': fimo_status, 'hits': parsed_hits, 'message': fimo_message}
    except Exception as e: print(f"ERROR during FIMO execution setup or file handling: {e}", file=sys.stderr); traceback.print_exc(); return {'status': 'error_fimo_setup', 'message': str(e), 'hits': []}
    finally:
        if 'dna_filepath' in locals() and os.path.exists(dna_filepath) and not dna_file.delete: os.remove(dna_filepath)
        if 'motif_filepath' in locals() and os.path.exists(motif_filepath) and not motif_file.delete: os.remove(motif_filepath)

def run_string_query(protein_name: str, species_id: int, min_score: int, max_interactors: int) -> Dict:
    print(f"\n--- TOOL: Querying STRING DB for '{protein_name}' ---")
    interactors = []; request_url = "/".join([STRING_API_URL, STRING_OUTPUT_FORMAT, "interaction_partners"])
    params = {"identifiers": protein_name, "species": species_id, "required_score": min_score, "limit": max_interactors * 2}; status = 'error_string_request_failed'
    try:
        response = requests.get(request_url, params=params, timeout=45); response.raise_for_status(); data = response.json(); status = 'success'
        if data:
            processed_count = 0
            for item in data:
                interactor_name = item.get('preferredName_B'); score = item.get('score'); string_id_b = item.get('stringId_B')
                if interactor_name and score is not None and string_id_b: interactors.append({"name": interactor_name, "score": score, "string_id": string_id_b}); processed_count += 1
                else: print(f"  Warn: Skipping STRING entry due to missing data: {item}", file=sys.stderr)
            interactors.sort(key=lambda x: x['score'], reverse=True); interactors = interactors[:max_interactors]; print(f"  Found {len(interactors)} interactors (processed {processed_count}) passing threshold.")
        else: print(f"  No interactors found in STRING DB for '{protein_name}' above score {min_score}.")
    except requests.exceptions.Timeout: print(f"  ERROR: STRING DB request timed out.", file=sys.stderr); status = 'error_string_timeout'
    except requests.exceptions.RequestException as e: status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'; print(f"  ERROR querying STRING DB (Status: {status_code}): {e}", file=sys.stderr); status = f'error_string_http_{status_code}';
    except json.JSONDecodeError as e: print(f"  ERROR decoding STRING DB JSON response: {e}", file=sys.stderr); status = 'error_string_json_decode'
    except Exception as e: print(f"  Unexpected ERROR during STRING DB query: {e}", file=sys.stderr); traceback.print_exc(); status = 'error_string_unknown'
    return {'status': status, 'interactors': interactors}

# --- NEW: On-the-fly DNABERT Embedding Calculation ---
def calculate_dnabert_embedding(
    dna_sequence: str,
    dnabert_model: BertModel,
    dnabert_tokenizer: AutoTokenizer,
    kmer: int,
    max_len: int,
    device: torch.device
) -> Union[np.ndarray, None]:
    """Generates DNABERT embedding for a given sequence."""
    print(f"  Calculating DNABERT {kmer}-mer embedding on the fly...")
    try:
        # Format sequence into k-mers separated by spaces
        sequence_kmers = ' '.join([dna_sequence[i:i+kmer] for i in range(0, (len(dna_sequence)-kmer+1))])
        if not sequence_kmers:
             print("  WARN: K-mer sequence is empty after formatting.", file=sys.stderr)
             return None

        with torch.no_grad():
            # Tokenize
            model_input = dnabert_tokenizer.encode_plus(
                sequence_kmers,
                add_special_tokens=True, # Add [CLS] and [SEP]
                max_length=max_len,
                truncation=True,
                return_tensors='pt' # Return PyTorch tensors
            )["input_ids"]

            if model_input.shape[1] <= 2: # Should have at least CLS, SEP, and one kmer token
                 print(f"  WARN: Tokenized input is too short ({model_input.shape[1]} tokens). Sequence: '{dna_sequence[:50]}...'", file=sys.stderr)
                 return None

            # Move to device
            model_input = model_input.to(device)

            # Get embeddings from DNABERT model
            output = dnabert_model(model_input)
            embed = output[0] # Last hidden state
            embed = torch.squeeze(embed) # Remove batch dimension

            # Discard [CLS] (first) and [SEP] (last) token embeddings
            if embed.dim() > 1 and embed.shape[0] > 2:
                 embed = embed[1:-1, :]
            elif embed.dim() == 1 and embed.shape[0] > 2: # Should not happen with hidden states, but handle defensively
                 print(" WARN: Unexpected embedding dimension after squeeze, returning None.", file=sys.stderr)
                 return None
            else: # Handle cases too short to trim
                 print(f" WARN: Embedding shape {embed.shape} too small to trim CLS/SEP, returning None.", file=sys.stderr)
                 return None


            # Convert to NumPy array on CPU
            embed_np = embed.cpu().numpy()
            print(f"  On-the-fly embedding calculated. Shape: {embed_np.shape}")
            return embed_np

    except Exception as e:
        print(f"  ERROR calculating DNABERT embedding on the fly: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None
    
# --- MODIFIED: run_transformer_prediction ---
def run_transformer_prediction(
    dna_sequence: str,
    protein_name: str,
    transformer_model, # The main predictor model
    pro_embs_dict: Dict,
    dna_db_con, # Connection to precomputed embeddings
    transformer_config: Dict,
    device: torch.device, # Device for *transformer_model*
    # --- New arguments for on-the-fly DNA embedding ---
    dnabert_model: BertModel,
    dnabert_tokenizer: AutoTokenizer,
    dnabert_kmer: int,
    dnabert_max_len: int,
    dnabert_device: torch.device # Device for *DNABERT model* (can be same as device)

) -> Dict:
    """Runs the transformer model prediction. Calculates DNA embedding on the fly if not found in DB."""
    print(f"\n--- TOOL: Running Transformer Prediction for '{protein_name}' ---")
    prob = None; status = 'success'; error_msg = None; dna_emb_source = "Unknown"

    if transformer_model is None: return {'status': 'error_model_not_loaded', 'probability': None, 'message': "Transformer predictor model object is None."}
    if not pro_embs_dict: return {'status': 'error_pro_emb_missing', 'probability': None, 'message': "Protein embeddings dictionary is missing."}
    if dna_db_con is None: print("  WARN: DNA embedding DB connection is None. Will attempt on-the-fly calculation only.")

    dna_emb_np = None

    # 1. Try loading DNA embedding from DB
    if dna_db_con:
        try:
            print(f"  Querying DNA embedding DB for sequence (length {len(dna_sequence)})...")
            result = dna_db_con.execute(f"SELECT embedding FROM dna_embeddings WHERE dna_sequence = ?", [dna_sequence]).fetchone()
            if result and result[0]:
                dna_emb_np = pickle.loads(result[0])
                dna_emb_source = "Database"
                print(f"  DNA embedding loaded from DB. Shape: {dna_emb_np.shape}")
            else:
                print("  DNA embedding not found in DB.")
        except Exception as e:
            status = 'error_dna_emb_db_load'; error_msg = f"DB Query/Load Error: {e}"
            print(f"  ERROR loading DNA embedding from DB: {e}", file=sys.stderr)
            # Continue to try on-the-fly calculation if DB load fails

    # 2. Calculate DNA embedding on the fly if not loaded from DB
    if dna_emb_np is None and status == 'success': # Only proceed if no critical DB error stopped us
        if dnabert_model and dnabert_tokenizer:
            dna_emb_np = calculate_dnabert_embedding(
                dna_sequence=dna_sequence,
                dnabert_model=dnabert_model,
                dnabert_tokenizer=dnabert_tokenizer,
                kmer=dnabert_kmer,
                max_len=dnabert_max_len,
                device=dnabert_device # Use the device DNABERT is on
            )
            if dna_emb_np is not None:
                dna_emb_source = "On-the-fly"
            else:
                # Calculation failed
                status = 'error_dna_emb_calc_failed'
                error_msg = "On-the-fly DNA embedding calculation failed."
        else:
            status = 'error_dnabert_missing'
            error_msg = "DNA embedding not found in DB and DNABERT resources unavailable for on-the-fly calculation."

    # 3. Load Protein embedding
    pro_emb_np = pro_embs_dict.get(protein_name)

    # 4. Check if embeddings are available before proceeding
    if dna_emb_np is None and status == 'success': # Should be caught by previous steps, but double check
        status = 'error_dna_emb_unavailable'; error_msg = "DNA embedding could not be loaded or calculated."
    if pro_emb_np is None and status == 'success':
        status = 'error_pro_emb_not_found'; error_msg = f"Protein embedding not found for '{protein_name}'."

    # 5. Preprocess and run inference if embeddings are ready
    if status == 'success':
        try: max_d = int(transformer_config['max_dna_seq']); max_p = int(transformer_config['max_protein_seq'])
        except (ValueError, TypeError, KeyError) as e_cfg:
             status = 'error_config_invalid_length'; error_msg = f"Invalid/missing max seq length in config: {e_cfg}"; print(f"  ERROR: Invalid config: {e_cfg}", file=sys.stderr);
             return {'status': status, 'probability': None, 'message': error_msg}
        try:
            # Preprocess embeddings (using the main transformer's target device)
            dna_emb_tensor, input_mask_d_np = eval_unify_dim_embedding(dna_emb_np, max_d, return_start_index=False)
            pro_emb_tensor, input_mask_p_np = eval_unify_dim_embedding(pro_emb_np, max_p, return_start_index=False)
            # Move tensors to the *transformer model's* device
            dna_emb = dna_emb_tensor.unsqueeze(0).to(device); pro_emb = pro_emb_tensor.unsqueeze(0).to(device)
            dna_imask = torch.from_numpy(input_mask_d_np).unsqueeze(0).to(device).float()
            pro_imask = torch.from_numpy(input_mask_p_np).unsqueeze(0).to(device).float()
            # Create masks
            dna_mask_sa = dna_imask.unsqueeze(1).unsqueeze(2); protein_mask_sa = pro_imask.unsqueeze(1).unsqueeze(3)
            cross_attn_mask_4d = torch.matmul(protein_mask_sa, dna_mask_sa); dna_mask_4d = torch.matmul(dna_mask_sa.permute(0,1,3,2), dna_mask_sa); protein_mask_4d = torch.matmul(protein_mask_sa, protein_mask_sa.permute(0,1,3,2))
        except Exception as e_prep: status = 'error_tensor_prep'; error_msg = f"Tensor Prep Error: {e_prep}"; print(f"  ERROR preparing tensors: {e_prep}", file=sys.stderr)

    if status == 'success':
        transformer_model.eval()
        with torch.no_grad():
            try:
                logits = transformer_model(dna_emb, pro_emb, dna_mask_4d, protein_mask_4d, cross_attn_mask_4d)
                if logits.ndim > 1 and logits.shape[1] > 1: logits = logits[:, 0]
                elif logits.ndim > 1 and logits.shape[1] == 1: logits = logits.squeeze(1)
                prob = torch.sigmoid(logits).item(); print(f"  Transformer Raw Probability: {prob:.4f} (DNA Emb Source: {dna_emb_source})")
            except Exception as e_inf: status = 'error_inference'; error_msg = f"Inference Error: {e_inf}"; print(f"  ERROR during transformer inference: {e_inf}", file=sys.stderr); traceback.print_exc()

    # Include dna_emb_source in the output message if helpful
    final_message = f"DNA Source: {dna_emb_source}. {error_msg or ''}".strip()
    return {'status': status, 'probability': prob, 'message': final_message}

# --- NEW: Wrapper Function ---
def get_transformer_pred_node_wrapper(
    state: AgentState,
    # These will be pre-filled by partial
    transformer_model,
    pro_embs_dict,
    dna_db_con,
    transformer_config,
    device,
    training_protein_set,
    dnabert_model,
    dnabert_tokenizer,
    dnabert_kmer,
    dnabert_max_len,
    dnabert_device
    ) -> AgentState:
    """
    Wrapper for get_transformer_pred_node that LangGraph can easily introspect.
    It takes state and calls the actual function with resources bound by partial.
    """
    if state.get("error"):
        return state # Pass through errors

    try:
        # Call the original prediction function, passing necessary state components
        # and all the other arguments which were bound by partial to this wrapper.
        results = run_transformer_prediction(
            dna_sequence=state["dna_sequence"],
            protein_name=state["protein_name"],
            # Pass through resources
            transformer_model=transformer_model,
            pro_embs_dict=pro_embs_dict,
            dna_db_con=dna_db_con,
            transformer_config=transformer_config,
            device=device,
            dnabert_model=dnabert_model,
            dnabert_tokenizer=dnabert_tokenizer,
            dnabert_kmer=dnabert_kmer,
            dnabert_max_len=dnabert_max_len,
            dnabert_device=dnabert_device
        )

        prob = results.get('probability')
        is_known = state["protein_name"] in training_protein_set if training_protein_set else False

        if results.get('status', '').startswith('error'):
             print(f"  ERROR: Transformer Prediction Failed ({results.get('status')}): {results.get('message')}", file=sys.stderr)
             # Allow to continue, but record the issue
             # Store None for probability, maybe set error? Or just keep known status?
             # Let's just return the state update without error for now, error is in results['message']
             return {**state, "transformer_prob": None, "is_known_protein": is_known} # Keep is_known status

        print(f"  Transformer Prob: {prob:.4f}, Known Protein (in Training Set): {is_known}")
        return {**state, "transformer_prob": prob, "is_known_protein": is_known}

    except Exception as e:
        print(f"  Critical Error in get_transformer_pred_node_wrapper: {e}");
        traceback.print_exc();
        return {**state, "error": f"Transformer Node Wrapper Error: {e}"}



# --- LLM Interaction (Refactored) ---
def _parse_llm_response(raw_response_text: str) -> Tuple[Union[int, None], Union[str, None]]:
    """Parses the prediction label and explanation from LLM raw text."""
    prediction = None; 
    explanation = raw_response_text.strip() if raw_response_text else ""
    patterns = [
        r"interaction label is\s*:?\s*\*?([01])\**", 
        r"\bFinal Label\s*:\s*\*?([01])\**", 
        r"\bprediction\s*:\s*\*?([01])\**", 
        r"\blabel\s*:\s*\*?([01])\**",]
    parsed_explanation = explanation
    for pattern in patterns:
        match = re.search(pattern, explanation, re.IGNORECASE | re.DOTALL)
        if match: prediction = int(match.group(1)); parsed_explanation = explanation[:match.start()].strip(); break
    if prediction is None:
        found_digits = re.findall(r"\b([01])\b", explanation)
        if found_digits: prediction = int(found_digits[-1]); parsed_explanation = explanation # Keep full explanation if using fallback
    if prediction is None:
        if explanation.endswith("1"): prediction = 1; parsed_explanation = explanation[:-1].strip()
        elif explanation.endswith("0"): prediction = 0; parsed_explanation = explanation[:-1].strip()
    if parsed_explanation:
        parsed_explanation = re.sub(r'<\/?(?:think|thought|reasoning|analysis|explanation)>', '', parsed_explanation, flags=re.IGNORECASE).strip()
        parsed_explanation = re.sub(r'^##+.*\b(?:label|prediction|output)\b.*$', '', parsed_explanation, flags=re.IGNORECASE | re.MULTILINE).strip()
        if parsed_explanation.lower().startswith("explanation:"): 
            parsed_explanation = parsed_explanation[len("explanation:"):].strip()
        if parsed_explanation.lower().startswith("analysis:"): 
            parsed_explanation = parsed_explanation[len("analysis:"):].strip()

        # --- NEW: Trim trailing whitespace and incomplete words/common artifacts ---
        # Remove trailing whitespace rigorously
        parsed_explanation = parsed_explanation.rstrip()
        # Remove common trailing words/prepositions if they are left dangling
        common_trailing = ["the", "a", "an", "is", "of", "in", "on", "at", "for", "to", "and", "or", "but"]
        words = parsed_explanation.split()
        if words and words[-1].lower() in common_trailing:
            print(f"  Trimming trailing word: '{words[-1]}'")
            parsed_explanation = " ".join(words[:-1]).rstrip()

        # Remove trailing punctuation if desired (optional)
        # parsed_explanation = parsed_explanation.rstrip('.,;:')            
    if prediction is None: print("  LLM WARN: Could not parse prediction label (0 or 1) from response.")
    return prediction, parsed_explanation if parsed_explanation else explanation

def call_gemini_model(prompt: str, model_name: str, api_delay: float) -> Tuple[Union[int, None], Union[str, None], Union[str, None]]: # Return 3 items
    """Calls the specified Google Gemini model. Returns (prediction, parsed_explanation, raw_response_text)."""
    print(f"\n--- Calling Gemini Model: {model_name} ---")
    raw_response_text = "" # Initialize raw text
    parsed_explanation = None
    prediction = None
    error_explanation = f"Error: Initial Gemini call state."

    try:
        model = genai.GenerativeModel(model_name)
        safety_settings=[ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048)
        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        raw_response_text = response.text.strip() # Store raw response
        print(f"  LLM Raw Response Length: {len(raw_response_text)}")
        prediction, parsed_explanation = _parse_llm_response(raw_response_text) # Parse
        print(f"  LLM Parsed Vote: {prediction}")
        # On success, clear error explanation
        error_explanation = None

    except genai.types.generation_types.BlockedPromptException as bpe:
         print(f"  ERROR: Gemini Call Blocked (Safety): {bpe}", file=sys.stderr)
         error_explanation = f"Error: LLM Call Blocked due to safety settings."
         parsed_explanation = error_explanation # Use error as explanation
    except exceptions.ResourceExhausted as r_exc: # Catch the specific rate limit error
         print(f"  ERROR: Gemini Rate Limit Exceeded: {r_exc}", file=sys.stderr)
         suggested_delay_match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', str(r_exc))
         wait_time = int(suggested_delay_match.group(1)) if suggested_delay_match else api_delay
         print(f"  -> Will sleep for {wait_time:.1f} seconds due to rate limit.")
         time.sleep(wait_time) # Wait extra
         error_explanation = f"Error: Gemini Rate Limit Exceeded ({r_exc})"
         parsed_explanation = error_explanation
    except Exception as e:
         print(f"  ERROR calling/parsing Gemini model '{model_name}': {e}", file=sys.stderr); traceback.print_exc()
         error_explanation = f"Error during Gemini call: {e}"
         parsed_explanation = error_explanation # Use error as explanation
    finally:
        if api_delay > 0:
             print(f"  Pausing for {api_delay:.1f} seconds after API call...")
             time.sleep(api_delay)
        # Return parsed results AND raw text
        return prediction, parsed_explanation if error_explanation is None else error_explanation, raw_response_text


def call_ollama_model(prompt: str, model_tag: str, ollama_api_url: str) -> Tuple[Union[int, None], Union[str, None], Union[str, None]]: # Return 3 items
    """Calls the specified Ollama model via its API. Returns (prediction, parsed_explanation, raw_response_text)."""
    print(f"\n--- Calling Ollama Model: {model_tag} via API: {ollama_api_url} ---")
    headers = {'Content-Type': 'application/json'}
    data = {"model": model_tag, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 2048}}
    raw_response_text = ""
    parsed_explanation = None
    prediction = None
    error_explanation = f"Error: Initial Ollama call state."

    try:
        response = requests.post(ollama_api_url, headers=headers, data=json.dumps(data), timeout=180)
        response.raise_for_status()
        result = response.json()
        raw_response_text = result.get('response', '').strip() # Store raw response
        print(f"  LLM Raw Response Length: {len(raw_response_text)}")
        prediction, parsed_explanation = _parse_llm_response(raw_response_text) # Parse
        print(f"  LLM Parsed Vote: {prediction}")
        error_explanation = None # Clear error on success
    except requests.exceptions.ConnectionError as ce: error_explanation = f"Error: Cannot connect to Ollama API ({ce})"; print(f"  ERROR: {error_explanation}", file=sys.stderr); parsed_explanation=error_explanation
    except requests.exceptions.Timeout: error_explanation = "Error: Ollama API request timed out."; print(f"  ERROR: {error_explanation}", file=sys.stderr); parsed_explanation=error_explanation
    except requests.exceptions.RequestException as re_err: error_explanation = f"Error querying Ollama API: {re_err}"; print(f"  ERROR: {error_explanation}", file=sys.stderr); parsed_explanation=error_explanation
    except json.JSONDecodeError as je: error_explanation = f"Error decoding Ollama JSON response: {je}"; print(f"  ERROR: {error_explanation}", file=sys.stderr); parsed_explanation=error_explanation
    except Exception as e: error_explanation = f"Error during Ollama call: {e}"; print(f"  ERROR: {error_explanation}", file=sys.stderr); traceback.print_exc(); parsed_explanation=error_explanation

    return prediction, parsed_explanation if error_explanation is None else error_explanation, raw_response_text

def call_huggingface_model(prompt: str, model, tokenizer, device: torch.device) -> Tuple[Union[int, None], Union[str, None], Union[str, None]]: # Return 3 items
    """Runs inference using a pre-loaded HF model. Returns (prediction, parsed_explanation, raw_response_text)."""
    # ... (Setup: model_name, checks, eos/pad id logic unchanged) ...
    model_name = model.config._name_or_path if hasattr(model, 'config') else "HF Model"; print(f"\n--- Calling Hugging Face Model: {model_name} ---")
    eos_token_id = tokenizer.eos_token_id; pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id; print(f"  Using EOS: {eos_token_id}, PAD: {pad_token_id}")

    raw_response_text = ""
    parsed_explanation = None
    prediction = None
    error_explanation = f"Error: Initial HF call state."

    try:
        # Apply formatting (Choose ONE based on model)
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>" # Deepseek format
        # formatted_prompt = f"[INST] {prompt} [/INST]" # Llama2 format
        print(f"  Formatted Prompt (start): {formatted_prompt[:200]}...")

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        generation_params = {
            "max_new_tokens": 2048, # Increased
            "temperature": 0.5,
            "do_sample": True,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.1
        }
        print(f"  Generating response with params: {generation_params}")

        with torch.no_grad():
            print(f"  Input IDs shape: {inputs.input_ids.shape}")
            outputs = model.generate(**inputs, **generation_params)
            print(f"  Output IDs shape: {outputs.shape}")

        input_length = inputs.input_ids.shape[1]
        if outputs.shape[1] <= input_length:
             print("  WARN: Generation output length is <= input length. No new tokens generated.")
             raw_response_text = "" # Still store empty raw text
        else:
             # Decode ONLY the newly generated tokens for raw response
             raw_response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

        print(f"  LLM Raw Response Length: {len(raw_response_text)}")
        print(f"  Decoded Raw Text: '{raw_response_text}'")
        prediction, parsed_explanation = _parse_llm_response(raw_response_text) # Parse
        print(f"  LLM Parsed Vote: {prediction}")
        error_explanation = None # Clear error on success

    except Exception as e:
        print(f"  ERROR during Hugging Face model generation: {e}", file=sys.stderr)
        traceback.print_exc()
        error_explanation = f"Error during Hugging Face model generation: {e}"
        parsed_explanation = error_explanation # Use error as explanation

    return prediction, parsed_explanation if error_explanation is None else error_explanation, raw_response_text


# In your main script, where you have confidence parameters
# ... (original confidence parameters) ...

# --- Confidence Score Parameters (FOR TRANSFORMER-PRIORITY PROMPT) ---
TP_CONFIDENCE_BASELINE = 0.45
TP_DIRECT_MATCH_BOOST = 0.10
# ... (all other TP_ constants) ...
TP_TRANSFORMER_NOVELTY_FACTOR = 0.5
TP_SCAN_FAIL_FACTOR = 0.3 # Assuming SCAN_FAIL_FACTOR is generic enough

def calculate_confidence_score(state: AgentState, prompt_style: str) -> float:
    print(f"--- Calculating Rule-Based Confidence Score (Style: {prompt_style}) ---")
    llm_vote = state.get('llm_vote')
    if llm_vote is None:
        print("  LLM vote is missing, cannot calculate confidence. Returning 0.0")
        return 0.0
    
    # Initialize motif quality parameters with general defaults
    num_hits_bonus_max = GENERAL_NUM_HITS_BONUS_MAX
    num_hits_scale_factor = GENERAL_NUM_HITS_SCALE_FACTOR
    max_hits_for_bonus_cap = GENERAL_MAX_HITS_FOR_BONUS_CAP
    low_pvalue_threshold = GENERAL_LOW_PVALUE_THRESHOLD
    low_pvalue_bonus = GENERAL_LOW_PVALUE_BONUS

    # Select parameters based on prompt_style
    if prompt_style == "transformer-priority":
        baseline = TP_CONFIDENCE_BASELINE
        direct_match_boost = TP_DIRECT_MATCH_BOOST
        # ... (all other TP_ parameters as before) ...
        transformer_novelty_factor = TP_TRANSFORMER_NOVELTY_FACTOR

    elif prompt_style == "motif-priority": # <<< NEW SECTION
        baseline = MP_CONFIDENCE_BASELINE
        direct_match_boost = MP_DIRECT_MATCH_BOOST
        direct_reliable_match_extra = MP_DIRECT_RELIABLE_MATCH_EXTRA
        direct_mismatch_penalty = MP_DIRECT_MISMATCH_PENALTY
        direct_reliable_mismatch_extra = MP_DIRECT_RELIABLE_MISMATCH_EXTRA
        direct_missing_penalty = MP_DIRECT_MISSING_PENALTY
        direct_absence_boost = MP_DIRECT_ABSENCE_BOOST

        indirect_match_boost = MP_INDIRECT_MATCH_BOOST
        indirect_mismatch_penalty = MP_INDIRECT_MISMATCH_PENALTY
        indirect_missing_penalty = MP_INDIRECT_MISSING_PENALTY
        indirect_absence_boost = MP_INDIRECT_ABSENCE_BOOST

        transformer_match_boost = MP_TRANSFORMER_MATCH_BOOST
        transformer_confident_extra = MP_TRANSFORMER_CONFIDENT_EXTRA
        transformer_mismatch_penalty = MP_TRANSFORMER_MISMATCH_PENALTY
        transformer_confident_mismatch_extra = MP_TRANSFORMER_CONFIDENT_MISMATCH_EXTRA
        transformer_novelty_factor = MP_TRANSFORMER_NOVELTY_FACTOR

        # Motif Quality Parameters
        num_hits_bonus_max = MP_NUM_HITS_BONUS_MAX
        num_hits_scale_factor = MP_NUM_HITS_SCALE_FACTOR
        max_hits_for_bonus_cap = MP_MAX_HITS_FOR_BONUS_CAP
        low_pvalue_threshold = MP_LOW_PVALUE_THRESHOLD
        low_pvalue_bonus = MP_LOW_PVALUE_BONUS

    else: # Default to original parameters (verbose or concise if not transformer-priority)
        baseline = CONFIDENCE_BASELINE
        direct_match_boost = DIRECT_MATCH_BOOST
        # ... (all other original parameters as before) ...
        transformer_novelty_factor = TRANSFORMER_NOVELTY_FACTOR

    scan_fail_factor = SCAN_FAIL_FACTOR # Generic
    transformer_confident_high = TRANSFORMER_CONFIDENT_HIGH # Generic
    transformer_confident_low = TRANSFORMER_CONFIDENT_LOW   # Generic

    confidence = baseline
    adjustments = []

    # --- 1. Direct Motif Evidence ---
    direct_summary = state.get('direct_motif_analysis_summary') or {}
    direct_found = direct_summary.get('motif_found_in_dna', False)
    direct_reliability = direct_summary.get('motif_reliability_source_category', 'Unknown')
    direct_scan_ok = not direct_summary.get('scan_status', 'unknown').startswith('error')
    direct_fetch_ok = not (state.get('direct_motif_fetch_results') or {}).get('status', 'unknown').startswith('error')
    is_reliable_direct = "High" in direct_reliability
    direct_num_hits = direct_summary.get('number_of_hits', 0)
    direct_best_pvalue = direct_summary.get('lowest_hit_pvalue') # Can be None

    direct_mod = scan_fail_factor if not direct_scan_ok else 1.0

    if direct_fetch_ok:
        if llm_vote == 1:
            if direct_found:
                adj = direct_match_boost * direct_mod
                adjustments.append(f"Direct Hit Match: +{adj:.2f}")
                confidence += adj
                if is_reliable_direct:
                    adj_extra = direct_reliable_match_extra * direct_mod
                    adjustments.append(f"  Reliable: +{adj_extra:.2f}")
                    confidence += adj_extra
                
                # Motif Quality Bonus (for motif-priority or if params are set)
                if num_hits_scale_factor > 0 and direct_num_hits > 0:
                    hits_bonus = min(num_hits_bonus_max, num_hits_scale_factor * min(direct_num_hits, max_hits_for_bonus_cap)) * direct_mod
                    adjustments.append(f"  Direct Num Hits Bonus ({direct_num_hits} hits): +{hits_bonus:.2f}")
                    confidence += hits_bonus
                if low_pvalue_bonus > 0 and direct_best_pvalue is not None and direct_best_pvalue < low_pvalue_threshold:
                    pval_b = low_pvalue_bonus * direct_mod
                    adjustments.append(f"  Direct Low PVal Bonus ({direct_best_pvalue:.1e}): +{pval_b:.2f}")
                    confidence += pval_b
            else: # Vote=1, no direct hit
                adj = direct_missing_penalty * direct_mod
                adjustments.append(f"Direct Hit Missing (Vote=1): {adj:.2f}")
                confidence += adj
        elif llm_vote == 0:
            if direct_found:
                adj = direct_mismatch_penalty * direct_mod
                adjustments.append(f"Direct Hit Mismatch (Vote=0): {adj:.2f}")
                confidence += adj
                if is_reliable_direct:
                    adj_extra = direct_reliable_mismatch_extra * direct_mod
                    adjustments.append(f"  Reliable Mismatch: {adj_extra:.2f}")
                    confidence += adj_extra
            else: # Vote=0, no direct hit
                adj = direct_absence_boost * direct_mod
                adjustments.append(f"Direct Hit Absence (Vote=0): +{adj:.2f}")
                confidence += adj
    else: adjustments.append("Direct Motif Fetch Failed: No Adjustment")

    # --- 2. Indirect Motif Evidence ---
    indirect_summary = state.get('indirect_motif_analysis_summary') or {}
    indirect_performed = indirect_summary.get('search_performed', False)

    if indirect_performed:
        indirect_found = indirect_summary.get('any_indirect_motif_found', False)
        indirect_scan_ok = not indirect_summary.get('scan_status', 'unknown').startswith('error')
        indirect_mod = scan_fail_factor if not indirect_scan_ok else 1.0
        indirect_num_hits = len(indirect_summary.get('interacting_tf_hits', []))
        indirect_best_hit = indirect_summary.get('best_interacting_tf_hit') # Dict or None
        indirect_best_pvalue = indirect_best_hit.get('pvalue') if indirect_best_hit else None


        if llm_vote == 1:
            if indirect_found:
                adj = indirect_match_boost * indirect_mod
                adjustments.append(f"Indirect Hit Match: +{adj:.2f}")
                confidence += adj
                # Motif Quality Bonus (for motif-priority or if params are set)
                if num_hits_scale_factor > 0 and indirect_num_hits > 0:
                    hits_bonus = min(num_hits_bonus_max, num_hits_scale_factor * min(indirect_num_hits, max_hits_for_bonus_cap)) * indirect_mod # Using same scale factor for simplicity, can be different
                    adjustments.append(f"  Indirect Num Hits Bonus ({indirect_num_hits} hits): +{hits_bonus:.2f}")
                    confidence += hits_bonus
                if low_pvalue_bonus > 0 and indirect_best_pvalue is not None and indirect_best_pvalue < low_pvalue_threshold:
                    pval_b = low_pvalue_bonus * indirect_mod
                    adjustments.append(f"  Indirect Low PVal Bonus ({indirect_best_pvalue:.1e}): +{pval_b:.2f}")
                    confidence += pval_b
            else: # Vote=1, no indirect hit
                adj = indirect_missing_penalty * indirect_mod
                adjustments.append(f"Indirect Hit Missing (Vote=1): {adj:.2f}")
                confidence += adj
        elif llm_vote == 0:
            if indirect_found:
                adj = indirect_mismatch_penalty * indirect_mod
                adjustments.append(f"Indirect Hit Mismatch (Vote=0): {adj:.2f}")
                confidence += adj
            else: # Vote=0, no indirect hit
                adj = indirect_absence_boost * indirect_mod
                adjustments.append(f"Indirect Hit Absence (Vote=0): +{adj:.2f}")
                confidence += adj
    else: adjustments.append("Indirect Search Skipped: No Adjustment")


    # --- 3. Transformer Evidence ---
    # (This part remains largely the same, the difference is in the *magnitude* of boosts/penalties determined by prompt_style)
    trans_prob = state.get('transformer_prob')
    is_known = state.get('is_known_protein', False)

    if trans_prob is not None:
        transformer_vote = 1 if trans_prob > 0.5 else 0
        is_transformer_pred_confident_for_bonus = (trans_prob > transformer_confident_high or trans_prob < transformer_confident_low)
        novelty_mod = transformer_novelty_factor if not is_known else 1.0

        if llm_vote == transformer_vote:
            adj = transformer_match_boost * novelty_mod
            adjustments.append(f"Transformer Match: +{adj:.2f}{'' if is_known else f' (Novel *{transformer_novelty_factor:.1f})'}")
            confidence += adj
            if is_transformer_pred_confident_for_bonus:
                adj_extra = transformer_confident_extra * novelty_mod
                adjustments.append(f"  Transformer Confident Bonus: +{adj_extra:.2f}")
                confidence += adj_extra
        else: # LLM vote contradicts transformer_vote
            adj = transformer_mismatch_penalty * novelty_mod
            adjustments.append(f"Transformer Mismatch: {adj:.2f}{'' if is_known else f' (Novel *{transformer_novelty_factor:.1f})'}")
            confidence += adj
            if is_transformer_pred_confident_for_bonus:
                adj_extra = transformer_confident_mismatch_extra * novelty_mod
                adjustments.append(f"  Transformer Confident Mismatch Penalty: {adj_extra:.2f}")
                confidence += adj_extra
    else: adjustments.append("Transformer Prob Missing: No Adjustment")

    final_confidence = max(0.0, min(1.0, confidence))

    print(f"  Baseline: {baseline:.2f}")
    for adj_str in adjustments: print(f"  {adj_str}")
    print(f"  => Final Confidence: {final_confidence:.3f}")

    return final_confidence


# --- Graph Nodes ---
def start_analysis(state: AgentState) -> AgentState: # Modified
    print("\n--- Starting Analysis ---"); print(f" Protein: {state['protein_name']}"); print(f" DNA Seq Length: {len(state['dna_sequence'])}")
    print(f"  Ground Truth Received: {state.get('ground_truth_label', 'MISSING')}") # Check input dict
    initial_state = {k: None for k in AgentState.__annotations__};
    initial_state.update({
        "protein_name": state['protein_name'],
        "dna_sequence": state['dna_sequence'],
        "ground_truth_label": state.get('ground_truth_label'), # Pass through ground truth
        "is_known_protein": False,
        "final_confidence": None
    });
    return initial_state

def fetch_direct_motifs_node(state: AgentState, cisbp_tf_info_df: pd.DataFrame, cisbp_pwm_dir: str) -> AgentState:
    print("\n--- Node: Fetch Direct Motifs ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state
    try: results = run_cisbp_fetch(protein_name=state["protein_name"], cisbp_tf_info_df=cisbp_tf_info_df, cisbp_pwm_dir=cisbp_pwm_dir); return {**state, "direct_motif_fetch_results": results}
    except Exception as e: print(f"  Critical Error in fetch_direct_motifs_node: {e}"); traceback.print_exc(); return {**state, "error": f"Fetch Direct Node Error: {e}"}

def scan_direct_dna_node(state: AgentState) -> AgentState:
    print("\n--- Node: Scan DNA for Direct Motifs ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state; 
    fetch_results = state.get("direct_motif_fetch_results") or {}; motifs_meta = fetch_results.get('motifs_metadata')
    if not motifs_meta: print("  Skipping scan: No direct motifs fetched or available."); return {**state, "direct_motif_scan_results": {"status": "success_skipped", "hits": []}}
    try: scan_results = run_motif_scan(dna_sequence=state["dna_sequence"], motifs_with_metadata=motifs_meta, pvalue_threshold=DEFAULT_P_VALUE_THRESHOLD); return {**state, "direct_motif_scan_results": scan_results}
    except Exception as e: print(f"  Critical Error in scan_direct_dna_node: {e}"); traceback.print_exc(); return {**state, "error": f"Scan Direct Node Error: {e}"}

def summarize_direct_analysis_node(state: AgentState) -> AgentState:
    print("\n--- Node: Summarize Direct Motif Analysis ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state; 
    fetch_results = state.get("direct_motif_fetch_results") or {}; scan_results = state.get("direct_motif_scan_results") or {"hits": [], "status": "unknown"}; motifs_metadata_list = [meta for _, meta in fetch_results.get('motifs_metadata', [])]; reliability_str = _calculate_cisbp_motif_reliability_for_prompt(motifs_metadata_list); all_significant_hits = scan_results.get("hits", []); motif_found_in_dna = len(all_significant_hits) > 0; best_hit = all_significant_hits[0] if motif_found_in_dna else None; best_score = best_hit.get('score') if best_hit else None; best_pvalue = best_hit.get('pvalue') if best_hit else None; scan_status = scan_results.get('status', 'unknown'); scan_message = scan_results.get('message');
    summary = {"p_value_threshold": DEFAULT_P_VALUE_THRESHOLD, "motif_found_in_dna": motif_found_in_dna, "motif_reliability_source_category": reliability_str, "highest_hit_score": best_score, "lowest_hit_pvalue": best_pvalue, "number_of_hits": len(all_significant_hits), "best_hit_details": best_hit, "protein_motif_info": fetch_results, "significant_hits": all_significant_hits, "scan_status": scan_status, "scan_message": scan_message}; print(f"  Direct Analysis Summary: Hits Found={summary['motif_found_in_dna']}, Reliability={summary['motif_reliability_source_category']}, Scan Status={scan_status}"); return {**state, "direct_motif_analysis_summary": summary}

def query_string_node(state: AgentState) -> AgentState:
    print("\n--- Node: Query STRING DB ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state
    try: results = run_string_query(protein_name=state["protein_name"], species_id=TARGET_SPECIES_TAXONOMY_ID, min_score=STRING_MIN_INTERACTION_SCORE, max_interactors=MAX_STRING_INTERACTORS_TO_CHECK); return {**state, "string_interactors": results.get('interactors', [])}
    except Exception as e: print(f"  Critical Error in query_string_node: {e}"); traceback.print_exc(); return {**state, "error": f"STRING Node Error: {e}"}

def filter_string_interactors_node(state: AgentState, cisbp_tf_info_df: pd.DataFrame) -> AgentState:
    print("\n--- Node: Filter STRING Interactors for TFs ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state; 
    string_interactors = state.get("string_interactors") or []; tf_interactors = []; tfs_to_analyze = []
    if cisbp_tf_info_df is None or cisbp_tf_info_df.empty: print("  WARN: CisBP TF Info DataFrame not available for filtering.", file=sys.stderr)
    else:
        known_tf_names = set(cisbp_tf_info_df['TF_Name'].unique()); tf_interactors = [i for i in string_interactors if i.get("name") in known_tf_names]; print(f"  Found {len(tf_interactors)} known TFs among {len(string_interactors)} STRING interactors.")
        tfs_to_analyze = tf_interactors
        if len(tf_interactors) > MAX_INTERACTING_TFS_TO_ANALYZE: print(f"  Limiting analysis to top {MAX_INTERACTING_TFS_TO_ANALYZE} TFs based on STRING score."); tfs_to_analyze.sort(key=lambda x: x.get('score', 0), reverse=True); tfs_to_analyze = tfs_to_analyze[:MAX_INTERACTING_TFS_TO_ANALYZE]
    return {**state, "tf_interactors_cisbp": tf_interactors, "tfs_analyzed_indirectly": tfs_to_analyze}

def fetch_indirect_motifs_node(state: AgentState, cisbp_tf_info_df: pd.DataFrame, cisbp_pwm_dir: str) -> AgentState:
    print("\n--- Node: Fetch Indirect Motifs ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state; 
    tfs_to_analyze = state.get("tfs_analyzed_indirectly") or []; all_indirect_motifs_meta = []; combined_status = "success_cisbp"; fetch_failures = 0
    if not tfs_to_analyze: print("  No interacting TFs to fetch motifs for."); return {**state, "indirect_motif_fetch_results": {"motifs_metadata": [], "status": "no_tfs_to_analyze"}}
    print(f"  Fetching motifs for {len(tfs_to_analyze)} interacting TFs: {[tf.get('name', '?') for tf in tfs_to_analyze]}")
    try:
        for tf_interactor in tfs_to_analyze:
            tf_name = tf_interactor.get('name'); string_score = tf_interactor.get('score');
            if not tf_name: continue
            results = run_cisbp_fetch(protein_name=tf_name, cisbp_tf_info_df=cisbp_tf_info_df, cisbp_pwm_dir=cisbp_pwm_dir)
            if results.get('status', '').startswith('success'):
                for motif_data, meta in results.get('motifs_metadata', []): meta['interacting_tf_name'] = tf_name; meta['interacting_tf_string_score'] = string_score; all_indirect_motifs_meta.append((motif_data, meta))
            else: fetch_failures += 1
        if fetch_failures == len(tfs_to_analyze) and len(tfs_to_analyze) > 0: combined_status = "error_cisbp_fetch_all_indirect"
        print(f"  Fetched {len(all_indirect_motifs_meta)} motifs for interacting TFs."); return {**state, "indirect_motif_fetch_results": {"motifs_metadata": all_indirect_motifs_meta, "status": combined_status}}
    except Exception as e: print(f"  Critical Error in fetch_indirect_motifs_node: {e}"); traceback.print_exc(); return {**state, "error": f"Fetch Indirect Node Error: {e}"}

def scan_indirect_dna_node(state: AgentState) -> AgentState:
    print("\n--- Node: Scan DNA for Indirect Motifs ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state; 
    fetch_results = state.get("indirect_motif_fetch_results") or {}; motifs_meta = fetch_results.get('motifs_metadata')
    if not motifs_meta: print("  Skipping scan: No indirect motifs fetched or available."); return {**state, "indirect_motif_scan_results": {"status": "success_skipped", "hits": []}}
    try:
        scan_results = run_motif_scan(dna_sequence=state["dna_sequence"], motifs_with_metadata=motifs_meta, pvalue_threshold=DEFAULT_P_VALUE_THRESHOLD)
        metadata_map_indirect = {meta['Motif_ID']: meta for _, meta in motifs_meta if meta.get('Motif_ID')}
        for hit in scan_results.get("hits", []):
            if 'interacting_tf' not in hit or hit['interacting_tf'] is None: 
                meta = metadata_map_indirect.get(hit.get('motif_id'));
                if meta: hit['interacting_tf'] = meta.get('interacting_tf_name'); hit['interacting_tf_string_score'] = meta.get('interacting_tf_string_score');
        return {**state, "indirect_motif_scan_results": scan_results}
    except Exception as e: print(f"  Critical Error in scan_indirect_dna_node: {e}"); traceback.print_exc(); return {**state, "error": f"Scan Indirect Node Error: {e}"}

def summarize_indirect_analysis_node(state: AgentState) -> AgentState:
    print("\n--- Node: Summarize Indirect Motif Analysis ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state; 
    string_interactors = state.get("string_interactors") or []; tf_interactors_cisbp = state.get("tf_interactors_cisbp") or []; tfs_analyzed_indirectly = state.get("tfs_analyzed_indirectly") or []; scan_results = state.get("indirect_motif_scan_results") or {"hits": [], "status": "unknown"}; scan_status = scan_results.get('status', 'unknown'); scan_message = scan_results.get('message'); all_significant_hits = scan_results.get("hits", []); any_indirect_motif_found = len(all_significant_hits) > 0; all_significant_hits.sort(key=lambda x: x.get('pvalue', float('inf'))); best_hit = all_significant_hits[0] if any_indirect_motif_found else None;
    summary = {"search_performed": True, "skipped_reason": None, "interactors_found_string": string_interactors, "interactors_tf_cisbp": tf_interactors_cisbp, "interactors_tf_analyzed": tfs_analyzed_indirectly, "interacting_tf_hits": all_significant_hits, "best_interacting_tf_hit": best_hit, "any_indirect_motif_found": any_indirect_motif_found, "scan_status": scan_status, "scan_message": scan_message, "error_message": None}
    # Use pre-set summary if indirect path was skipped
    if state.get("indirect_motif_analysis_summary") and not state["indirect_motif_analysis_summary"].get("search_performed"):
        summary = state["indirect_motif_analysis_summary"]
        print(f"  Indirect Analysis Summary: Skipped - {summary.get('skipped_reason')}")
    else: print(f"  Indirect Analysis Summary: Search Performed, Hits Found={summary['any_indirect_motif_found']}, #Interactors={len(summary.get('interactors_found_string',[]))}, #TF_Interactors={len(summary.get('interactors_tf_cisbp',[]))}, #TFs_Analyzed={len(summary.get('interactors_tf_analyzed',[]))}, Scan Status={scan_status}");
    return {**state, "indirect_motif_analysis_summary": summary}

def get_transformer_pred_node(state: AgentState, transformer_model, pro_embs_dict, dna_db_con, transformer_config, device, training_protein_set) -> AgentState:
    print("\n--- Node: Get Transformer Prediction ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return state
    try:
        results = run_transformer_prediction(dna_sequence=state["dna_sequence"], protein_name=state["protein_name"], transformer_model=transformer_model, pro_embs_dict=pro_embs_dict, dna_db_con=dna_db_con, transformer_config=transformer_config, device=device)
        prob = results.get('probability'); is_known = state["protein_name"] in training_protein_set if training_protein_set else False
        if results.get('status', '').startswith('error'): print(f"  ERROR: Transformer Prediction Failed ({results.get('status')}): {results.get('message')}", file=sys.stderr); return {**state, "transformer_prob": None, "is_known_protein": is_known} # Keep known status
        print(f"  Transformer Prob: {prob:.4f}, Known Protein (in Training Set): {is_known}"); return {**state, "transformer_prob": prob, "is_known_protein": is_known}
    except Exception as e: print(f"  Critical Error in get_transformer_pred_node: {e}"); traceback.print_exc(); return {**state, "error": f"Transformer Node Error: {e}"}


def generate_concise_llm_prompt(state: AgentState) -> str:
    """Generates the concise LLM prompt matching the fine-tuning format."""
    protein_name = state.get('protein_name', 'N/A')
    direct_summary = state.get("direct_motif_analysis_summary") or {}
    indirect_summary = state.get("indirect_motif_analysis_summary") or {}
    trans_prob = state.get("transformer_prob")
    is_known = state.get("is_known_protein", False)

    # --- Format Protein Info ---
    protein_novelty = "Known (Present in Transformer Training Set)" if is_known else "Novel (Absent from Transformer Training Set)"
    protein_info_str = f"{protein_name} ({protein_novelty})"

    # --- Format Direct Motif Summary ---
    direct_p_val = f"{direct_summary.get('p_value_threshold', DEFAULT_P_VALUE_THRESHOLD):.1e}"
    direct_motif_found_dna = direct_summary.get('motif_found_in_dna', False)
    direct_reliability_str = direct_summary.get('motif_reliability_source_category', 'Unknown')
    direct_scan_status = direct_summary.get('scan_status', 'unknown')
    direct_hits_count = direct_summary.get('number_of_hits', 0)
    direct_fetch_status = (state.get('direct_motif_fetch_results') or {}).get('status', 'unknown')
    if not direct_fetch_status.startswith('success'): direct_motif_summary = f"Fetch Error ({direct_fetch_status})"
    elif not direct_scan_status.startswith('success'): direct_motif_summary = f"Scan Error ({direct_scan_status})"
    elif direct_motif_found_dna: direct_motif_summary = f"Hits Found: YES ({direct_hits_count} hits, Reliability: {direct_reliability_str})"
    else: direct_motif_summary = f"Hits Found: NO (Reliability: {direct_reliability_str})"

    # --- Format Indirect Motif Summary ---
    indirect_p_val = f"{direct_summary.get('p_value_threshold', DEFAULT_P_VALUE_THRESHOLD):.1e}"
    indirect_performed = indirect_summary.get("search_performed", False)
    indirect_skipped_reason = indirect_summary.get("skipped_reason")
    any_indirect_motif_found = indirect_summary.get("any_indirect_motif_found", False)
    indirect_scan_status = indirect_summary.get('scan_status', 'unknown')
    indirect_hits_count = len(indirect_summary.get("interacting_tf_hits", []))
    if not indirect_performed: 
        reason = f": {indirect_skipped_reason}" if indirect_skipped_reason else ""
        indirect_motif_summary = f"Skipped{reason}" 
    elif not indirect_scan_status.startswith('success'): indirect_motif_summary = f"Scan Error ({indirect_scan_status})"
    elif any_indirect_motif_found: indirect_motif_summary = f"Hits Found: YES ({indirect_hits_count} hits from interacting TFs)"
    else:
        tfs_analyzed_count = len(indirect_summary.get("interactors_tf_analyzed", []))
        if tfs_analyzed_count > 0: indirect_motif_summary = f"Hits Found: NO (Analyzed {tfs_analyzed_count} TFs)"
        else: indirect_motif_summary = "Hits Found: NO (No relevant TFs analyzed)"

    # --- Format Transformer Summary ---
    transformer_prob_str = "N/A"; transformer_interpretation = "N/A"; trustworthiness_context = "Unknown"
    if trans_prob is not None:
         transformer_prob_str = f"{trans_prob:.4f}"
         if trans_prob > TRANSFORMER_CONFIDENT_HIGH: transformer_interpretation = "High Probability"
         elif trans_prob > 0.6: transformer_interpretation = "Moderate Probability"
         elif trans_prob > 0.4: transformer_interpretation = "Uncertain"
         elif trans_prob > TRANSFORMER_CONFIDENT_LOW: transformer_interpretation = "Low Probability"
         else: transformer_interpretation = "Very Low Probability"
         confidence_level = "Unknown"; # ... (determine confidence level) ...
         trustworthiness_context = f"Model Confidence: {confidence_level}";
         if not is_known: trustworthiness_context += " (Novel Protein - Reduced Trust)"
         else: trustworthiness_context += " (Known Protein - Standard Trust)"
    else: transformer_prob_str = "N/A (Failed)"; trustworthiness_context = "N/A (Failed)"

    # --- Construct the Concise Prompt ---
    prompt = f"""Below is evidence regarding a DNA sequence and protein interaction. Provide a step-by-step thinking process synthesizing this evidence to reach a final label (0 or 1).

### Instruction:
You are a computational biology expert evaluating potential DNA-protein interactions based on multiple evidence sources. Synthesize the provided evidence step-by-step within the `<think>` tags and provide the final interaction label.

### Protein:
{protein_info_str}

### Retrieved Evidence:
* Direct Motif Scan (p<={direct_p_val}): {direct_motif_summary}
* Indirect Motif Scan (p<={indirect_p_val}): {indirect_motif_summary}
* Transformer Probability: {transformer_prob_str} ({transformer_interpretation}) - [Trustworthiness: {trustworthiness_context}]

### Analysis Task:
Generate a step-by-step reasoning process within <think> tags, synthesizing the 'Retrieved Evidence' above to determine the interaction label. Conclude with '### Interaction:\nThe interaction label is [0 or 1]'.

### Analysis:
<think>
"""
    return prompt

def _format_common_evidence_for_prompt(state: AgentState) -> Dict[str, str]:
    """
    Helper function to format common evidence pieces used in various LLM prompts.
    Returns a dictionary of formatted strings.
    """
    protein_name = state.get('protein_name', 'N/A')
    dna_sequence = state.get('dna_sequence', '')
    direct_summary = state.get("direct_motif_analysis_summary") or {}
    indirect_summary = state.get("indirect_motif_analysis_summary") or {}
    trans_prob = state.get("transformer_prob")
    is_known = state.get("is_known_protein", False)

    evidence_parts = {}

    # --- Format DNA info ---
    dna_len = len(dna_sequence)
    max_dna_len_full_display = 1024
    disp_dna = dna_sequence if dna_len <= max_dna_len_full_display else f"{dna_sequence[:60]}...{dna_sequence[-60:]}"
    evidence_parts['disp_dna_str'] = disp_dna
    evidence_parts['dna_info_str'] = f"(Length: {dna_len})"

    # --- Format Direct Motif Evidence ---
    direct_motif_info_summary = direct_summary.get('protein_motif_info', {}) # Renamed to avoid conflict
    direct_motif_status = direct_motif_info_summary.get('status', 'unknown')
    evidence_parts['direct_protein_has_motifs_str'] = 'Yes' if direct_motif_status.startswith('success') else 'No/Error'
    evidence_parts['direct_reliability_str'] = direct_summary.get('motif_reliability_source_category', 'Unknown')
    evidence_parts['direct_motif_found_dna_str'] = str(direct_summary.get('motif_found_in_dna', False)) # As string "True"/"False"
    
    direct_hits = direct_summary.get('significant_hits', [])
    p_thresh = direct_summary.get('p_value_threshold', DEFAULT_P_VALUE_THRESHOLD)
    p_thresh_print = f"{p_thresh:.1e}" if isinstance(p_thresh, float) else 'N/A'

    # Direct Motif - CisBP Search Details
    direct_cisbp_search_details_list = [f"Status: {direct_motif_status}. Reliability: {evidence_parts['direct_reliability_str']}."]
    if direct_motif_status.startswith('success'):
        num_direct = direct_motif_info_summary.get('num_direct_selected', 0)
        num_inferred = direct_motif_info_summary.get('num_inferred_selected', 0)
        num_selected = direct_motif_info_summary.get('num_selected', 0)
        num_total = direct_motif_info_summary.get('num_total_found', 0)
        filtering_applied = direct_motif_info_summary.get('filtering_applied', False)
        filter_msg = f"(Top {num_selected} selected from {num_total})" if filtering_applied else f"({num_selected} total)"
        direct_cisbp_search_details_list.append(f" Found {num_direct} Direct, {num_inferred} Inferred motifs {filter_msg}.")
        direct_motif_detail_list_meta = [m for _, m in direct_motif_info_summary.get('motifs_metadata', [])] # Renamed
        max_direct_motifs_to_list = 3
        listed_motifs = 0
        for m_meta in direct_motif_detail_list_meta: # Renamed
            if listed_motifs < max_direct_motifs_to_list:
                direct_cisbp_search_details_list.append(f" | {m_meta.get('Motif_ID', '?')} ({m_meta.get('TF_Status', '?')}/{m_meta.get('MSource_Identifier','?')})")
                listed_motifs += 1
            else:
                direct_cisbp_search_details_list.append(f" | ...")
                break
    elif direct_motif_status.startswith('error'):
        direct_cisbp_search_details_list.append(f" (Error during fetch: {direct_motif_status})")
    evidence_parts['direct_motif_cisbp_search_details_str'] = "".join(direct_cisbp_search_details_list)

    # Direct Motif - DNA Scan Details
    direct_scan_info_list = []
    max_direct_scan_hits_to_list = 2
    direct_scan_status = direct_summary.get('scan_status', 'unknown') # Renamed
    if direct_scan_status.startswith('error'):
        direct_scan_info_list.append(f"  Scan Error: {direct_scan_status} ({direct_summary.get('scan_message', 'No details')})")
    elif direct_hits:
        direct_scan_info_list.append(f"  Hits in DNA (p<={p_thresh_print}, Top {min(len(direct_hits), max_direct_scan_hits_to_list)} by p-value):")
        for i, h in enumerate(direct_hits):
            if i < max_direct_scan_hits_to_list:
                hit_info = f"Mot:{h.get('motif_id','?')} @Pos:{h.get('position','?')}({h.get('strand','?')}) Score:{h.get('score',float('nan')):.2f} p:{h.get('pvalue', float('nan')):.2e}"
                direct_scan_info_list.append(f"    - {hit_info}")
        if len(direct_hits) > max_direct_scan_hits_to_list:
            direct_scan_info_list.append("    ...")
    else:
        direct_scan_info_list.append(f"  No significant direct motif hits found (p <= {p_thresh_print}). Scan status: {direct_scan_status}")
    evidence_parts['direct_motif_scan_details_str'] = "\n".join(direct_scan_info_list)

    # --- Format Indirect Motif Evidence ---
    indirect_search_performed = indirect_summary.get("search_performed", False)
    evidence_parts['indirect_search_performed_str'] = str(indirect_search_performed)
    evidence_parts['indirect_skipped_reason_str'] = indirect_summary.get("skipped_reason", "")
    evidence_parts['indirect_tfs_analyzed_count_str'] = str(len(indirect_summary.get("interactors_tf_analyzed", [])))
    evidence_parts['indirect_motifs_found_dna_str'] = str(indirect_summary.get("any_indirect_motif_found", False))

    # Indirect Motif - Search Details (STRING + TF analysis)
    indirect_info_list = []
    if not indirect_search_performed and evidence_parts['indirect_skipped_reason_str']:
        indirect_info_list.append(f"  Indirect search skipped: {evidence_parts['indirect_skipped_reason_str']}")
    elif not indirect_search_performed:
        indirect_info_list.append("  Indirect search was not performed.")
    else:
        string_interactors_count = len(indirect_summary.get("interactors_found_string", []))
        tf_interactors = indirect_summary.get("interactors_tf_cisbp", [])
        tfs_analyzed = indirect_summary.get("interactors_tf_analyzed", [])
        
        indirect_info_list.append(f"  STRING DB Search: Found {string_interactors_count} interactors (Score > {STRING_MIN_INTERACTION_SCORE}).")
        if not tf_interactors:
            indirect_info_list.append("    - None of these interactors are known TFs in our CisBP database.")
        else:
            indirect_info_list.append(f"    - {len(tf_interactors)} interactors are known TFs. Analyzed top {len(tfs_analyzed)}:")
            listed_tfs = 0
            max_indirect_interactors_to_list = 5
            for tf_i in tfs_analyzed:
                if listed_tfs < max_indirect_interactors_to_list:
                    indirect_info_list.append(f"      - {tf_i.get('name','?')} (STRING Score: {tf_i.get('score','?')})")
                    listed_tfs += 1
            if len(tfs_analyzed) > max_indirect_interactors_to_list:
                indirect_info_list.append(f"      ...")
    evidence_parts['indirect_motif_search_details_str'] = "\n".join(indirect_info_list)

    # Indirect Motif - DNA Scan Details
    indirect_scan_hits_list = []
    if indirect_search_performed: # Only show scan details if search was performed
        indirect_hits_data = indirect_summary.get("interacting_tf_hits", []) # Renamed
        indirect_scan_status_val = indirect_summary.get('scan_status', 'unknown') # Renamed
        max_indirect_scan_hits_to_list = 3
        if indirect_scan_status_val.startswith('error'):
            indirect_scan_hits_list.append(f"  Scan Error for Indirect Motifs: {indirect_scan_status_val} ({indirect_summary.get('scan_message', 'No details')})")
        elif indirect_hits_data:
            indirect_scan_hits_list.append(f"  Hits in DNA from Interacting TFs (p<={p_thresh_print}, Top {min(len(indirect_hits_data), max_indirect_scan_hits_to_list)} by p-value):")
            for i, h_indirect in enumerate(indirect_hits_data): # Renamed
                if i < max_indirect_scan_hits_to_list:
                    hit_info_indirect = f"TF:{h_indirect.get('interacting_tf','?')} (Mot:{h_indirect.get('motif_id','?')}) @Pos:{h_indirect.get('position','?')}({h_indirect.get('strand','?')}) Score:{h_indirect.get('score',float('nan')):.2f} p:{h_indirect.get('pvalue', float('nan')):.2e}"
                    indirect_scan_hits_list.append(f"    - {hit_info_indirect}")
            if len(indirect_hits_data) > max_indirect_scan_hits_to_list:
                indirect_scan_hits_list.append("    ...")
        elif indirect_summary.get("interactors_tf_cisbp"): # If TFs were found but no hits
             indirect_scan_hits_list.append(f"  No significant indirect motif hits found (p <= {p_thresh_print}) for analyzed interacting TFs. Scan status: {indirect_scan_status_val}")
        # else: if no TFs to analyze, this part is naturally empty.
    evidence_parts['indirect_motif_scan_details_str'] = "\n".join(indirect_scan_hits_list) if indirect_scan_hits_list else "  (Indirect scan not applicable or no hits/errors to report based on prior steps)"


    # --- Format Transformer Evidence ---
    evidence_parts['transformer_prob_str'] = f"{trans_prob:.4f}" if trans_prob is not None else 'N/A (Prediction Failed or Skipped)'
    evidence_parts['protein_status_str'] = "Known (Present in Transformer Training Set)" if is_known else "Novel (Absent from Transformer Training Set)"
    transformer_confidence_str = "Unknown"
    if trans_prob is not None:
        if trans_prob < 0.1 or trans_prob > 0.9: transformer_confidence_str = "High"
        elif 0.4 <= trans_prob <= 0.6: transformer_confidence_str = "Low (Uncertain)"
        else: transformer_confidence_str = "Moderate"
    evidence_parts['transformer_confidence_str'] = transformer_confidence_str
    evidence_parts['transformer_explanation_str'] = f"""  - Model Info: Trained on ~1500 ChIP-seq datasets (~800 TFs). Generally high accuracy on known TFs, potentially less certain on novel ones."""

    return evidence_parts

def generate_verbose_llm_prompt(state: AgentState) -> str:
    protein_name = state.get('protein_name', 'N/A')
    evidence = _format_common_evidence_for_prompt(state) # Call the helper

    prompt = f"""Analyze the potential interaction between the DNA sequence and the Protein '{protein_name}' ({TARGET_SPECIES}). Your goal is to predict if an interaction occurs (1) or not (0), based *only* on the evidence provided below.

### Input Data:
Protein: {protein_name}
DNA {evidence['dna_info_str']}: {evidence['disp_dna_str']}

### Evidence Summary:
- Protein Novelty (vs Transformer Training): {evidence['protein_status_str']}
- Transformer Prediction Probability: {evidence['transformer_prob_str']} (Confidence: {evidence['transformer_confidence_str']})
- Direct Motif Evidence (Protein's own motifs):
    - Protein has known motifs? {evidence['direct_protein_has_motifs_str']} (Reliability: {evidence['direct_reliability_str']})
    - Direct motifs found in this DNA sequence? {evidence['direct_motif_found_dna_str']}
- Indirect Motif Evidence (via Protein-Protein Interaction):
    - Indirect Search Performed? {evidence['indirect_search_performed_str']} {f"(Skipped: {evidence['indirect_skipped_reason_str']})" if evidence['indirect_skipped_reason_str'] and evidence['indirect_search_performed_str'] == 'False' else ''}
    - Interacting TFs analyzed? {evidence['indirect_tfs_analyzed_count_str']}
    - Motifs of analyzed interacting TFs found in this DNA? {evidence['indirect_motifs_found_dna_str']}

### Supporting Information Details:
1. Direct Motif Analysis (Protein: {protein_name}):
    - CisBP Search Result: {evidence['direct_motif_cisbp_search_details_str']}
    - DNA Scan Results for Direct Motifs:
{evidence['direct_motif_scan_details_str']}

2. Indirect Motif Analysis (via {protein_name}'s interactors):
{evidence['indirect_motif_search_details_str']}
    - DNA Scan Results for Interacting TF Motifs:
{evidence['indirect_motif_scan_details_str']}

3. Transformer Model Prediction Details:
    - Probability: {evidence['transformer_prob_str']}
    - Protein Status (vs Training Set): {evidence['protein_status_str']}
{evidence['transformer_explanation_str']}

### Analysis Task:
Please perform a step-by-step analysis based *strictly* on the provided evidence:
1.  **Direct Evidence:** Evaluate the direct motif evidence. Does the protein have known binding motifs? Are they reliable? Were any significant hits found in the DNA sequence? Assess the strength of this evidence (strong, moderate, weak, none, or error).
2.  **Indirect Evidence:** Evaluate the indirect motif evidence. Was the search performed? Did the protein interact with known TFs? Were any motifs belonging to these interacting TFs found in the DNA sequence? Assess the strength of this evidence, considering it's less direct than the protein's own motifs.
3.  **Transformer Evidence:** Evaluate the Transformer prediction. What is the probability score? How confident does the model seem (High/Moderate/Low)? Does the protein's novelty affect your trust in this prediction?
4.  **Synthesis:** Synthesize all evidence streams. Do they agree or conflict? Which evidence seems most compelling or decisive in this specific case? For example, is strong direct evidence sufficient? Is indirect evidence relevant if direct evidence is missing or weak? How does the transformer prediction align?
5.  **Conclusion:** State your final prediction (0 for no interaction, 1 for interaction) based on your synthesis.

### MANDATORY Output Format:
Explanation: [Your detailed step-by-step reasoning following points 1-5 above]
The interaction label is: [0 or 1]"""
    return prompt

def generate_transformer_priority_llm_prompt(state: AgentState) -> str:
    protein_name = state.get('protein_name', 'N/A')
    evidence = _format_common_evidence_for_prompt(state)

    prompt = f"""Analyze the potential interaction between the DNA sequence and the Protein '{protein_name}' ({TARGET_SPECIES}). Your goal is to predict if an interaction occurs (1) or not (0), based *only* on the evidence provided below.

### Input Data:
Protein: {protein_name}
DNA {evidence['dna_info_str']}: {evidence['disp_dna_str']}

### Evidence Summary:
- Protein Novelty (vs Transformer Training): {evidence['protein_status_str']}
- Transformer Prediction Probability: {evidence['transformer_prob_str']} (Confidence: {evidence['transformer_confidence_str']})
- Direct Motif Evidence (Protein's own motifs):
    - Protein has known motifs? {evidence['direct_protein_has_motifs_str']} (Reliability: {evidence['direct_reliability_str']})
    - Direct motifs found in this DNA sequence? {evidence['direct_motif_found_dna_str']}
- Indirect Motif Evidence (via Protein-Protein Interaction):
    - Indirect Search Performed? {evidence['indirect_search_performed_str']} {f"(Skipped: {evidence['indirect_skipped_reason_str']})" if evidence['indirect_skipped_reason_str'] and evidence['indirect_search_performed_str'] == 'False' else ''}
    - Interacting TFs analyzed? {evidence['indirect_tfs_analyzed_count_str']}
    - Motifs of analyzed interacting TFs found in this DNA? {evidence['indirect_motifs_found_dna_str']}

### Supporting Information Details:
1. Direct Motif Analysis (Protein: {protein_name}):
    - CisBP Search Result: {evidence['direct_motif_cisbp_search_details_str']}
    - DNA Scan Results for Direct Motifs:
{evidence['direct_motif_scan_details_str']}

2. Indirect Motif Analysis (via {protein_name}'s interactors):
{evidence['indirect_motif_search_details_str']}
    - DNA Scan Results for Interacting TF Motifs:
{evidence['indirect_motif_scan_details_str']}

3. Transformer Model Prediction Details:
    - Probability: {evidence['transformer_prob_str']}
    - Protein Status (vs Training Set): {evidence['protein_status_str']}
{evidence['transformer_explanation_str']}

### Analysis Task:
Please perform a step-by-step analysis based *strictly* on the provided evidence:
1.  **Direct Evidence:** Evaluate the direct motif evidence. Does the protein have known binding motifs? Are they reliable? Were any significant hits found in the DNA sequence? Assess the strength of this evidence (strong, moderate, weak, none, or error).
2.  **Indirect Evidence:** Evaluate the indirect motif evidence. Was the search performed? Did the protein interact with known TFs? Were any motifs belonging to these interacting TFs found in the DNA sequence? Assess the strength of this evidence, considering it's less direct than the protein's own motifs.
3.  **Transformer Evidence:** Evaluate the Transformer prediction. What is the probability score? How confident does the model seem (High/Moderate/Low)? Does the protein's novelty affect your trust in this prediction?
4.  **Synthesis (Prioritizing Transformer):** Synthesize the evidence. **Crucially, give significant weight to the Transformer prediction, especially if its confidence is High or Moderate.** If direct/indirect motif evidence is weak, absent, or contradicts a confident Transformer prediction, the Transformer evidence should generally take precedence, particularly for Known proteins. Explicitly state how you are weighing the evidence based on this prioritization. Note any remaining conflicts.
5.  **Conclusion:** Based on your prioritized synthesis, state your final prediction (0 or 1).

### MANDATORY Output Format:
Explanation: [Your detailed step-by-step reasoning following points 1-5 above, clearly showing the prioritized weighing]
The interaction label is: [0 or 1]"""
    return prompt

def generate_motif_priority_llm_prompt(state: AgentState) -> str:
    protein_name = state.get('protein_name', 'N/A')
    evidence = _format_common_evidence_for_prompt(state)

    prompt = f"""Analyze the potential interaction between the DNA sequence and the Protein '{protein_name}' ({TARGET_SPECIES}). Your goal is to predict if an interaction occurs (1) or not (0), based *only* on the evidence provided below.

### Input Data:
Protein: {protein_name}
DNA {evidence['dna_info_str']}: {evidence['disp_dna_str']}

### Evidence Summary:
- Protein Novelty (vs Transformer Training): {evidence['protein_status_str']}
- Transformer Prediction Probability: {evidence['transformer_prob_str']} (Confidence: {evidence['transformer_confidence_str']})
- Direct Motif Evidence (Protein's own motifs):
    - Protein has known motifs? {evidence['direct_protein_has_motifs_str']} (Reliability: {evidence['direct_reliability_str']})
    - Direct motifs found in this DNA sequence? {evidence['direct_motif_found_dna_str']}
- Indirect Motif Evidence (via Protein-Protein Interaction):
    - Indirect Search Performed? {evidence['indirect_search_performed_str']} {f"(Skipped: {evidence['indirect_skipped_reason_str']})" if evidence['indirect_skipped_reason_str'] and evidence['indirect_search_performed_str'] == 'False' else ''}
    - Interacting TFs analyzed? {evidence['indirect_tfs_analyzed_count_str']}
    - Motifs of analyzed interacting TFs found in this DNA? {evidence['indirect_motifs_found_dna_str']}

### Supporting Information Details:
1. Direct Motif Analysis (Protein: {protein_name}):
    - CisBP Search Result: {evidence['direct_motif_cisbp_search_details_str']}
    - DNA Scan Results for Direct Motifs:
{evidence['direct_motif_scan_details_str']}

2. Indirect Motif Analysis (via {protein_name}'s interactors):
{evidence['indirect_motif_search_details_str']}
    - DNA Scan Results for Interacting TF Motifs:
{evidence['indirect_motif_scan_details_str']}

3. Transformer Model Prediction Details:
    - Probability: {evidence['transformer_prob_str']}
    - Protein Status (vs Training Set): {evidence['protein_status_str']}
{evidence['transformer_explanation_str']}

### Analysis Task:
Please perform a step-by-step analysis based *strictly* on the provided evidence:
1.  **Direct Evidence:** Evaluate the direct motif evidence. Does the protein have known binding motifs? Are they reliable? Were any significant hits found in the DNA sequence? Consider the number of hits and their p-values/scores. Assess the strength of this evidence (strong, moderate, weak, none, or error).
2.  **Indirect Evidence:** Evaluate the indirect motif evidence. Was the search performed? Did the protein interact with known TFs? Were any motifs of these interacting TFs found in the DNA? Consider hit quantity and quality. Assess the strength of this evidence.
3.  **Transformer Evidence:** Evaluate the Transformer prediction. What is the probability? How confident is it? How does protein novelty affect trust?
4.  **Synthesis (Prioritizing Motif Evidence):** Synthesize all evidence. **Crucially, give significant weight to direct and indirect motif evidence.** If the Transformer prediction conflicts with strong or clear motif evidence (e.g., multiple high-quality direct hits), the motif evidence should generally take precedence. If motif evidence is weak, absent, or ambiguous, the Transformer prediction can be more influential. Explicitly state how you are weighing the evidence based on this prioritization. Note any remaining conflicts.
5.  **Conclusion:** Based on your prioritized synthesis, state your final prediction (0 or 1).

### MANDATORY Output Format:
Explanation: [Your detailed step-by-step reasoning following points 1-5 above, clearly showing the prioritized weighing]
The interaction label is: [0 or 1]"""
    return prompt
 

def generate_llm_prompt_node(state: AgentState,
                             requested_prompt_style: str,
                             llm_identifier: str, # Full identifier including prefix, e.g., hf/...
                             finetuned_model_id: str # The specific ID requiring concise prompt
                             ) -> AgentState:
    """Node to generate the LLM prompt based on selected style and model constraints."""
    if state.get("error"): return state

    effective_prompt_style = requested_prompt_style
    is_finetuned_model = False

    # Check if the selected model is the specific fine-tuned one
    if llm_identifier.startswith("hf/"):
        model_path_or_id = llm_identifier.split("hf/", 1)[1]
        if model_path_or_id == finetuned_model_id:
            is_finetuned_model = True
            if requested_prompt_style != "concise":
                print(f"  WARN: Requested '{requested_prompt_style}' prompt style, but overriding to 'concise' for fine-tuned model '{finetuned_model_id}'.")
                effective_prompt_style = "concise"

    print(f"\n--- Node: Generate LLM Prompt (Style: {effective_prompt_style}) ---")

    try:
        if effective_prompt_style == "verbose":
            prompt = generate_verbose_llm_prompt(state)
        elif effective_prompt_style == "concise":
            prompt = generate_concise_llm_prompt(state)
        elif effective_prompt_style == "transformer-priority":
            prompt = generate_transformer_priority_llm_prompt(state)               
        elif effective_prompt_style == "motif-priority":
            prompt = generate_motif_priority_llm_prompt(state)               
        else:
            # Should not happen if argparse choices are set correctly
            raise ValueError(f"Invalid prompt style requested: {effective_prompt_style}")

        print(f"  Generated {effective_prompt_style} LLM Prompt.")
        return {**state, "llm_prompt": prompt}
    except Exception as e:
        print(f"  Critical Error generating {effective_prompt_style} prompt: {e}")
        traceback.print_exc()
        return {**state, "error": f"LLM Prompt Generation Error ({effective_prompt_style}): {e}"}


def get_llm_pred_node(state: AgentState,
                      llm_model_name: str,
                      ollama_api_url: str,
                      hf_model, hf_tokenizer, device,
                      api_delay: float # Added api_delay argument
                     ) -> AgentState:
    """Node to get the LLM prediction and store parsed + raw results."""
    print("\n--- Node: Get LLM Prediction ---")
    if state.get("error"): return state
    prompt = state.get("llm_prompt")
    if not prompt:
        return {**state, "error": "LLM prompt not generated", "llm_vote": None, "llm_explanation": "Error: Prompt generation failed.", "raw_llm_response": ""}

    # Initialize return values
    vote, explanation, raw_text = None, "Error: LLM call did not execute.", ""

    try:
        if llm_model_name.startswith("gemini/"):
            gemini_model_id = llm_model_name.split('/', 1)[1]
            vote, explanation, raw_text = call_gemini_model(prompt, gemini_model_id, api_delay)
        elif llm_model_name.startswith("ollama/"):
            ollama_model_tag = llm_model_name.split('/', 1)[1]
            vote, explanation, raw_text = call_ollama_model(prompt, ollama_model_tag, ollama_api_url)
        elif llm_model_name.startswith("hf/"):
            if not TRANSFORMERS_AVAILABLE and not UNSLOTH_AVAILABLE: raise RuntimeError("HF Transformers/Unsloth library unavailable.")
            vote, explanation, raw_text = call_huggingface_model(prompt, hf_model, hf_tokenizer, device)
        else:
            raise ValueError(f"Unsupported LLM model prefix: {llm_model_name}")

        # Store all three results in the state
        return {**state, "llm_vote": vote, "llm_explanation": explanation, "raw_llm_response": raw_text}

    except Exception as e:
        print(f"  Critical Error in get_llm_pred_node ({llm_model_name}): {e}")
        traceback.print_exc()
        # Store error message in explanation and empty raw text
        return {**state, "error": f"LLM Prediction Node Error: {e}", "llm_vote": None, "llm_explanation": f"Error during LLM call: {e}", "raw_llm_response": ""}

def calculate_final_confidence_node(state: AgentState, prompt_style: str) -> AgentState:
    """Node to calculate and store the final confidence score."""
    if state.get("error"):
        print("--- Skipping Confidence Calculation due to previous error ---")
        return {**state, "final_confidence": 0.0} # Assign low confidence on error
    try:
        # --- Add Check for Garbage Explanation ---
        is_garbage = False
        min_expected_len = 100 # Minimum chars expected for a decent explanation
        llm_explanation = state.get('llm_explanation')
        if not llm_explanation or len(llm_explanation) < min_expected_len:
            is_garbage = True
            print("  WARN: Explanation is missing or too short. Flagging as garbage.")
        # Simple check for excessive numerical repetition (adjust pattern if needed)
        elif re.search(r"(\b\d\b\W*?){20,}", llm_explanation): # Matches 20+ single digits with optional spaces/punct
            is_garbage = True
            print("  WARN: Detected excessive numerical repetition. Flagging as garbage.")
        elif "[your detailed reasoning]" in llm_explanation.lower():
            is_garbage = True
            print("  WARN: Detected template placeholder text. Flagging as garbage.")
        # Add more checks if needed (e.g., for specific repetitive phrases)

        if is_garbage:
            consistency_warning = "[WARNING: LLM explanation appears incomplete or corrupted] "
            print(f"  {consistency_warning}")
            # Override confidence to near zero
            final_confidence = 0.01
            updated_explanation = consistency_warning + (llm_explanation or "")
            # Maybe invalidate vote too? Or keep parsed vote with near-zero confidence.
            # Let's keep the vote for now but signal low confidence.
            original_llm_vote = state.get("llm_vote") # Use original vote in output dict
            return {**state, "final_confidence": final_confidence, "llm_vote": original_llm_vote, "llm_explanation": updated_explanation}
        # --- End Garbage Check ---

        confidence = calculate_confidence_score(state, prompt_style)
        return {**state, "final_confidence": confidence}
    except Exception as e:
        print(f"  ERROR calculating confidence score: {e}", file=sys.stderr)
        traceback.print_exc()
        # Assign baseline confidence if calculation fails
        error_baseline = TP_CONFIDENCE_BASELINE if prompt_style == "transformer-priority" else CONFIDENCE_BASELINE
        return {**state, "final_confidence": error_baseline, "error": f"Confidence Calculation Error: {e}"}

def make_serializable(obj):
    """Recursively converts non-JSON serializable objects in dicts/lists."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(elem) for elem in obj]
    elif isinstance(obj, set):
        return [make_serializable(elem) for elem in sorted(list(obj))] # Convert set to sorted list
    elif isinstance(obj, (np.ndarray, np.number)):
        if np.isnan(obj).any(): # Handle NaN specifically for arrays/numbers
            warnings.warn(f"Converting numpy NaN to string 'NaN' for JSON serialization.")
            if isinstance(obj, np.ndarray):
                 return np.where(np.isnan(obj), "NaN", obj).tolist()
            else: return "NaN" # Single numpy NaN
        elif np.isinf(obj).any():
             warnings.warn(f"Converting numpy Inf to string 'Infinity'/' -Infinity' for JSON.")
             if isinstance(obj, np.ndarray):
                 return np.where(np.isposinf(obj), "Infinity", np.where(np.isneginf(obj), "-Infinity", obj)).tolist()
             else: return "Infinity" if np.isposinf(obj) else "-Infinity" # Single numpy Inf
        return obj.tolist() if isinstance(obj, np.ndarray) else obj.item() # Convert numpy numbers to standard types
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        # Convert NaN/Inf within pandas objects before converting to dict
        obj = obj.replace([np.inf, -np.inf], ["Infinity", "-Infinity"]).fillna("NaN")
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        else: # Series
            return obj.to_dict()
    elif isinstance(obj, (float, int)) and (math.isnan(obj) or math.isinf(obj)):
         warnings.warn(f"Converting float {obj} to string for JSON serialization.")
         return str(obj) # Convert standard float NaN/Inf to string
    elif hasattr(obj, 'isoformat'): # Handle datetime objects if they appear
        return obj.isoformat()
    elif isinstance(obj, torch.device):
         return str(obj) # Convert torch.device to string
    # Add other specific type checks here if needed (e.g., custom classes)
    # Basic types (str, int, float (non-nan/inf), bool, None) pass through
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback: Convert unknown types to string, with a warning
        warnings.warn(f"Converting unrecognized type {type(obj)} to string for JSON: {str(obj)[:100]}...")
        return str(obj)


# --- Conditional Edges ---
# ... (Keep decide_direct_motif_path, decide_indirect_path, decide_indirect_motif_path exactly as in test3.py) ...
def decide_direct_motif_path(state: AgentState) -> str:
    print("\n--- Edge Logic: decide_direct_motif_path ---");
    if state.get("error"): return "error_path"; 
    fetch_results = state.get("direct_motif_fetch_results", {})
    if fetch_results.get('status', '').startswith('success') and fetch_results.get('motifs_metadata'): print("  --> Direct motifs found, proceed to scan DNA."); return "scan_direct"
    else: print(f"  --> No direct motifs found/fetched (status: {fetch_results.get('status', 'unknown')}), skip direct scan."); return "summarize_direct"

def decide_indirect_path(state: AgentState) -> str:
    print("\n--- Edge Logic: decide_indirect_path ---"); 
    if state.get("error"): return "error_path"; 
    direct_summary = state.get("direct_motif_analysis_summary", {}); direct_protein_info = direct_summary.get("protein_motif_info", {}); direct_fetch_status = direct_protein_info.get('status', 'unknown');
    if direct_fetch_status.startswith('success') and direct_protein_info.get('num_selected', 0) > 0:
        print("  --> Direct motifs were successfully fetched for protein. Skipping Indirect analysis.")
        indirect_summary_skipped = {"search_performed": False, "skipped_reason": "Direct motifs were successfully fetched for the protein.", "interactors_found_string": [], "interactors_tf_cisbp": [], "interactors_tf_analyzed": [], "interacting_tf_hits": [], "best_interacting_tf_hit": None, "any_indirect_motif_found": False, "scan_status": "skipped", "scan_message": None, "error_message": None}
        state["indirect_motif_analysis_summary"] = indirect_summary_skipped # Update state directly
        return "skip_indirect"
    else: print(f"  --> Direct motif fetch did not succeed or found no motifs (Status: {direct_fetch_status}). Proceeding to Indirect analysis."); return "start_indirect"

def decide_indirect_motif_path(state: AgentState) -> str:
    print("\n--- Edge Logic: decide_indirect_motif_path ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return "error_path"; 
    tfs_to_analyze = state.get("tfs_analyzed_indirectly", [])
    if not tfs_to_analyze: print("  --> No relevant interacting TFs found to analyze. Skip indirect scan."); return "summarize_indirect"
    fetch_results = state.get("indirect_motif_fetch_results", {})
    if fetch_results.get('motifs_metadata'): print("  --> Indirect TFs identified and motifs fetched, proceed to scan DNA."); return "scan_indirect"
    else: print(f"  --> Indirect TFs identified, but failed/found no motifs (status: {fetch_results.get('status', 'unknown')}). Skip indirect scan."); return "summarize_indirect"

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DNA-Protein Interaction Prediction Agent.")
    # --- Evaluation File Argument ---
    parser.add_argument(
        "--evaluation-file", type=str, required=True,
        help="Path to the evaluation dataset file (.tsv or .pkl). Required columns: 'dna', 'protein', 'label'."
    )
    parser.add_argument(
        "--llm-model", type=str,
        # Default to a small, fast HF model if transformers are available, else Gemini
        default=f"hf/{DEFAULT_HF_MODEL}" if TRANSFORMERS_AVAILABLE else f"gemini/{DEFAULT_GEMINI_MODEL}",
        help=f"LLM ID. Use prefix 'gemini/', 'ollama/', or 'hf/'. For 'hf/', provide Hub ID or LOCAL PATH (e.g., 'hf//path/to/your/model'). Default: HF if available, else Gemini."
    )
    parser.add_argument(
        "--ollama-url", type=str, default=DEFAULT_OLLAMA_API_URL,
        help=f"URL for the Ollama API endpoint if using an ollama/ model. Default: {DEFAULT_OLLAMA_API_URL}"
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="Force using CPU even if CUDA is available (useful for testing or low VRAM)."
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="Hugging Face API token (optional, needed for gated models)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./dpi_agent_outputs",
        help="Directory to save output JSON files. Default: ./dpi_agent_outputs"
    )
    parser.add_argument(
        "--prompt-style", type=str, default="verbose", choices=["verbose", "concise", "transformer-priority", "motif-priority"],
        help="Style of prompt to generate for the LLM. 'concise' is required for the specific fine-tuned model."
    )
    parser.add_argument("--random-state", type=int, default=13, help="random state for randomization of the evaluation file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N samples from the evaluation file.")
    parser.add_argument(
        "--api-delay", type=float, default=1.5, # Default to 1.5 seconds
        help="Delay (in seconds) between Gemini API calls to avoid rate limits. Default: 1.5"
    )
    args = parser.parse_args()

    # --- Create Output Directory ---
    # Include LLM name and prompt style in output subdir for organization
    selected_llm_safe = re.sub(r'[^\w\-_\.]', '_', args.llm_model.replace('/', '_'))
    output_subdir_name = f"{selected_llm_safe}_prompt_{args.prompt_style}"
    output_dir = os.path.join(args.output_dir, output_subdir_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except OSError as e:
        print(f"ERROR: Could not create output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Determine Device ---
    if args.force_cpu:
        DEVICE = torch.device('cpu')
        print("Forcing CPU usage.")
    else:
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Load DNABERT Model & Tokenizer ---
    print("--- Loading DNABERT Resources for Embeddings ---") # Modified print
    dnabert_model = None
    dnabert_tokenizer = None
    try:
        print(f"Loading DNABERT tokenizer from Hugging Face Hub: zhihan1996/DNA_bert_6")
        dnabert_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
        
        print(f"Loading DNABERT model (BertModel) from Hugging Face Hub: zhihan1996/DNA_bert_6")
        dnabert_model = BertModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True) # Changed from local path + config
        
        dnabert_model.to(DEVICE) # Move DNABERT to the determined device
        dnabert_model.eval()
        print(f"DNABERT {DNABERT_KMER}-mer model and tokenizer for embeddings loaded successfully from Hub to {DEVICE}.")
    except Exception as e:
        print(f"ERROR loading DNABERT model/tokenizer for embeddings from Hugging Face Hub: {e}", file=sys.stderr)
        traceback.print_exc()
        # Set to None so the agent knows they aren't available
        dnabert_model = None
        dnabert_tokenizer = None


    # --- Validate LLM Choice & Dependencies ---
    selected_llm = args.llm_model
    ollama_api_url_arg = args.ollama_url
    hf_model_id = None
    hf_model = None
    hf_tokenizer = None

    print(f"Selected LLM Identifier: {selected_llm}")

    if selected_llm.startswith("gemini/"):
        if not GOOGLE_API_KEY_CONFIGURED:
            print("ERROR: Gemini model selected, but GOOGLE_API_KEY is not set. Exiting.", file=sys.stderr); sys.exit(1)
    elif selected_llm.startswith("ollama/"):
        print(f"Using Ollama API URL: {ollama_api_url_arg}")
    elif selected_llm.startswith("hf/"):
        if not TRANSFORMERS_AVAILABLE and not UNSLOTH_AVAILABLE: # Need at least one
            print("ERROR: HF model selected, but neither 'transformers' nor 'unsloth' library is available.", file=sys.stderr); sys.exit(1)
        hf_model_id = selected_llm.split('/', 1)[1] # hf_model_id can be Hub ID or local path
        print(f"Attempting to load HF model/path: {hf_model_id}")
        try:
            # Load tokenizer first (standard way)
            print("Loading tokenizer...");
            hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=args.hf_token)
            print("Tokenizer loaded.")

            print("Loading model...")
            load_start_time = time.time()
            model_loaded_by_unsloth = False

            # --- PRIORITIZE Unsloth Loading if available ---
            if UNSLOTH_AVAILABLE and DEVICE.type == 'cuda':
                print("Attempting optimized loading with Unsloth...")
                try:
                    # Assume 4bit, float16/bfloat16, max_seq_len - adjust if needed
                    hf_model, hf_tokenizer = unsloth.FastLanguageModel.from_pretrained(
                        model_name = hf_model_id,
                        # max_seq_length = 2048, # Optional: Set based on model/needs
                        dtype = None, # Let Unsloth choose
                        load_in_4bit = True, # Assume 4-bit for large models
                        token = args.hf_token,
                        device_map = "auto"
                    )
                    print("Successfully loaded model with Unsloth.")
                    model_loaded_by_unsloth = True
                    # Optional: Check if PEFT adapters need loading/merging if it's a PEFT model
                    # if hasattr(hf_model, "merge_and_unload"): # Example check
                    #    print("Attempting PEFT adapter merge...")
                    #    hf_model = hf_model.merge_and_unload()

                except Exception as e_unsloth:
                     print(f"WARN: Unsloth loading failed ({e_unsloth}). Will attempt standard Transformers loading.")
                     hf_model = None # Ensure fallback

            # --- Fallback to Standard Transformers Loading ---
            if not model_loaded_by_unsloth:
                if not TRANSFORMERS_AVAILABLE: # Should have exited above, but double-check
                     print("ERROR: Standard Transformers library not available, cannot load model.", file=sys.stderr); sys.exit(1)

                print("Using standard Hugging Face transformers loading...")
                quantization_config = None
                if DEVICE.type == 'cuda':
                     print("Configuring standard 4-bit quantization...")
                     quantization_config = BitsAndBytesConfig(
                         load_in_4bit=True,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True,
                     )
                elif DEVICE.type == 'cpu': print("WARN: Loading HF model on CPU.")

                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if DEVICE.type == 'cuda' and quantization_config is None else None,
                    token=args.hf_token,
                    trust_remote_code=True
                )
                print("Standard Hugging Face model loaded.")

            load_end_time = time.time()
            print(f"Model loading took {load_end_time - load_start_time:.2f} seconds.")

            # Ensure model is in eval mode
            hf_model.eval()

        except Exception as e:
            print(f"ERROR loading Hugging Face model '{hf_model_id}': {e}", file=sys.stderr)
            traceback.print_exc()
            print("Cannot proceed without the LLM. Exiting.")
            sys.exit(1)
    else:
         print(f"ERROR: Invalid --llm-model format: '{selected_llm}'. Must start with 'gemini/', 'ollama/', or 'hf/'. Exiting.", file=sys.stderr)
         sys.exit(1)

    # --- Resource Loading ---
    print("--- Loading Domain Specific Resources ---")
    # TRANSFORMER_MODEL_PATH = '/new-stg/home/cong/DPI/scripts/model2_Transformer/v5/output/model/main_singletask_Encode3and4_seed101_800_proteins-lr=0.05,epoch=2,dropout=0.2:0,hid_dim=240,n_layer=2,n_heads=6,batch=128,input=train_min10,max_dna=512,max_protein=768.pt'
    TRANSFORMER_MODEL_PATH = '/new-stg/home/cong/DPI/scripts/model2_Transformer/v5/output/model/main_singletask_Encode3and4_all_847_proteins-lr=0.05,epoch=2,dropout=0.2:0,hid_dim=240,n_layer=2,n_heads=6,batch=128,input=train_min10,max_dna=512,max_protein=768,mixed_precision=False.pt'

    DNA_EMB_DB_PATH = '/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/embeddings/valid_min10/dna_embeddings.duckdb'
    PRO_EMB_PATH = '/new-stg/home/cong/DPI/dataset/all_human_tfs_protein_embedding_mean_deduplicated.pkl'

    TRAINING_PROTEIN_LIST_PATH = '/new-stg/home/cong/DPI/dataset/Encode3and4/proteins_with_embed.txt'

    CISBP_BASE_DIR = '/new-stg/home/cong/DPI/scripts/deepseek/motif_database/'
    CISBP_TF_INFO_VARIANT = 'standard'
    cisbp_tf_info_df = None; cisbp_pwm_dir = None
    try:
        if os.path.isdir(CISBP_BASE_DIR):
             cisbp_pwm_dir = os.path.join(CISBP_BASE_DIR, 'pwms_all_motifs'); # ... (rest of CisBP loading)
             if not os.path.isdir(cisbp_pwm_dir): raise FileNotFoundError(f"PWM directory not found: {cisbp_pwm_dir}")
             tf_info_filename = 'TF_Information.txt'; # ... (handle variants)
             if CISBP_TF_INFO_VARIANT == 'all_motifs': tf_info_filename = 'TF_Information_all_motifs.txt'
             elif CISBP_TF_INFO_VARIANT == 'all_motifs_plus': tf_info_filename = 'TF_Information_all_motifs_plus.txt'
             elif CISBP_TF_INFO_VARIANT != 'standard': raise ValueError(f"Invalid cisbp_tf_info_variant: {CISBP_TF_INFO_VARIANT}")
             cisbp_tf_info_filepath = os.path.join(CISBP_BASE_DIR, tf_info_filename)
             if not os.path.exists(cisbp_tf_info_filepath): raise FileNotFoundError(f"TF Info file not found: {cisbp_tf_info_filepath}")
             print(f"Loading CisBP TF Info from: {cisbp_tf_info_filepath}"); cisbp_tf_info_df = pd.read_csv(cisbp_tf_info_filepath, sep='\t', low_memory=False)
             original_count = len(cisbp_tf_info_df); cisbp_tf_info_df = cisbp_tf_info_df[cisbp_tf_info_df['TF_Species'] == TARGET_SPECIES].copy()
             print(f"Loaded and filtered {len(cisbp_tf_info_df)} CisBP entries for {TARGET_SPECIES} (from {original_count}).")
        else: raise FileNotFoundError(f"CisBP base directory not found: {CISBP_BASE_DIR}")
    except Exception as e: print(f"FATAL ERROR loading CisBP data: {e}", file=sys.stderr); sys.exit(1)
    # Protein Embeddings
    pro_embs_dict = None; pro_embs_df = None; # ... (Keep protein embedding loading with duplicate removal)
    try:
        if os.path.exists(PRO_EMB_PATH):
            print(f"Loading protein embeddings file: {PRO_EMB_PATH}"); pro_embs_df = pd.read_pickle(PRO_EMB_PATH); required_cols = ['protein', 'protein_embedding']
            if not all(col in pro_embs_df.columns for col in required_cols): print(f"ERROR: Protein embedding file {PRO_EMB_PATH} missing required columns: {required_cols}", file=sys.stderr)
            else:
                protein_counts = pro_embs_df['protein'].value_counts(); duplicate_proteins = protein_counts[protein_counts > 1].index.tolist()
                if duplicate_proteins:
                    print(f"WARN: Found {len(duplicate_proteins)} protein names with multiple entries. Removing ALL entries for these proteins."); print(f"  Sample duplicate names being removed: {duplicate_proteins[:10]} {'...' if len(duplicate_proteins) > 10 else ''}")
                    filtered_df = pro_embs_df[~pro_embs_df['protein'].isin(duplicate_proteins)].copy(); print(f"  Original count: {len(pro_embs_df)}, Filtered count: {len(filtered_df)}")
                    if not filtered_df.empty: pro_embs_dict = filtered_df.set_index('protein')['protein_embedding'].to_dict(); print(f"Loaded {len(pro_embs_dict)} protein embeddings after removing duplicates.")
                    else: print("WARN: No embeddings remaining after removing all duplicates."); pro_embs_dict = {}
                else:
                    print("No duplicate protein names found.");
                    if pro_embs_df['protein'].is_unique: pro_embs_dict = pro_embs_df.set_index('protein')['protein_embedding'].to_dict(); print(f"Loaded {len(pro_embs_dict)} protein embeddings.")
                    else: print("ERROR: Inconsistency detected - duplicates list empty but column not unique.", file=sys.stderr); pro_embs_dict = None
        else: print(f"WARN: Protein embeddings file not found: {PRO_EMB_PATH}", file=sys.stderr)
    except Exception as e: print(f"ERROR loading protein embeddings: {e}", file=sys.stderr); traceback.print_exc(); pro_embs_dict = None
    # DNA Embeddings
    dna_db_con = None; # ... (Keep DNA embedding loading)
    try:
        if os.path.exists(DNA_EMB_DB_PATH): dna_db_con = duckdb.connect(database=DNA_EMB_DB_PATH, read_only=True); print(f"Connected to DNA embeddings DB: {DNA_EMB_DB_PATH}")
        else: print(f"WARN: DNA embedding DB not found: {DNA_EMB_DB_PATH}", file=sys.stderr)
    except Exception as e: print(f"ERROR connecting to DNA embeddings DB: {e}", file=sys.stderr)
    # Training Protein List
    training_protein_set = set(); # ... (Keep training protein list loading)
    try:
        if os.path.exists(TRAINING_PROTEIN_LIST_PATH):
             with open(TRAINING_PROTEIN_LIST_PATH, 'r') as f: training_proteins = [line.strip() for line in f if line.strip()]
             training_protein_set = set(training_proteins); print(f"Loaded {len(training_protein_set)} training proteins.")
        else: print(f"WARN: Training protein list not found: {TRAINING_PROTEIN_LIST_PATH}", file=sys.stderr)
    except Exception as e: print(f"ERROR loading training protein list: {e}", file=sys.stderr)
    # Transformer Model
    transformer_model = None; transformer_config = None; # ... (Keep transformer loading)
    try:
        transformer_config = eval_get_config(); keys_to_remove = ['warmup', 'iteration_per_split', 'files_per_split', 'valid_dna_size', 'lr'];
        for key in keys_to_remove: transformer_config.pop(key, None)
        transformer_config['return_attention'] = False; transformer_config['device'] = DEVICE; # Use determined device
        matches = re.findall(r'([a-zA-Z_]+)=([\d.]+)', os.path.basename(TRANSFORMER_MODEL_PATH)); parsed = {key: (int(v) if v.isdigit() else float(v) if '.' in v else v) for key, v in matches}
        if 'max_dna' in parsed and 'max_dna_seq' not in parsed: parsed['max_dna_seq'] = parsed.pop('max_dna')
        if 'max_protein' in parsed and 'max_protein_seq' not in parsed: parsed['max_protein_seq'] = parsed.pop('max_protein')
        transformer_config.update(parsed); print(f"Using Transformer Config: {transformer_config}")
        transformer_model = Predictor(**transformer_config)
        state_dict = torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE) # Map to determined device
        if any(key.startswith('module.') for key in state_dict.keys()): print("Adjusting state_dict keys (removing 'module.' prefix)..."); new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()]); transformer_model.load_state_dict(new_state_dict)
        else: transformer_model.load_state_dict(state_dict)
        transformer_model.to(DEVICE); transformer_model.eval(); print("Transformer model loaded.")
    except Exception as e: print(f"ERROR loading Transformer model: {e}", file=sys.stderr); traceback.print_exc()

    print("--- Resource Loading Complete ---")

    # Check essential resources
    if cisbp_tf_info_df is None or not cisbp_pwm_dir or pro_embs_dict is None or dna_db_con is None or transformer_model is None:
         print("\nFATAL ERROR: One or more essential domain-specific resources failed to load. Cannot proceed.", file=sys.stderr); sys.exit(1)
    if selected_llm.startswith("hf/") and (hf_model is None or hf_tokenizer is None):
        print("\nFATAL ERROR: Failed to load the selected Hugging Face LLM. Cannot proceed.", file=sys.stderr); sys.exit(1)

    # --- Load, RENAME, Shuffle, and Prepare Evaluation Data ---
    print(f"--- Loading Evaluation Data from: {args.evaluation_file} ---")
    eval_inputs = []
    try:
        if args.evaluation_file.endswith('.pkl'):
            df_eval_raw = pd.read_pickle(args.evaluation_file)
            print(f"  Loaded {len(df_eval_raw)} raw samples from pickle.")
        elif args.evaluation_file.endswith('.tsv'):
            df_eval_raw = pd.read_csv(args.evaluation_file, sep='\t')
            print(f"  Loaded {len(df_eval_raw)} raw samples from tsv.")
        else:
            raise ValueError("Unsupported evaluation file format. Use .pkl or .tsv")

        # --- Rename columns ---
        rename_map = {'dna': 'dna_sequence', 'protein': 'protein_name', 'label': 'ground_truth_label'}
        required_original_cols = list(rename_map.keys())
        if not all(col in df_eval_raw.columns for col in required_original_cols):
             raise ValueError(f"Evaluation file missing required original columns: {required_original_cols}")

        df_eval = df_eval_raw.rename(columns=rename_map)
        print(f"  Renamed columns to: {list(df_eval.columns)}")

        # --- Shuffle DataFrame ---
        print(f"  Shuffling dataset with random_state={args.random_state}...")
        df_eval = df_eval.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
        print("  Dataset shuffled.")

        # Apply limit if specified (AFTER shuffling)
        if args.limit is not None and args.limit >= 0: # Allow limit=0
             if args.limit < len(df_eval):
                  df_eval = df_eval.head(args.limit)
                  print(f"  Limiting processing to first {args.limit} shuffled samples.")
             else:
                  print(f"  Limit ({args.limit}) is >= dataset size ({len(df_eval)}), processing all shuffled samples.")
        else:
             print(f"  Processing all {len(df_eval)} shuffled samples.")

        # Convert to list of dictionaries
        final_required_cols = list(rename_map.values()) # ['dna_sequence', 'protein_name', 'ground_truth_label']
        eval_inputs = df_eval[final_required_cols].to_dict('records')

    except FileNotFoundError: print(f"ERROR: Evaluation file not found: {args.evaluation_file}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"ERROR loading or parsing evaluation file: {e}", file=sys.stderr); sys.exit(1)

    if not eval_inputs and args.limit != 0: # Check if list is empty but limit wasn't 0
         print("ERROR: No evaluation samples loaded after processing. Exiting.", file=sys.stderr); sys.exit(1)
    elif args.limit == 0:
         print("Limit set to 0. No samples will be processed.")

    # --- Build Graph ---
    workflow = StateGraph(AgentState)

    # Add nodes, passing resources via partial
    # Domain-specific resources
    workflow.add_node("start", start_analysis)
    workflow.add_node("fetch_direct_motifs", partial(fetch_direct_motifs_node, 
                                                     cisbp_tf_info_df=cisbp_tf_info_df, 
                                                     cisbp_pwm_dir=cisbp_pwm_dir))
    workflow.add_node("scan_direct_dna", scan_direct_dna_node)
    workflow.add_node("summarize_direct_analysis", summarize_direct_analysis_node)
    workflow.add_node("query_string", query_string_node)
    workflow.add_node("filter_string_interactors", partial(filter_string_interactors_node, cisbp_tf_info_df=cisbp_tf_info_df))
    workflow.add_node("fetch_indirect_motifs", partial(fetch_indirect_motifs_node, cisbp_tf_info_df=cisbp_tf_info_df, cisbp_pwm_dir=cisbp_pwm_dir))
    workflow.add_node("scan_indirect_dna", scan_indirect_dna_node)
    workflow.add_node("summarize_indirect_analysis", summarize_indirect_analysis_node)

    transformer_pred_partial = partial(
        get_transformer_pred_node_wrapper, # Target the wrapper
        # Bind all necessary resources to the wrapper
        transformer_model=transformer_model,
        pro_embs_dict=pro_embs_dict,
        dna_db_con=dna_db_con,
        transformer_config=transformer_config,
        device=DEVICE,
        training_protein_set=training_protein_set,
        dnabert_model=dnabert_model,
        dnabert_tokenizer=dnabert_tokenizer,
        dnabert_kmer=DNABERT_KMER,
        dnabert_max_len=DNABERT_MAX_LEN,
        dnabert_device=DEVICE
    )
    workflow.add_node("get_transformer_pred", transformer_pred_partial)

    workflow.add_node("generate_llm_prompt", partial(
        generate_llm_prompt_node,
        requested_prompt_style=args.prompt_style, # Pass the user's choice
        llm_identifier=selected_llm,              # Pass the selected LLM identifier
        finetuned_model_id=FINETUNED_LOCAL_MODEL_PATH_ID # Pass the ID to check against
    ))
    # LLM resources passed to the LLM node
    workflow.add_node("get_llm_pred", partial(get_llm_pred_node,
                                              llm_model_name=selected_llm,
                                              ollama_api_url=ollama_api_url_arg,
                                              hf_model=hf_model,
                                              hf_tokenizer=hf_tokenizer,
                                              device=DEVICE,
                                              api_delay=args.api_delay # Pass the delay from args
                                              ))
    # NEW confidence node
    workflow.add_node("calculate_final_confidence", partial(calculate_final_confidence_node, prompt_style=args.prompt_style))
    # Error node now sets default confidence
    workflow.add_node("handle_error", lambda state: {**state, "llm_vote": None, "final_confidence": 0.0, "llm_explanation": f"Processing stopped due to error: {state.get('error', 'Unknown error')}"})

    # Define Edges (UPDATED FLOW)
    workflow.set_entry_point("start")
    workflow.add_edge("start", "fetch_direct_motifs")
    workflow.add_conditional_edges("fetch_direct_motifs", decide_direct_motif_path, 
                                   {"scan_direct": "scan_direct_dna", 
                                    "summarize_direct": "summarize_direct_analysis", 
                                    "error_path": "handle_error"})
    workflow.add_edge("scan_direct_dna", "summarize_direct_analysis")
    workflow.add_conditional_edges("summarize_direct_analysis", decide_indirect_path, 
                                   {"start_indirect": "query_string", 
                                    "skip_indirect": "get_transformer_pred", 
                                    "error_path": "handle_error"})
    workflow.add_edge("query_string", "filter_string_interactors")
    workflow.add_edge("filter_string_interactors", "fetch_indirect_motifs")
    workflow.add_conditional_edges("fetch_indirect_motifs", decide_indirect_motif_path, {"scan_indirect": "scan_indirect_dna", "summarize_indirect": "summarize_indirect_analysis", "error_path": "handle_error"})
    workflow.add_edge("scan_indirect_dna", "summarize_indirect_analysis")
    workflow.add_edge("summarize_indirect_analysis", "get_transformer_pred")
    workflow.add_edge("get_transformer_pred", "generate_llm_prompt")
    workflow.add_edge("generate_llm_prompt", "get_llm_pred")
    workflow.add_edge("get_llm_pred", "calculate_final_confidence")
    workflow.add_edge("calculate_final_confidence", END)
    workflow.add_edge("handle_error", END)

    # Compile the graph
    print("--- Compiling LangGraph Workflow ---")
    app = workflow.compile()
    print("--- Workflow Compiled ---")

    # --- Run Evaluation Loop & Save Outputs ---
    print(f"\n--- Running Evaluation (LLM: {selected_llm}, Prompt: {args.prompt_style}) ---")

    results_summary = [] # Store simple results for potential aggregation later
    if not eval_inputs: # Handle limit=0 case
        print("No samples to process due to limit=0.")
    else:
        for i, input_data in enumerate(eval_inputs):
            protein_name = input_data.get("protein_name", f"unknown_protein_{i+1}")
            # Ensure ground truth is integer
            try: input_data['ground_truth_label'] = int(input_data['ground_truth_label'])
            except (ValueError, TypeError): print(f"WARN: Invalid ground truth '{input_data.get('ground_truth_label')}' for {protein_name}. Skipping sample {i+1}.", file=sys.stderr); continue

            print(f"\n--- Processing Sample {i+1}/{len(eval_inputs)} [{protein_name}] ---")
            start_time = time.time(); final_state = None

            # --- Define output paths with LLM name and index ---
            safe_protein_name = re.sub(r'[^\w\-_\.]', '_', protein_name)
            run_identifier = f"sample{i+1:04d}" # Use padded index for sorting
            llm_name_safe = re.sub(r'[^\w\-_\.]', '_', selected_llm.replace('/', '_'))
            # Output files go into the specific subdirectory for this run
            simple_output_path = os.path.join(output_dir, f"{run_identifier}_{safe_protein_name}_{llm_name_safe}_simple.json")
            comp_output_path = os.path.join(output_dir, f"{run_identifier}_{safe_protein_name}_{llm_name_safe}_comp.json")
            # --------------------------------------------------

            try:
                # Invoke the agent with the input data (including ground truth)
                final_state = app.invoke(input_data, {"recursion_limit": 50})

                # Prepare simple output dictionary
                simple_output = {
                    "protein_name": final_state.get("protein_name"),
                    "dna_sequence_length": len(final_state.get("dna_sequence", "")),
                    "ground_truth_label": final_state.get("ground_truth_label"), # Include GT
                    "predicted_label": final_state.get("llm_vote"),
                    "confidence_score": final_state.get("final_confidence"),
                    "llm_explanation": final_state.get("llm_explanation"),
                    "error": final_state.get("error") # Will be None if successful
                }
                results_summary.append(simple_output) # Add to summary list

                # Save simple output JSON
                try:
                    with open(simple_output_path, 'w') as f: json.dump(simple_output, f, indent=4)
                    # print(f"  Simple output saved to: {simple_output_path}") # Less verbose for batch runs
                except Exception as e_save: print(f"  ERROR saving simple output JSON for {protein_name}: {e_save}", file=sys.stderr)

                # Save comprehensive output JSON (using make_serializable)
                try:
                    serializable_state = make_serializable(final_state)
                    with open(comp_output_path, 'w') as f: json.dump(serializable_state, f, indent=2)
                    # print(f"  Comprehensive output saved to: {comp_output_path}")
                except Exception as e_save: print(f"  ERROR saving comprehensive output JSON for {protein_name}: {e_save}", file=sys.stderr); traceback.print_exc(file=sys.stderr)

            # --- Handle invoke errors ---
            except Exception as e_invoke:
                error_msg = f"Agent invocation failed: {e_invoke}"
                print(f"\nFATAL ERROR during agent invocation {i+1}: {e_invoke}", file=sys.stderr); traceback.print_exc();
                final_state = {"error": error_msg, **input_data, "llm_vote": None, "final_confidence": 0.0, "llm_explanation": f"Error: {error_msg}."}; # Create partial state

                # Attempt to save minimal error state
                simple_output_error = {
                    "protein_name": input_data.get("protein_name"),
                    "dna_sequence_length": len(input_data.get("dna_sequence", "")),
                    "ground_truth_label": input_data.get("ground_truth_label"),
                    "predicted_label": None, "confidence_score": 0.0,
                    "llm_explanation": final_state.get("llm_explanation"),
                    "error": final_state.get("error")
                }
                results_summary.append(simple_output_error) # Add error entry
                try:
                    with open(simple_output_path, 'w') as f: json.dump(simple_output_error, f, indent=4)
                    # print(f"  Simple error output saved to: {simple_output_path}")
                except Exception as e_save: print(f"  ERROR saving simple error output JSON for {protein_name}: {e_save}", file=sys.stderr)
                # Optionally save comprehensive error state too if desired

            end_time = time.time()
            print(f"--- Sample {i+1} Finished (Took {end_time - start_time:.2f} seconds) ---")

            # --- Optional: Print intermediate progress ---
            if (i + 1) % 50 == 0: # Print every 50 samples
                print(f"*** Progress: Processed {i+1}/{len(eval_inputs)} samples ***")

        print(f"\nAgent evaluation run complete for {selected_llm} (Prompt: {args.prompt_style}).")
        print(f"Outputs saved in: {output_dir}")

    # --- Optional: Save aggregated simple results ---
    agg_simple_path = os.path.join(args.output_dir, f"{output_subdir_name}_aggregated_simple_results.json")
    try:
        with open(agg_simple_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Aggregated simple results saved to: {agg_simple_path}")
    except Exception as e_save:
        print(f"ERROR saving aggregated simple results: {e_save}", file=sys.stderr)


    # --- Clean up ---
    # Close DB connection
    if dna_db_con:
        try: dna_db_con.close(); print("DNA DB Connection closed.")
        except Exception as e: print(f"Error closing DNA DB: {e}")

    # Clear GPU memory potentially held by models
    if hf_model: del hf_model
    if hf_tokenizer: del hf_tokenizer
    if transformer_model: del transformer_model
    if DEVICE.type == 'cuda':
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    print("Forcing garbage collection...")
    gc.collect()
    print("Garbage collection finished.")