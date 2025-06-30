# better way to handle unknown protein when using sei and deepsea: 
# For an unknown protein:
### a. Find the N most similar known/trained proteins using sequence similarity (MMseqs2, BLAST, Levenshtein).
### b. For each of these N similar known proteins, run the SEI/DeepSEA model with the original unknown protein's DNA sequence but using the known similar protein's ID/embedding.
### c. Collect the output logits (pre-sigmoid probabilities) from these N model runs.
### d. Average these N logits.
### e. Apply the sigmoid function to this average logit to get the final probability for the unknown protein-DNA pair.
# perfect for testing unseen proteins: chip690 data

## also available for ablation study: --no-indirect-analysis. v9 doesn't have this feature

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
from tqdm import tqdm
from pathlib import Path

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
from google.api_core import exceptions # For Gemini rate limit
import requests # For Ollama API

# --- Dependencies from Hybrid Script & HF Models ---
import torch
import torch.cuda
import torch.nn.functional as F # Ensure F is imported for potential use in models
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


# --- Import DPI Model Components ---
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

# --- NEW: Import from protein_utils ---
try:
    from protein_utils import (
        ProteinDNADatasetInference,
        parse_fasta_for_gene_names,
        fetch_uniprot_sequence_by_gene_or_id,
        MMSEQS_AVAILABLE, BLAST_AVAILABLE, LEVENSHTEIN_AVAILABLE, # Tool availability flags
        dna_to_one_hot # If you moved dna_to_one_hot here too
    )
    print("INFO: Successfully imported utilities from protein_utils.py")
except ImportError as e_pu:
    print(f"ERROR: Could not import from protein_utils.py: {e_pu}", file=sys.stderr)
    print("Ensure protein_utils.py is in the Python path.", file=sys.stderr)
    sys.exit(1)

# --- Import DeepSEA Model Components--
try:
    # import torch.nn.functional as F # Already imported above
    from deepsea_v0.eval_deepsea import dna_to_one_hot, DeepSEAProteinInteraction
    print("Successfully imported Predictor and helper functions.")
except ImportError as e:
    print(f"ERROR: Could not import required components from source scripts: {e}", file=sys.stderr)
    print("Ensure eval_deepsea.py is in the Python path.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
     print(f"ERROR during import: {e}", file=sys.stderr); sys.exit(1)

PROTEIN_ALIASES = {"AP2A": "TFAP2A", "AP2C": "TFAP2C"}

# --- NEW: Import SEI Model Components ---
try:
    # MODIFIED IMPORT: Import the new class
    from sei_extended import SeiProteinInteractionWithMeanEmbedding
    # For convenience, you can alias it if the rest of the script refers to SeiProteinInteraction
    # SeiProteinInteraction = SeiProteinInteractionWithMeanEmbedding 
    print("Successfully imported SeiProteinInteractionWithMeanEmbedding model from sei_extended.py.")
except ImportError as e:
    print(f"ERROR: Could not import SeiProteinInteractionWithMeanEmbedding from sei_extended.py: {e}", file=sys.stderr)
    print("Ensure sei_extended.py is in the Python path and imports its own dependencies correctly.", file=sys.stderr)
    sys.exit(1) # Make it fatal if SEI is to be used
except Exception as e:
     print(f"ERROR during SEI (extended) import: {e}", file=sys.stderr); sys.exit(1)


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
INDIRECT_MOTIF_P_VALUE_THRESHOLD = 1e-5 # tried to reduce false positives
TARGET_SPECIES = "Homo_sapiens"
MAX_MOTIFS_PER_TF_TO_PROCESS = 3
STRING_API_URL = "https://string-db.org/api/"
STRING_OUTPUT_FORMAT = "json"
STRING_MIN_INTERACTION_SCORE = 700
MAX_STRING_INTERACTORS_TO_CHECK = 50
TARGET_SPECIES_TAXONOMY_ID = 9606
MAX_INTERACTING_TFS_TO_ANALYZE = 3 # tried to reduce false positives


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
TP_CONFIDENCE_BASELINE = 0.50

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

# --- Confidence Score Parameters (ADDITIONS/MODIFICATIONS for multi-model handling) ---
SUBSEQUENT_MODEL_AGREEMENT_FACTOR = 0.3  # Boost from 2nd, 3rd, etc., agreeing model is scaled by this
SUBSEQUENT_MODEL_MISMATCH_PENALTY_FACTOR = 0.3 # Penalty from 2nd, 3rd, etc., mismatching model is scaled by this
MIN_MODELS_FOR_CONFLICT_AMPLIFICATION = 2
MOTIF_CONFLICT_AMPLIFICATION_FACTOR = 1.2

# --- NEW: Use constants for paths and settings for unknown handling ---
# These should be defined near your other argparse defaults or global constants
FASTA_FILE_PATH_CONFIG = Path(os.environ.get("DPI_FASTA_PATH", '/new-stg/home/cong/DPI/downloads/fasta/uniprot_ChIP_690_Encode3and4_2024_03_01.fasta'))
TF_FAMILY_FILE_PATH_CONFIG = Path(os.environ.get("DPI_TF_FAMILY_PATH",'/new-stg/home/cong/DPI/dataset/all_pros_family.csv'))
SIMILARITY_ENGINE_CONFIG = os.environ.get("DPI_SIMILARITY_ENGINE", 'mmseqs')
NUM_TOP_SIMILAR_FOR_AVG_LOGITS_CONFIG = int(os.environ.get("DPI_NUM_SIMILAR_LOGITS", "3"))
MAX_LEVENSHTEIN_DIST_CONFIG = int(os.environ.get("DPI_MAX_LEV_DIST", "250"))
MIN_MMSEQS_BLAST_BITSCORE_CONFIG = int(os.environ.get("DPI_MIN_BITSCORE", "50"))
MAX_MMSEQS_BLAST_EVALUE_CONFIG = float(os.environ.get("DPI_MAX_EVALUE", "0.001"))
FETCH_MISSING_SEQUENCES_FROM_UNIPROT_CONFIG = os.environ.get("DPI_FETCH_UNIPROT", "True").lower() == "true"
UNIPROT_ORGANISM_ID_FOR_FETCH_CONFIG = os.environ.get("DPI_UNIPROT_ORG", "9606")
UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG = Path(os.environ.get("DPI_UNKNOWN_TEMP_DIR", "./unknown_handling_temp"))


GLOBAL_PROTEIN_SEQUENCES_MAP = {}
PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED = {}
MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING = {}
_dataset_inference_cache = {} 
_unknown_handling_stats_cache = {} # Moved

# --- Agent State Definition (MODIFIED) ---
class AgentState(TypedDict):
    protein_name: str
    dna_sequence: str
    ground_truth_label: Union[int, None]
    direct_motif_fetch_results: Union[dict, None]
    direct_motif_scan_results: Union[dict, None]
    direct_motif_analysis_summary: Union[dict, None]
    string_interactors: Union[List[dict], None]
    tf_interactors_cisbp: Union[List[dict], None]
    tfs_analyzed_indirectly: Union[List[dict], None]
    indirect_motif_fetch_results: Union[dict, None]
    indirect_motif_scan_results: Union[dict, None]
    indirect_motif_analysis_summary: Union[dict, None]

    models_to_run: List[str] # e.g., ["dpi", "sei"]
    all_model_predictions: List[Dict]

    llm_prompt: Union[str, None]
    llm_vote: Union[int, None]
    llm_explanation: Union[str, None]
    raw_llm_response: Union[str, None]
    final_confidence: Union[float, None]
    error: Union[str, None]
    llm_prompt_style_selected: Union[str, None]
    llm_prompt_selection_reason: Union[str, None]
    ablate_indirect_analysis: bool # Flag to control indirect analysis path
    ablate_direct_analysis: bool   # Flag to control direct analysis path


def _parse_cisbp_pwm(pwm_filepath: str) -> Union[Dict[str, List[float]], None]:
    if not os.path.exists(pwm_filepath):
        # print(f"  DEBUG: PWM file {os.path.basename(pwm_filepath)} does not exist.", file=sys.stdout) # Optional: for debugging missing files
        return None
    counts = {'A': [], 'C': [], 'G': [], 'T': []}

    try:
        with open(pwm_filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"  INFO: PWM file {os.path.basename(pwm_filepath)} is empty. No motif data extracted.", file=sys.stdout)
            return None

        actual_data_lines = []
        header_lines_skipped_count = 0
        first_line_is_pos_header = False

        if lines: # Check if lines is not empty before accessing lines[0]
            first_line_stripped = lines[0].strip()
            if first_line_stripped.upper().startswith("POS") and len(first_line_stripped.split()) >=5 : # A common CisBP header
                first_line_is_pos_header = True

        for line_idx, line_content in enumerate(lines):
            stripped_line = line_content.strip()
            if not stripped_line: # skip empty lines
                continue
            if stripped_line.startswith('>'): # MEME-like header, skip
                header_lines_skipped_count += 1
                continue
            if line_idx == 0 and first_line_is_pos_header: # CisBP header line
                header_lines_skipped_count += 1
                continue
            actual_data_lines.append(stripped_line)

        if not actual_data_lines:
            if header_lines_skipped_count > 0 or first_line_is_pos_header:
                print(f"  INFO: PWM file {os.path.basename(pwm_filepath)} contained only header-like lines. No motif data extracted.", file=sys.stdout)
            else:
                print(f"  INFO: PWM file {os.path.basename(pwm_filepath)} contains no parsable data lines after initial filtering. No motif data extracted.", file=sys.stdout)
            return None

        # Determine format from the first actual data line
        first_data_line_parts = actual_data_lines[0].split()
        is_pos_values_format = False # True if format is "Pos A C G T" (where Pos is a number or identifier)

        if len(first_data_line_parts) == 4: # Standard A C G T format
            # Try to convert to float to ensure they are numbers
            try:
                for part in first_data_line_parts: float(part)
            except ValueError:
                 raise ValueError(f"Expected A C G T numeric format, but found non-numeric values. Line: {actual_data_lines[0]}")
        elif len(first_data_line_parts) >= 5: # Potentially "Pos A C G T ..." format
            # Check if parts [1:5] are numeric
            try:
                for i in range(1, 5): float(first_data_line_parts[i])
                is_pos_values_format = True
            except ValueError:
                 raise ValueError(f"Expected Pos A C G T numeric format (cols 2-5), but found non-numeric. Line: {actual_data_lines[0]}")
        else:
            raise ValueError(f"Unrecognized PWM format. Expected 4 (A C G T) or >=5 (Pos A C G T) columns. First data line: '{actual_data_lines[0]}'")

        for line_idx_data, data_line_content in enumerate(actual_data_lines):
            parts = data_line_content.split()
            freqs_str_list = []

            if is_pos_values_format:
                if len(parts) < 5:
                    # print(f"  DEBUG: Skipping malformed data line (expected Pos A C G T) in {os.path.basename(pwm_filepath)} line {line_idx_data+1}: '{data_line_content}'", file=sys.stdout)
                    continue
                freqs_str_list = parts[1:5]
            else: # A C G T format
                if len(parts) < 4:
                    # print(f"  DEBUG: Skipping malformed data line (expected A C G T) in {os.path.basename(pwm_filepath)} line {line_idx_data+1}: '{data_line_content}'", file=sys.stdout)
                    continue
                freqs_str_list = parts[0:4]

            try:
                freqs = list(map(float, freqs_str_list))
            except ValueError:
                # print(f"  DEBUG: Non-numeric frequency in data line in {os.path.basename(pwm_filepath)} line {line_idx_data+1}: '{data_line_content}'", file=sys.stdout)
                continue

            counts['A'].append(freqs[0]); counts['C'].append(freqs[1]); counts['G'].append(freqs[2]); counts['T'].append(freqs[3])

        if not counts['A']: # Check if any valid rows were processed
            print(f"  INFO: PWM file {os.path.basename(pwm_filepath)} resulted in zero-length motif after processing data lines (e.g., all lines malformed/skipped). No motif data extracted.", file=sys.stdout)
            return None

        # Normalization logic
        for i in range(len(counts['A'])):
            pos_sum = counts['A'][i] + counts['C'][i] + counts['G'][i] + counts['T'][i]
            if abs(pos_sum - 1.0) > 1e-3 and pos_sum > 1e-6:
                if pos_sum == 0: # All counts were zero, should not happen if we demand valid freqs
                    counts['A'][i]=0.25; counts['C'][i]=0.25; counts['G'][i]=0.25; counts['T'][i]=0.25;
                    continue
                counts['A'][i] /= pos_sum; counts['C'][i] /= pos_sum; counts['G'][i] /= pos_sum; counts['T'][i] /= pos_sum
            elif abs(pos_sum - 1.0) <= 1e-3: pass # Already normalized
            elif pos_sum == 0: # All counts are zero for this position, assign background
                counts['A'][i]=0.25; counts['C'][i]=0.25; counts['G'][i]=0.25; counts['T'][i]=0.25;
        return counts

    except ValueError as ve:
        # Catches ValueErrors raised by the parsing logic itself (e.g., "Unrecognized PWM format")
        print(f"  INFO: Issue parsing PWM {os.path.basename(pwm_filepath)}: {ve}. No motif data extracted.", file=sys.stdout)
        return None
    except Exception as e:
        # For truly unexpected errors (e.g., permission denied during open, though less likely here)
        print(f"  ERROR: Unexpected issue reading/parsing PWM {os.path.basename(pwm_filepath)}: {e}. No motif data extracted.", file=sys.stderr)
        return None

def _calculate_cisbp_motif_reliability_for_prompt(motifs_metadata: List[Dict]) -> str:
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
    HIGH_CONFIDENCE_SOURCES_FOR_PRIORITY = ["JASPAR", "HOCOMOCO"]
    status = row.get('TF_Status'); source = row.get('MSource_Identifier', '')
    is_high_conf_source = any(hs.lower() in source.lower() for hs in HIGH_CONFIDENCE_SOURCES_FOR_PRIORITY if hs)
    if status == 'D': return 1 if is_high_conf_source else 2
    elif status == 'I': return 3 if is_high_conf_source else 4
    else: return 5

def convert_to_meme_format(motifs_with_metadata: List[tuple]) -> str:
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
def run_cisbp_fetch(protein_name: str, cisbp_tf_info_df: pd.DataFrame, cisbp_pwm_dir: str) -> Dict:
    print(f"\n--- TOOL: Fetching CisBP motifs for '{protein_name}' ---")
    results = {'motifs_metadata': [], 'status': 'not_found_cisbp', 'query_name': protein_name, 'num_total_found': 0, 'num_selected': 0, 'num_direct_selected': 0, 'num_inferred_selected': 0, 'sources_selected': set(), 'filtering_applied': False}
    if cisbp_tf_info_df is None or cisbp_tf_info_df.empty or not cisbp_pwm_dir:
        results['status'] = 'error_cisbp_data_not_loaded_or_empty'; print(f"  WARN: CisBP DataFrame empty or PWM dir missing. Cannot fetch for {protein_name}."); results['sources_selected'] = list(results['sources_selected']); return results
    try:
        # --- Start of Alias Handling ---
        query_protein_name = PROTEIN_ALIASES.get(protein_name, protein_name)

        if query_protein_name != protein_name:
            print(f"  Protein '{protein_name}' has an alias: '{query_protein_name}'. Using alias for initial search.")

        protein_matches = cisbp_tf_info_df[cisbp_tf_info_df['TF_Name'] == query_protein_name].copy()

        # If alias was used and successful, update results['query_name']
        if not protein_matches.empty and query_protein_name != protein_name:
            results['query_name'] = query_protein_name
            print(f"  Alias '{query_protein_name}' search successful. Found {len(protein_matches)} matches.")
        # If alias was used but failed, revert to original protein_name for search and results['query_name']
        elif protein_matches.empty and query_protein_name != protein_name:
            print(f"  Alias '{query_protein_name}' for '{protein_name}' yielded no results. Trying original name '{protein_name}'.")
            query_protein_name = protein_name # Fallback to original name
            results['query_name'] = protein_name # Ensure query_name reflects the original name being tried now
            protein_matches = cisbp_tf_info_df[cisbp_tf_info_df['TF_Name'] == query_protein_name].copy()
        # If no alias was involved, results['query_name'] is already protein_name (correct)

        results['num_total_found'] = len(protein_matches) # Update total found based on final query_protein_name

        if protein_matches.empty:
            print(f"  No entries found for TF_Name '{query_protein_name}' in loaded CisBP data.")
            # If an alias was tried and failed, query_protein_name is now the original name.
            # If protein_name was used from the start, it's also the original name.
            # No need for an extra print about the original name failing if an alias was involved,
            # as the above print already shows the final (original) name that failed.
            results['sources_selected'] = list(results['sources_selected'])
            return results

        selected_matches_df = protein_matches
        if results['num_total_found'] > MAX_MOTIFS_PER_TF_TO_PROCESS:
            print(f"  Applying filtering: Selecting top {MAX_MOTIFS_PER_TF_TO_PROCESS} motifs based on quality for '{query_protein_name}'."); results['filtering_applied'] = True # Added query_protein_name for clarity
            protein_matches['priority'] = protein_matches.apply(_calculate_motif_priority, axis=1); selected_matches_df = protein_matches.nsmallest(MAX_MOTIFS_PER_TF_TO_PROCESS, 'priority', keep='first'); print(f"  Selected {len(selected_matches_df)} motifs for '{query_protein_name}' after filtering.") # Added query_protein_name
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
        else: results['status'] = 'error_cisbp_filtering' # This could happen if all PWMs for selected motifs failed to load
        source_str = ", ".join(list(results['sources_selected'])[:5]) + ('...' if len(results['sources_selected']) > 5 else ''); filter_msg = f"(Filtered from {results['num_total_found']})" if results['filtering_applied'] else ""
        # Use results['query_name'] in the final success message as it reflects what was found
        print(f"  Successfully loaded {loaded_motifs_count} motif data structures for '{results['query_name']}' {filter_msg}. Status: {results['status']}. Sources: {source_str}")
    except KeyError as ke: print(f"ERROR: Missing expected column during CisBP fetch: {ke}", file=sys.stderr); results['status'] = 'error_cisbp_processing'
    # Use results['query_name'] in the general error message as well
    except Exception as e: print(f"ERROR during CisBP motif fetching for '{results['query_name']}' (original input: '{protein_name}'): {e}", file=sys.stderr); results['status'] = 'error_cisbp_processing'; traceback.print_exc()
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

def run_transformer_prediction(
    dna_sequence: str,
    protein_name: str,
    transformer_model,
    pro_embs_dict: Dict,
    dna_db_con,
    transformer_config: Dict,
    device: torch.device,
    dnabert_model: BertModel,
    dnabert_tokenizer: AutoTokenizer,
    dnabert_kmer: int,
    dnabert_max_len: int,
    dnabert_device: torch.device

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


# --- Function to get protein input strategy ---
def get_protein_input_strategy_for_unknown(
    protein_name_uc: str,
    model_type: str,
    model_specific_resources: dict,
    global_protein_sequences_map_arg: dict,
    protein_id_to_tf_family_map_shared_arg: dict,
    similarity_engine_cfg: str,
    num_top_similar_for_avg_logits_cfg: int,
    temp_dir_base_path_cfg: Path,
    max_levenshtein_dist_cfg: float,
    min_mmseqs_blast_bitscore_cfg: float,
    max_mmseqs_blast_evalue_cfg: float,
    verbose_matching: bool = False
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, list], None], str, str]: # Payload can be tensor or tuple
    global _dataset_inference_cache, _unknown_handling_stats_cache

    protein_payload = None # Can be tensor or tuple (ids_tensor, gene_names_list)
    protein_input_type = "error_init"
    handling_message = f"Error determining strategy for {protein_name_uc} (model: {model_type})"
    # ... (cache initialization logic for _dataset_inference_cache remains the same) ...
    if not isinstance(temp_dir_base_path_cfg, Path):
        temp_dir_base_path_cfg = Path(temp_dir_base_path_cfg)
    temp_dir_base_path_cfg.mkdir(parents=True, exist_ok=True)

    if model_type not in _dataset_inference_cache:
        if verbose_matching: print(f"INFO (test8_eval): Initializing ProteinDNADatasetInference for model type '{model_type}' for unknown handling...")
        required_keys = ["protein_id_to_model_id_map", "known_protein_ids_for_model",
                         "protein_id_to_embedding_map_for_fallback", "fallback_cpu_embedding",
                         "tf_family_to_model_known_protein_ids_map", "mmseqs_db_path_prefix", "blast_db_path_prefix"]
        if not all(key in model_specific_resources for key in required_keys):
            missing_keys = [key for key in required_keys if key not in model_specific_resources]
            handling_message = f"ERROR: Missing required resources {missing_keys} for model {model_type} in get_protein_input_strategy"
            print(handling_message, file=sys.stderr)
            return None, "error_missing_resources", handling_message
        _dataset_inference_cache[model_type] = ProteinDNADatasetInference(
            data_df=pd.DataFrame(), protein_id_to_model_id_map=model_specific_resources["protein_id_to_model_id_map"],
            dna_seq_len=0, global_protein_sequences_map=global_protein_sequences_map_arg,
            protein_id_to_embedding_map=model_specific_resources["protein_id_to_embedding_map_for_fallback"],
            known_protein_ids_for_model=model_specific_resources["known_protein_ids_for_model"],
            protein_id_to_tf_family_map=protein_id_to_tf_family_map_shared_arg,
            tf_family_to_known_protein_ids_map=model_specific_resources["tf_family_to_model_known_protein_ids_map"],
            fallback_embedding_for_unknown=model_specific_resources["fallback_cpu_embedding"],
            verbose_unknown_matching=verbose_matching,
            num_top_similar_for_avg_logits=num_top_similar_for_avg_logits_cfg,
            similarity_engine=similarity_engine_cfg,
            max_levenshtein_dist=max_levenshtein_dist_cfg,
            min_mmseqs_blast_bitscore=min_mmseqs_blast_bitscore_cfg,
            max_mmseqs_blast_evalue=max_mmseqs_blast_evalue_cfg,
            mmseqs_db_path_prefix=model_specific_resources["mmseqs_db_path_prefix"],
            blast_db_path_prefix=model_specific_resources["blast_db_path_prefix"],
            temp_dir_base=temp_dir_base_path_cfg
        )
        _unknown_handling_stats_cache[model_type] = _dataset_inference_cache[model_type].unknown_handling_stats
        if verbose_matching: print(f"INFO (test8_eval): ProteinDNADatasetInference for '{model_type}' initialized.")

    dataset_handler = _dataset_inference_cache[model_type]
    original_verbosity = dataset_handler.verbose_unknown_matching
    dataset_handler.verbose_unknown_matching = verbose_matching
    
    try:
        protein_payload, protein_input_type = dataset_handler._handle_unknown_protein(protein_name_uc)
        handling_message = (f"Unknown '{protein_name_uc}' for {model_type}: Strategy='{protein_input_type}'.")
        if protein_input_type == "ensemble_similar_ids" and protein_payload is not None:
            ids_tensor, gene_names_list = protein_payload # Unpack tuple
            handling_message += f" Found {ids_tensor.numel()} similar IDs: {', '.join(gene_names_list)}." # MODIFIED
        elif protein_input_type == "embedding":
            handling_message += " Using a fallback embedding."
        elif protein_input_type == "id" and protein_payload is not None:
            handling_message += f" Using fallback ID {protein_payload.item()}."
    except Exception as e_handle:
        handling_message = f"ERROR during _handle_unknown_protein for '{protein_name_uc}' (model: {model_type}): {e_handle}"
        print(handling_message, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        protein_payload = None
        protein_input_type = "error_handle_unknown"

    dataset_handler.verbose_unknown_matching = original_verbosity
    return protein_payload, protein_input_type, handling_message


# --- Modify run_deepsea_prediction and run_sei_prediction ---
def run_deepsea_prediction(
    dna_sequence: str, protein_name: str, deepsea_model: DeepSEAProteinInteraction,
    model_specific_resources_uh: dict, global_sequences_map_uh: dict, tf_family_map_uh: dict,
    dna_seq_len: int, device: torch.device
) -> Dict:
    protein_name_uc = protein_name.upper()
    print(f"\n--- TOOL: Running DeepSEA Prediction for '{protein_name_uc}' (Advanced Unknown Handling) ---")
    prob = None; status = 'success'; error_msg = None; handling_log = ""
    # ... (initial checks for model and resources remain the same) ...
    if deepsea_model is None: return {'status': 'error_model_not_loaded', 'probability': None, 'message': "DeepSEA model object is None."}
    if model_specific_resources_uh is None: return {'status': 'error_missing_uh_resources', 'probability': None, 'message': "Missing unknown handling resources for DeepSEA."}

    try:
        dna_one_hot = dna_to_one_hot(dna_sequence, dna_seq_len)
        dna_tensor_batch = torch.FloatTensor(dna_one_hot).unsqueeze(0).to(device)
        model_internal_id = model_specific_resources_uh["protein_id_to_model_id_map"].get(protein_name_uc)
        final_logit = None

        if model_internal_id is not None:
            handling_log = f"Known protein '{protein_name_uc}', using ID {model_internal_id}."
            protein_id_tensor = torch.LongTensor([model_internal_id]).to(device)
            final_logit = deepsea_model(dna_tensor_batch, protein_id_tensor, protein_input_type="id").squeeze()
        else:
            # Call get_protein_input_strategy_for_unknown (passing all configs)
            protein_input_payload, strategy_type, strategy_message = get_protein_input_strategy_for_unknown(
                protein_name_uc=protein_name_uc, model_type="deepsea",
                model_specific_resources=model_specific_resources_uh,
                global_protein_sequences_map_arg=global_sequences_map_uh,
                protein_id_to_tf_family_map_shared_arg=tf_family_map_uh,
                similarity_engine_cfg=SIMILARITY_ENGINE_CONFIG,
                num_top_similar_for_avg_logits_cfg=NUM_TOP_SIMILAR_FOR_AVG_LOGITS_CONFIG,
                temp_dir_base_path_cfg=UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG,
                max_levenshtein_dist_cfg=MAX_LEVENSHTEIN_DIST_CONFIG,
                min_mmseqs_blast_bitscore_cfg=MIN_MMSEQS_BLAST_BITSCORE_CONFIG,
                max_mmseqs_blast_evalue_cfg=MAX_MMSEQS_BLAST_EVALUE_CONFIG,
                verbose_matching=args.verbose_debug_unknown_handling if "args" in globals() and hasattr(args, "verbose_debug_unknown_handling") else False
            )
            handling_log = strategy_message # This now contains the detailed message including similar protein names if applicable

            if strategy_type == "id":
                protein_id_tensor = protein_input_payload.to(device)
                if protein_id_tensor.ndim == 0: protein_id_tensor = protein_id_tensor.unsqueeze(0)
                final_logit = deepsea_model(dna_tensor_batch, protein_id_tensor, protein_input_type="id").squeeze()
            elif strategy_type == "embedding":
                protein_emb_tensor = protein_input_payload.unsqueeze(0).to(device)
                final_logit = deepsea_model(dna_tensor_batch, protein_emb_tensor, protein_input_type="embedding").squeeze()
            elif strategy_type == "ensemble_similar_ids":
                ids_tensor, _ = protein_input_payload # Unpack the tuple
                similar_ids_tensor = ids_tensor.to(device)
                if similar_ids_tensor.numel() > 0:
                    ensemble_logits = []
                    for sim_id_scalar_tensor in similar_ids_tensor:
                        current_sim_id_batch = sim_id_scalar_tensor.reshape(1).to(device)
                        logit = deepsea_model(dna_tensor_batch, current_sim_id_batch, protein_input_type="id").squeeze()
                        ensemble_logits.append(logit)
                    if ensemble_logits: final_logit = torch.stack(ensemble_logits).mean()
                    else: status = 'error_deepsea_ensemble_no_logits'; error_msg = "Ensemble strategy yielded no logits despite having IDs."
                else: status = 'error_deepsea_ensemble_no_ids'; error_msg = "Ensemble strategy selected but no similar IDs tensor provided."
            elif strategy_type.startswith("error_"):
                 status = strategy_type
                 error_msg = handling_log # Use the detailed error from strategy function
            else:
                status = 'error_deepsea_unknown_strategy_type'; error_msg = f"Unhandled strategy type: {strategy_type}"

        if final_logit is not None:
            prob = torch.sigmoid(final_logit).item()
            # The handling_log from get_protein_input_strategy_for_unknown already has details.
            # We can append the final probability to it.
            handling_log_with_prob = f"{handling_log} Final Prob: {prob:.4f}."
            if args.verbose_debug_unknown_handling or not handling_log.startswith("Known protein"): # Print if unknown or verbose
                print(f"  {handling_log_with_prob}")
            handling_log = handling_log_with_prob # Update for return message
        elif status == 'success':
             status = 'error_deepsea_no_logit_produced'; error_msg = "No final logit was produced despite status=success."

    except Exception as e:
        status = 'error_deepsea_runtime'; error_msg = f"DeepSEA Prediction Runtime Error: {e}"
        print(f"  ERROR during DeepSEA prediction for '{protein_name_uc}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    final_message_to_return = handling_log
    if error_msg and error_msg not in final_message_to_return : final_message_to_return += f" Error: {error_msg}"
    return {'status': status, 'probability': prob, 'message': final_message_to_return.strip()}


def run_sei_prediction(
    dna_sequence: str, protein_name: str, sei_model: SeiProteinInteractionWithMeanEmbedding,
    model_specific_resources_uh: dict, global_sequences_map_uh: dict, tf_family_map_uh: dict,
    dna_seq_len: int, device: torch.device
) -> Dict:
    protein_name_uc = protein_name.upper()
    print(f"\n--- TOOL: Running SEI Prediction for '{protein_name_uc}' (Advanced Unknown Handling) ---")
    prob = None; status = 'success'; error_msg = None; handling_log = ""
    # ... (initial checks for model and resources remain the same) ...
    if sei_model is None: return {'status': 'error_model_not_loaded', 'probability': None, 'message': "SEI model object is None."}
    if model_specific_resources_uh is None: return {'status': 'error_missing_uh_resources', 'probability': None, 'message': "Missing unknown handling resources for SEI."}

    try:
        dna_one_hot = dna_to_one_hot(dna_sequence, dna_seq_len)
        dna_tensor_batch = torch.FloatTensor(dna_one_hot).unsqueeze(0).to(device)
        model_internal_id = model_specific_resources_uh["protein_id_to_model_id_map"].get(protein_name_uc)
        final_logit = None

        if model_internal_id is not None:
            handling_log = f"Known protein '{protein_name_uc}', using ID {model_internal_id}."
            protein_id_tensor = torch.LongTensor([model_internal_id]).to(device)
            final_logit = sei_model(dna_tensor_batch, protein_id_tensor, protein_input_type="id").squeeze()
        else:
            protein_input_payload, strategy_type, strategy_message = get_protein_input_strategy_for_unknown(
                protein_name_uc=protein_name_uc, model_type="sei",
                model_specific_resources=model_specific_resources_uh,
                global_protein_sequences_map_arg=global_sequences_map_uh,
                protein_id_to_tf_family_map_shared_arg=tf_family_map_uh,
                similarity_engine_cfg=SIMILARITY_ENGINE_CONFIG,
                num_top_similar_for_avg_logits_cfg=NUM_TOP_SIMILAR_FOR_AVG_LOGITS_CONFIG,
                temp_dir_base_path_cfg=UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG,
                max_levenshtein_dist_cfg=MAX_LEVENSHTEIN_DIST_CONFIG,
                min_mmseqs_blast_bitscore_cfg=MIN_MMSEQS_BLAST_BITSCORE_CONFIG,
                max_mmseqs_blast_evalue_cfg=MAX_MMSEQS_BLAST_EVALUE_CONFIG,
                verbose_matching=args.verbose_debug_unknown_handling if "args" in globals() and hasattr(args, "verbose_debug_unknown_handling") else False
            )
            handling_log = strategy_message

            if strategy_type == "id":
                protein_id_tensor = protein_input_payload.to(device)
                if protein_id_tensor.ndim == 0: protein_id_tensor = protein_id_tensor.unsqueeze(0)
                final_logit = sei_model(dna_tensor_batch, protein_id_tensor, protein_input_type="id").squeeze()
            elif strategy_type == "embedding":
                protein_emb_tensor = protein_input_payload.unsqueeze(0).to(device)
                final_logit = sei_model(dna_tensor_batch, protein_emb_tensor, protein_input_type="embedding").squeeze()
            elif strategy_type == "ensemble_similar_ids":
                ids_tensor, _ = protein_input_payload # Unpack
                similar_ids_tensor = ids_tensor.to(device)
                if similar_ids_tensor.numel() > 0:
                    ensemble_logits = []
                    for sim_id_scalar_tensor in similar_ids_tensor:
                        current_sim_id_batch = sim_id_scalar_tensor.reshape(1).to(device)
                        logit = sei_model(dna_tensor_batch, current_sim_id_batch, protein_input_type="id").squeeze()
                        ensemble_logits.append(logit)
                    if ensemble_logits: final_logit = torch.stack(ensemble_logits).mean()
                    else: status = 'error_sei_ensemble_no_logits'; error_msg = "Ensemble strategy yielded no logits."
                else: status = 'error_sei_ensemble_no_ids'; error_msg = "Ensemble strategy selected but no similar IDs."
            elif strategy_type.startswith("error_"):
                 status = strategy_type
                 error_msg = handling_log
            else:
                status = 'error_sei_unknown_strategy_type'; error_msg = f"Unhandled strategy type: {strategy_type}"

        if final_logit is not None:
            prob = torch.sigmoid(final_logit).item()
            handling_log_with_prob = f"{handling_log} Final Prob: {prob:.4f}."
            if args.verbose_debug_unknown_handling or not handling_log.startswith("Known protein"):
                print(f"  {handling_log_with_prob}")
            handling_log = handling_log_with_prob
        elif status == 'success':
             status = 'error_sei_no_logit_produced'; error_msg = "No final logit was produced."

    except Exception as e:
        status = 'error_sei_runtime'; error_msg = f"SEI Prediction Runtime Error: {e}"
        print(f"  ERROR during SEI prediction for '{protein_name_uc}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    final_message_to_return = handling_log
    if error_msg and error_msg not in final_message_to_return: final_message_to_return += f" Error: {error_msg}"
    return {'status': status, 'probability': prob, 'message': final_message_to_return.strip()}


def get_dpi_pred_node_wrapper(
    state: AgentState,
    dpi_transformer_model,
    dpi_pro_embs_dict,
    dpi_dna_db_con,
    dpi_transformer_config,
    dpi_device,
    dpi_training_protein_set,
    dnabert_model,
    dnabert_tokenizer,
    dnabert_kmer,
    dnabert_max_len,
    dnabert_device
) -> AgentState:
    print("\n--- Node: Get DPI Transformer Prediction ---")
    if state.get("error"): return state # Propagate errors

    # Check if DPI model should run based on models_to_run in state
    if "dpi" not in state.get("models_to_run", []):
        print("  DPI model not selected to run. Skipping.")
        # Optionally, add a placeholder to all_model_predictions if strict accounting is needed
        # current_predictions = state.get("all_model_predictions", [])
        # current_predictions.append({
        #     "model_name": "dpi", "probability": None, "is_known": False,
        #     "status": "skipped_not_selected", "message": "Not selected for this run."
        # })
        # return {**state, "all_model_predictions": current_predictions}
        return state # Pass state through if not running this model

    protein_name = state["protein_name"]
    model_output = {
        "model_name": "dpi", "probability": None, "is_known": False,
        "status": "init_error", "message": "DPI prediction did not run due to an unexpected issue."
    } # Default error state for this model

    try:
        results = run_transformer_prediction(
            dna_sequence=state["dna_sequence"],
            protein_name=protein_name,
            transformer_model=dpi_transformer_model,
            pro_embs_dict=dpi_pro_embs_dict,
            dna_db_con=dpi_dna_db_con,
            transformer_config=dpi_transformer_config,
            device=dpi_device,
            dnabert_model=dnabert_model,
            dnabert_tokenizer=dnabert_tokenizer,
            dnabert_kmer=dnabert_kmer,
            dnabert_max_len=dnabert_max_len,
            dnabert_device=dnabert_device
        )

        is_known = protein_name in dpi_training_protein_set if dpi_training_protein_set else False

        model_output["probability"] = results.get('probability')
        model_output["is_known"] = is_known
        model_output["status"] = results.get('status', 'unknown_error_status')
        model_output["message"] = results.get('message', '')

        if model_output["status"].startswith('error'):
             print(f"  ERROR: DPI Prediction Failed ({model_output['status']}): {model_output['message']}", file=sys.stderr)
             # The error is already in model_output
        else:
            print(f"  DPI Prob: {model_output['probability']:.4f}, Known Protein (in DPI Training Set): {model_output['is_known']}")

        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions}

    except Exception as e:
        print(f"  Critical Error in get_dpi_pred_node_wrapper: {e}"); traceback.print_exc();
        model_output["status"] = "error_node_wrapper"
        model_output["message"] = str(e)
        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions, "error": f"DPI Node Wrapper Error: {e}"}


def get_deepsea_pred_node_wrapper(
    state: AgentState,
    deepsea_model_instance,
    deepsea_model_specific_resources_uh, # NEW
    global_sequences_map_uh,          # NEW
    tf_family_map_uh,                 # NEW
    deepsea_dna_seq_len_param,
    deepsea_device_param
) -> AgentState:
    print("\n--- Node: Get DeepSEA Prediction (with advanced unknown handling) ---")
    if state.get("error"): return state
    if "deepsea" not in state.get("models_to_run", []):
        print("  DeepSEA model not selected to run. Skipping."); return state

    protein_name = state["protein_name"]
    model_output = {"model_name": "deepsea", "probability": None, "is_known": False, "status": "init_error", "message": "DeepSEA prediction did not run."}
    
    # Check if necessary resources for unknown handling are present if model_specific_resources_uh is None
    if deepsea_model_specific_resources_uh is None and protein_name.upper() not in (deepsea_model_specific_resources_uh or {}).get("protein_id_to_model_id_map", {}):
        # This case indicates 'deepsea' might not have been fully initialized in MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING
        # or the protein is unknown AND resources are missing.
        model_output["status"] = "error_deepsea_init_for_unknown"
        model_output["message"] = "DeepSEA resources for unknown protein handling not available."
        print(f"  ERROR: {model_output['message']} for protein {protein_name}")
        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions}


    try:
        results = run_deepsea_prediction(
            dna_sequence=state["dna_sequence"],
            protein_name=protein_name,
            deepsea_model=deepsea_model_instance,
            model_specific_resources_uh=deepsea_model_specific_resources_uh,
            global_sequences_map_uh=global_sequences_map_uh,
            tf_family_map_uh=tf_family_map_uh,
            dna_seq_len=deepsea_dna_seq_len_param,
            device=deepsea_device_param
        )
        # "is_known" now refers to whether it was directly in the model's original map
        is_known = protein_name.upper() in deepsea_model_specific_resources_uh["protein_id_to_model_id_map"] if deepsea_model_specific_resources_uh else False
        
        model_output["probability"] = results.get('probability')
        model_output["is_known"] = is_known
        model_output["status"] = results.get('status', 'unknown_error_status')
        model_output["message"] = results.get('message', '')

        if model_output["status"].startswith('error'):
             print(f"  ERROR: DeepSEA Prediction Failed ({model_output['status']}): {model_output['message']}", file=sys.stderr)
        # Message now contains handling details, so no need for separate print if successful
        
        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions}

    except Exception as e:
        print(f"  Critical Error in get_deepsea_pred_node_wrapper: {e}"); traceback.print_exc();
        model_output["status"] = "error_node_wrapper"; model_output["message"] = str(e)
        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions, "error": f"DeepSEA Node Wrapper Error: {e}"}

def get_sei_pred_node_wrapper(
    state: AgentState,
    sei_model_instance_param,
    sei_model_specific_resources_uh, # NEW
    global_sequences_map_uh,       # NEW
    tf_family_map_uh,              # NEW
    sei_dna_seq_len_param,
    sei_device_param
) -> AgentState:
    print("\n--- Node: Get SEI Prediction (with advanced unknown handling) ---")
    if state.get("error"): return state
    if "sei" not in state.get("models_to_run", []):
        print("  SEI model not selected to run. Skipping."); return state

    protein_name = state["protein_name"]
    model_output = {"model_name": "sei", "probability": None, "is_known": False, "status": "init_error", "message": "SEI prediction did not run."}

    if sei_model_specific_resources_uh is None and protein_name.upper() not in (sei_model_specific_resources_uh or {}).get("protein_id_to_model_id_map", {}):
        model_output["status"] = "error_sei_init_for_unknown"
        model_output["message"] = "SEI resources for unknown protein handling not available."
        print(f"  ERROR: {model_output['message']} for protein {protein_name}")
        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions}

    try:
        results = run_sei_prediction(
            dna_sequence=state["dna_sequence"],
            protein_name=protein_name,
            sei_model=sei_model_instance_param,
            model_specific_resources_uh=sei_model_specific_resources_uh,
            global_sequences_map_uh=global_sequences_map_uh,
            tf_family_map_uh=tf_family_map_uh,
            dna_seq_len=sei_dna_seq_len_param,
            device=sei_device_param
        )
        is_known = protein_name.upper() in sei_model_specific_resources_uh["protein_id_to_model_id_map"] if sei_model_specific_resources_uh else False
        
        model_output["probability"] = results.get('probability')
        model_output["is_known"] = is_known
        model_output["status"] = results.get('status', 'unknown_error_status')
        model_output["message"] = results.get('message', '')
        
        if model_output["status"].startswith('error'):
             print(f"  ERROR: SEI Prediction Failed ({model_output['status']}): {model_output['message']}", file=sys.stderr)

        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions}
        
    except Exception as e:
        print(f"  Critical Error in get_sei_pred_node_wrapper: {e}"); traceback.print_exc();
        model_output["status"] = "error_node_wrapper"; model_output["message"] = str(e)
        current_predictions = state.get("all_model_predictions", [])
        current_predictions.append(model_output)
        return {**state, "all_model_predictions": current_predictions, "error": f"SEI Node Wrapper Error: {e}"}

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

        # Remove trailing whitespace rigorously
        parsed_explanation = parsed_explanation.rstrip()
        # Remove common trailing words/prepositions if they are left dangling
        common_trailing = ["the", "a", "an", "is", "of", "in", "on", "at", "for", "to", "and", "or", "but"]
        words = parsed_explanation.split()
        if words and words[-1].lower() in common_trailing:
            # print(f"  Trimming trailing word: '{words[-1]}'") # Optional debug
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
        generation_config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048) # Increased max_output for potentially longer prompts
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
    data = {"model": model_tag, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 2048}} # Increased num_predict
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
        # Use a consistent format that works well, e.g. Deepseek or general instruction format
        # If specific models require specific chat templates, tokenizer.apply_chat_template might be better
        # For now, keeping the simple format:
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
        # print(f"  Decoded Raw Text: '{raw_response_text}'") # Can be very verbose
        prediction, parsed_explanation = _parse_llm_response(raw_response_text) # Parse
        print(f"  LLM Parsed Vote: {prediction}")
        error_explanation = None # Clear error on success

    except Exception as e:
        print(f"  ERROR during Hugging Face model generation: {e}", file=sys.stderr)
        traceback.print_exc()
        error_explanation = f"Error during Hugging Face model generation: {e}"
        parsed_explanation = error_explanation # Use error as explanation

    return prediction, parsed_explanation if error_explanation is None else error_explanation, raw_response_text


def calculate_confidence_score(state: AgentState, prompt_style: str, use_transformer_prob_as_baseline: bool) -> float:
    print(f"--- Calculating Rule-Based Confidence Score (Style: {prompt_style}, Use Model Prob Baseline: {use_transformer_prob_as_baseline}) ---")
    llm_vote = state.get('llm_vote')
    if llm_vote is None:
        print("  LLM vote is missing, cannot calculate confidence. Returning 0.0")
        return 0.0

    all_predictions = state.get("all_model_predictions", [])

    # --- Initialize ALL confidence parameters with their DEFAULT values first ---
    baseline = CONFIDENCE_BASELINE
    direct_match_boost = DIRECT_MATCH_BOOST
    direct_reliable_match_extra = DIRECT_RELIABLE_MATCH_EXTRA
    direct_mismatch_penalty = DIRECT_MISMATCH_PENALTY
    direct_reliable_mismatch_extra = DIRECT_RELIABLE_MISMATCH_EXTRA
    direct_missing_penalty = DIRECT_MISSING_PENALTY
    direct_absence_boost = DIRECT_ABSENCE_BOOST

    indirect_match_boost = INDIRECT_MATCH_BOOST
    indirect_mismatch_penalty = INDIRECT_MISMATCH_PENALTY
    indirect_missing_penalty = INDIRECT_MISSING_PENALTY
    indirect_absence_boost = INDIRECT_ABSENCE_BOOST

    model_match_boost_base = TRANSFORMER_MATCH_BOOST
    model_confident_extra_base = TRANSFORMER_CONFIDENT_EXTRA
    model_mismatch_penalty_base = TRANSFORMER_MISMATCH_PENALTY
    model_confident_mismatch_extra_base = TRANSFORMER_CONFIDENT_MISMATCH_EXTRA

    current_style_model_novelty_factor = TRANSFORMER_NOVELTY_FACTOR

    num_hits_bonus_max = GENERAL_NUM_HITS_BONUS_MAX; num_hits_scale_factor = GENERAL_NUM_HITS_SCALE_FACTOR
    max_hits_for_bonus_cap = GENERAL_MAX_HITS_FOR_BONUS_CAP; low_pvalue_threshold = GENERAL_LOW_PVALUE_THRESHOLD
    low_pvalue_bonus = GENERAL_LOW_PVALUE_BONUS

    if prompt_style == "transformer-priority":
        baseline = TP_CONFIDENCE_BASELINE
        direct_match_boost = TP_DIRECT_MATCH_BOOST; direct_reliable_match_extra = TP_DIRECT_RELIABLE_MATCH_EXTRA
        direct_mismatch_penalty = TP_DIRECT_MISMATCH_PENALTY; direct_reliable_mismatch_extra = TP_DIRECT_RELIABLE_MISMATCH_EXTRA
        direct_missing_penalty = TP_DIRECT_MISSING_PENALTY; direct_absence_boost = TP_DIRECT_ABSENCE_BOOST
        indirect_match_boost = TP_INDIRECT_MATCH_BOOST; indirect_mismatch_penalty = TP_INDIRECT_MISMATCH_PENALTY
        indirect_missing_penalty = TP_INDIRECT_MISSING_PENALTY; indirect_absence_boost = TP_INDIRECT_ABSENCE_BOOST

        model_match_boost_base = TP_TRANSFORMER_MATCH_BOOST; model_confident_extra_base = TP_TRANSFORMER_CONFIDENT_EXTRA
        model_mismatch_penalty_base = TP_TRANSFORMER_MISMATCH_PENALTY; model_confident_mismatch_extra_base = TP_TRANSFORMER_CONFIDENT_MISMATCH_EXTRA
        current_style_model_novelty_factor = TP_TRANSFORMER_NOVELTY_FACTOR

    elif prompt_style == "motif-priority":
        baseline = MP_CONFIDENCE_BASELINE
        direct_match_boost = MP_DIRECT_MATCH_BOOST; direct_reliable_match_extra = MP_DIRECT_RELIABLE_MATCH_EXTRA
        direct_mismatch_penalty = MP_DIRECT_MISMATCH_PENALTY; direct_reliable_mismatch_extra = MP_DIRECT_RELIABLE_MISMATCH_EXTRA
        direct_missing_penalty = MP_DIRECT_MISSING_PENALTY; direct_absence_boost = MP_DIRECT_ABSENCE_BOOST
        indirect_match_boost = MP_INDIRECT_MATCH_BOOST; indirect_mismatch_penalty = MP_INDIRECT_MISMATCH_PENALTY
        indirect_missing_penalty = MP_INDIRECT_MISSING_PENALTY; indirect_absence_boost = MP_INDIRECT_ABSENCE_BOOST

        model_match_boost_base = MP_TRANSFORMER_MATCH_BOOST; model_confident_extra_base = MP_TRANSFORMER_CONFIDENT_EXTRA
        model_mismatch_penalty_base = MP_TRANSFORMER_MISMATCH_PENALTY; model_confident_mismatch_extra_base = MP_TRANSFORMER_CONFIDENT_MISMATCH_EXTRA
        current_style_model_novelty_factor = MP_TRANSFORMER_NOVELTY_FACTOR

        num_hits_bonus_max = MP_NUM_HITS_BONUS_MAX; num_hits_scale_factor = MP_NUM_HITS_SCALE_FACTOR
        max_hits_for_bonus_cap = MP_MAX_HITS_FOR_BONUS_CAP; low_pvalue_threshold = MP_LOW_PVALUE_THRESHOLD
        low_pvalue_bonus = MP_LOW_PVALUE_BONUS

    if use_transformer_prob_as_baseline:
        derived_baselines_from_models = []
        for pred_res_baseline in all_predictions:
            prob_baseline = pred_res_baseline.get("probability")
            status_baseline = pred_res_baseline.get("status", "")
            if prob_baseline is not None and not status_baseline.startswith('error') and not status_baseline.startswith('skipped'):
                derived_baselines_from_models.append(0.5 + abs(prob_baseline - 0.5))
        if derived_baselines_from_models:
            avg_derived_baseline = sum(derived_baselines_from_models) / len(derived_baselines_from_models)
            original_style_baseline = baseline
            print(f"  Overriding style-based baseline ({original_style_baseline:.2f}) with average model-derived baseline: {avg_derived_baseline:.2f} (from {len(derived_baselines_from_models)} models)")
            baseline = avg_derived_baseline
        elif all_predictions:
            print(f"  WARN: --use-transformer-prob-as-baseline is True, but no valid model probabilities found. Using style-based baseline: {baseline:.2f}")

    scan_fail_factor = SCAN_FAIL_FACTOR
    transformer_confident_high = TRANSFORMER_CONFIDENT_HIGH
    transformer_confident_low = TRANSFORMER_CONFIDENT_LOW

    confidence = baseline
    adjustments = []

    num_successful_models = 0
    num_llm_agreements_from_successful_models = 0
    if all_predictions:
        for pred_res_count in all_predictions:
            _prob = pred_res_count.get("probability")
            _status = pred_res_count.get("status", "")
            if _prob is not None and not _status.startswith('error') and not _status.startswith('skipped'):
                num_successful_models +=1
                _vote = 1 if _prob > 0.5 else 0
                if _vote == llm_vote:
                    num_llm_agreements_from_successful_models +=1

    strong_model_agreement_with_llm_for_conflict = (
        num_successful_models >= MIN_MODELS_FOR_CONFLICT_AMPLIFICATION and
        num_llm_agreements_from_successful_models == num_successful_models
    )

    direct_summary = state.get('direct_motif_analysis_summary') or {}
    direct_fetch_results = state.get('direct_motif_fetch_results') or {} # Get the fetch results too

    # +++ START OF MODIFICATION for Direct Analysis Ablation +++
    direct_fetch_status = direct_fetch_results.get('status', 'unknown')
    direct_scan_status = direct_summary.get('scan_status', 'unknown')
    
    # Check if direct analysis was explicitly ablated
    is_direct_ablated = (
        direct_fetch_status == 'skipped_ablation_direct' or
        direct_scan_status == 'skipped_ablation_direct' or
        direct_summary.get('motif_reliability_source_category', '') == "Skipped (Ablation)"
    )

    if is_direct_ablated:
        reason = direct_summary.get('scan_message') or direct_fetch_results.get('message') or "Direct analysis ablated by user flag"
        adjustments.append(f"Direct Motif Analysis Skipped (Ablation: {reason}): No Adjustment")
    else:
        # This 'else' block contains the original logic for direct motif analysis when it was NOT ablated
        direct_found = direct_summary.get('motif_found_in_dna', False)
        direct_reliability = direct_summary.get('motif_reliability_source_category', 'Unknown')
        # direct_scan_ok now depends on the scan_status not being an error, AND not being ablated (already handled by outer if)
        direct_scan_ok = not direct_scan_status.startswith('error')
        # direct_fetch_ok now depends on the fetch_status not being an error, AND not being ablated
        direct_fetch_ok = not direct_fetch_status.startswith('error')

        is_reliable_direct = "High" in direct_reliability
        direct_num_hits = direct_summary.get('number_of_hits', 0)
        direct_best_pvalue = direct_summary.get('lowest_hit_pvalue')
        direct_mod = scan_fail_factor if not direct_scan_ok else 1.0 # This applies if scan was attempted but failed
        
        effective_direct_missing_penalty = direct_missing_penalty # Use style-specific values
        effective_direct_mismatch_penalty = direct_mismatch_penalty

        # Apply conflict amplification if relevant (this part can remain inside the 'else')
        if strong_model_agreement_with_llm_for_conflict and direct_fetch_ok and direct_scan_ok:
            conflict_note = f"  (Note: Strong model agreement with LLM vote={llm_vote} from {num_successful_models} models detected)"
            amplification_applied_log = False
            if llm_vote == 1 and not direct_found:
                effective_direct_missing_penalty *= MOTIF_CONFLICT_AMPLIFICATION_FACTOR
                adjustments.append(f"{conflict_note}\n    Amplifying direct missing penalty due to model/motif conflict (Factor: {MOTIF_CONFLICT_AMPLIFICATION_FACTOR:.1f})")
                amplification_applied_log = True
            elif llm_vote == 0 and direct_found:
                effective_direct_mismatch_penalty *= MOTIF_CONFLICT_AMPLIFICATION_FACTOR
                adjustments.append(f"{conflict_note}\n    Amplifying direct mismatch penalty due to model/motif conflict (Factor: {MOTIF_CONFLICT_AMPLIFICATION_FACTOR:.1f})")
                amplification_applied_log = True
            if not amplification_applied_log and conflict_note not in " ".join(adjustments) :
                 adjustments.append(conflict_note + " - No direct motif conflict to amplify.")

        if direct_fetch_ok: # Only proceed if fetch was OK (not an error, and not ablated)
            if llm_vote == 1:
                if direct_found:
                    adj = direct_match_boost * direct_mod; adjustments.append(f"Direct Hit Match: +{adj:.2f}"); confidence += adj
                    if is_reliable_direct: adj_extra = direct_reliable_match_extra * direct_mod; adjustments.append(f"  Reliable: +{adj_extra:.2f}"); confidence += adj_extra
                    if num_hits_scale_factor > 0 and direct_num_hits > 0: hits_bonus = min(num_hits_bonus_max, num_hits_scale_factor * min(direct_num_hits, max_hits_for_bonus_cap)) * direct_mod; adjustments.append(f"  Direct Num Hits Bonus ({direct_num_hits} hits): +{hits_bonus:.2f}"); confidence += hits_bonus
                    if low_pvalue_bonus > 0 and direct_best_pvalue is not None and direct_best_pvalue < low_pvalue_threshold: pval_b = low_pvalue_bonus * direct_mod; adjustments.append(f"  Direct Low PVal Bonus ({direct_best_pvalue:.1e}): +{pval_b:.2f}"); confidence += pval_b
                else: # Not found, but fetch and scan were attempted and OK
                    adj = effective_direct_missing_penalty * direct_mod
                    adjustments.append(f"Direct Hit Missing (Vote=1): {adj:.2f}")
                    confidence += adj
            elif llm_vote == 0:
                if direct_found:
                    adj = effective_direct_mismatch_penalty * direct_mod
                    adjustments.append(f"Direct Hit Mismatch (Vote=0): {adj:.2f}")
                    confidence += adj
                    if is_reliable_direct: adj_extra = direct_reliable_mismatch_extra * direct_mod; adjustments.append(f"  Reliable Mismatch: {adj_extra:.2f}"); confidence += adj_extra
                else: # Not found, and fetch and scan were attempted and OK
                    adj = direct_absence_boost * direct_mod
                    adjustments.append(f"Direct Hit Absence (Vote=0): +{adj:.2f}")
                    confidence += adj
        else: # Fetch was not OK (e.g., error during fetch, not ablation)
            adjustments.append(f"Direct Motif Fetch/Scan Failed or Errored (Status: {direct_fetch_status}/{direct_scan_status}): No Adjustment")

    # --- Indirect Motif Analysis Contribution ---
    indirect_summary = state.get('indirect_motif_analysis_summary') or {}
    indirect_performed = indirect_summary.get('search_performed', False)
    indirect_skipped_reason = indirect_summary.get('skipped_reason', '') # Get the reason if skipped

    # +++ START OF MODIFICATION for Indirect Analysis Ablation +++
    # Check if indirect analysis was explicitly ablated
    is_indirect_ablated = (
        not indirect_performed and 
        ("ablat" in indirect_skipped_reason.lower() or 
         indirect_summary.get('scan_status', '') == 'skipped_ablation')
    )

    if is_indirect_ablated:
        reason = indirect_skipped_reason or "Indirect analysis ablated by user flag"
        adjustments.append(f"Indirect Motif Analysis Skipped (Ablation: {reason}): No Adjustment")
    elif indirect_performed: # This 'elif' block contains the original logic when indirect analysis was performed
        indirect_found = indirect_summary.get('any_indirect_motif_found', False)
        indirect_scan_status = indirect_summary.get('scan_status', 'unknown')
        indirect_scan_ok = not indirect_scan_status.startswith('error') # scan_status not an error, and not ablated
        indirect_mod = scan_fail_factor if not indirect_scan_ok else 1.0

        indirect_num_hits = len(indirect_summary.get('interacting_tf_hits', []))
        indirect_best_hit = indirect_summary.get('best_interacting_tf_hit')
        indirect_best_pvalue = indirect_best_hit.get('pvalue') if indirect_best_hit else None
        
        if llm_vote == 1:
            if indirect_found:
                adj = indirect_match_boost * indirect_mod; adjustments.append(f"Indirect Hit Match: +{adj:.2f}"); confidence += adj
                if num_hits_scale_factor > 0 and indirect_num_hits > 0: hits_bonus = min(num_hits_bonus_max, num_hits_scale_factor * min(indirect_num_hits, max_hits_for_bonus_cap)) * indirect_mod; adjustments.append(f"  Indirect Num Hits Bonus ({indirect_num_hits} hits): +{hits_bonus:.2f}"); confidence += hits_bonus
                if low_pvalue_bonus > 0 and indirect_best_pvalue is not None and indirect_best_pvalue < low_pvalue_threshold: pval_b = low_pvalue_bonus * indirect_mod; adjustments.append(f"  Indirect Low PVal Bonus ({indirect_best_pvalue:.1e}): +{pval_b:.2f}"); confidence += pval_b
            else: # Not found, but search was performed and scan was OK
                adj = indirect_missing_penalty * indirect_mod; adjustments.append(f"Indirect Hit Missing (Vote=1): {adj:.2f}"); confidence += adj
        elif llm_vote == 0:
            if indirect_found:
                adj = indirect_mismatch_penalty * indirect_mod; adjustments.append(f"Indirect Hit Mismatch (Vote=0): {adj:.2f}"); confidence += adj
            else: # Not found, but search was performed and scan was OK
                adj = indirect_absence_boost * indirect_mod; adjustments.append(f"Indirect Hit Absence (Vote=0): +{adj:.2f}"); confidence += adj
    else: # Not ablated, but search was not performed for other reasons (e.g., no interactors found)
        # indirect_performed is False, and it wasn't an ablation
        # The existing message "Indirect Search Skipped: No Adjustment" is appropriate.
        # We can make it slightly more specific if a reason is available.
        skip_msg = "Indirect Search Skipped"
        if indirect_skipped_reason:
            skip_msg += f" (Reason: {indirect_skipped_reason})"
        else:
            skip_msg += " (e.g., no interactors)"
        skip_msg += ": No Adjustment"
        adjustments.append(skip_msg)
    # +++ END OF MODIFICATION for Indirect Analysis Ablation +++

    if not all_predictions:
        adjustments.append("No Prediction Models Run/Data Available: No Model Adjustment")

    agreeing_model_boost_applied_count = 0
    mismatching_model_penalty_applied_count = 0 # <<< NEW counter for mismatch penalties

    for pred_result in all_predictions:
        model_name_for_log = pred_result.get("model_name", "UnknownModel").upper()
        model_prob = pred_result.get("probability")
        model_is_known = pred_result.get("is_known", False)
        model_status = pred_result.get("status", "")

        if model_status.startswith('error') or model_status.startswith('skipped'):
            adjustments.append(f"{model_name_for_log} Prediction Skipped/Error: No Adjustment")
            continue
        if model_prob is None:
            adjustments.append(f"{model_name_for_log} Prob Missing: No Adjustment")
            continue

        novelty_mod_for_this_model = current_style_model_novelty_factor if not model_is_known else 1.0
        model_vote = 1 if model_prob > 0.5 else 0
        is_model_pred_confident_for_bonus = (model_prob > transformer_confident_high or model_prob < transformer_confident_low)

        adj_prefix_log = f"{model_name_for_log} "
        novelty_log_str = f" (Novel *{current_style_model_novelty_factor:.1f})" if not model_is_known else ""

        if llm_vote == model_vote: # Model agrees with LLM
            agreeing_model_boost_applied_count += 1

            current_match_boost = model_match_boost_base
            current_confident_extra = model_confident_extra_base
            diminishing_factor_log = ""

            if agreeing_model_boost_applied_count > 1:
                current_match_boost *= SUBSEQUENT_MODEL_AGREEMENT_FACTOR
                current_confident_extra *= SUBSEQUENT_MODEL_AGREEMENT_FACTOR
                diminishing_factor_log = f" (Diminished Boost *{SUBSEQUENT_MODEL_AGREEMENT_FACTOR:.1f})"

            adj = current_match_boost * novelty_mod_for_this_model
            adjustments.append(f"{adj_prefix_log}Match: +{adj:.2f}{novelty_log_str}{diminishing_factor_log}")
            confidence += adj
            if is_model_pred_confident_for_bonus:
                adj_extra = current_confident_extra * novelty_mod_for_this_model
                adjustments.append(f"  {adj_prefix_log}Confident Bonus: +{adj_extra:.2f}{diminishing_factor_log}")
                confidence += adj_extra
        else: # Model disagrees with LLM
            mismatching_model_penalty_applied_count +=1 # <<< Increment mismatch counter

            current_mismatch_penalty = model_mismatch_penalty_base
            current_confident_mismatch_extra = model_confident_mismatch_extra_base
            diminishing_penalty_log = ""

            if mismatching_model_penalty_applied_count > 1: # <<< Apply diminishing factor to subsequent mismatches
                current_mismatch_penalty *= SUBSEQUENT_MODEL_MISMATCH_PENALTY_FACTOR
                current_confident_mismatch_extra *= SUBSEQUENT_MODEL_MISMATCH_PENALTY_FACTOR
                diminishing_penalty_log = f" (Diminished Penalty *{SUBSEQUENT_MODEL_MISMATCH_PENALTY_FACTOR:.1f})"

            adj = current_mismatch_penalty * novelty_mod_for_this_model
            adjustments.append(f"{adj_prefix_log}Mismatch: {adj:.2f}{novelty_log_str}{diminishing_penalty_log}")
            confidence += adj
            if is_model_pred_confident_for_bonus:
                adj_extra = current_confident_mismatch_extra * novelty_mod_for_this_model
                adjustments.append(f"  {adj_prefix_log}Confident Mismatch Penalty: {adj_extra:.2f}{diminishing_penalty_log}")
                confidence += adj_extra

    final_confidence = max(0.0, min(1.0, confidence))

    print(f"  Baseline: {baseline:.2f}")
    # Sort adjustments for readability, especially with multi-line amplification notes
    sorted_adjustments = []
    temp_amplification_note = None
    for adj_str in adjustments:
        if "Amplifying direct" in adj_str: # The multi-line note
            temp_amplification_note = adj_str # Store it
        else:
            sorted_adjustments.append(adj_str)
    if temp_amplification_note: # Prepend it if it exists
        print(f"  {temp_amplification_note}") # Print the full note first

    # Print other adjustments
    # Basic sorting for the rest might be nice but not strictly necessary here
    # For now, just print them as they came, after the potential amplification note
    non_amplification_adjustments = [s for s in adjustments if "Amplifying direct" not in s]
    for adj_str_sorted in non_amplification_adjustments:
        print(f"  {adj_str_sorted}")

    print(f"  => Final Confidence: {final_confidence:.3f}")

    return final_confidence

# --- Graph Nodes ---
def start_analysis(state: AgentState, models_to_run_arg: List[str], ablate_indirect_analysis_flag: bool, ablate_direct_analysis_flag: bool) -> AgentState:
    print("\n--- Starting Analysis ---"); print(f" Protein: {state['protein_name']}"); print(f" DNA Seq Length: {len(state['dna_sequence'])}")
    print(f"  Ground Truth Received: {state.get('ground_truth_label', 'MISSING')}")
    print(f"  Models selected to run: {models_to_run_arg}")
    if ablate_indirect_analysis_flag:
        print("  Indirect analysis will be SKIPPED (ablated).")
    if ablate_direct_analysis_flag:
        print("  Direct analysis will be SKIPPED (ablated by --no-direct-analysis).")
    initial_state_keys = {k: None for k in AgentState.__annotations__}
    initial_state = {
        **initial_state_keys, # Initialize all keys to None first
        "protein_name": state['protein_name'],
        "dna_sequence": state['dna_sequence'],
        "ground_truth_label": state.get('ground_truth_label'),
        "final_confidence": None,
        "models_to_run": models_to_run_arg, # Store the list of models to run
        "all_model_predictions": [],
        "ablate_indirect_analysis": ablate_indirect_analysis_flag, # Store the flag in state
        "ablate_direct_analysis": ablate_direct_analysis_flag # Store the flag in state

    
    }
    return initial_state

def mark_direct_skipped_node(state: AgentState) -> AgentState:
    print("\n--- Node: Mark Direct Analysis as Skipped (Ablation) ---")
    if state.get("error"): return state # Propagate errors

    skipped_reason = "Direct analysis ablated by --no-direct-analysis flag."
    
    fetch_results_skipped = {
        'motifs_metadata': [], 'status': 'skipped_ablation_direct',
        'query_name': state["protein_name"], 'num_total_found': 0, 'num_selected': 0,
        'num_direct_selected': 0, 'num_inferred_selected': 0,
        'sources_selected': [], 'filtering_applied': False,
        'message': skipped_reason
    }
    scan_results_skipped = {
        "status": "skipped_ablation_direct", "hits": [],
        "message": skipped_reason
    }
    analysis_summary_skipped = {
        "p_value_threshold": DEFAULT_P_VALUE_THRESHOLD, # Keep for consistency, though not used
        "motif_found_in_dna": False,
        "motif_reliability_source_category": "Skipped (Ablation)",
        "highest_hit_score": None, "lowest_hit_pvalue": None,
        "number_of_hits": 0, "best_hit_details": None,
        "protein_motif_info": fetch_results_skipped, # Embed the skipped fetch results
        "significant_hits": [],
        "scan_status": "skipped_ablation_direct",
        "scan_message": skipped_reason
    }
    print(f"  Direct Analysis Summary: Skipped - {skipped_reason}")
    return {
        **state,
        "direct_motif_fetch_results": fetch_results_skipped,
        "direct_motif_scan_results": scan_results_skipped,
        "direct_motif_analysis_summary": analysis_summary_skipped
    }

def fetch_direct_motifs_node(state: AgentState, cisbp_tf_info_df: pd.DataFrame, cisbp_pwm_dir: str) -> AgentState:
    print("\n--- Node: Fetch Direct Motifs ---");
    if state.get("error"): return state
    try: results = run_cisbp_fetch(protein_name=state["protein_name"], cisbp_tf_info_df=cisbp_tf_info_df, cisbp_pwm_dir=cisbp_pwm_dir); return {**state, "direct_motif_fetch_results": results}
    except Exception as e: print(f"  Critical Error in fetch_direct_motifs_node: {e}"); traceback.print_exc(); return {**state, "error": f"Fetch Direct Node Error: {e}"}

def scan_direct_dna_node(state: AgentState) -> AgentState:
    print("\n--- Node: Scan DNA for Direct Motifs ---");
    if state.get("error"): return state;
    fetch_results = state.get("direct_motif_fetch_results") or {}; motifs_meta = fetch_results.get('motifs_metadata')
    if not motifs_meta: print("  Skipping scan: No direct motifs fetched or available."); return {**state, "direct_motif_scan_results": {"status": "success_skipped", "hits": []}}
    try: scan_results = run_motif_scan(dna_sequence=state["dna_sequence"], motifs_with_metadata=motifs_meta, pvalue_threshold=DEFAULT_P_VALUE_THRESHOLD); return {**state, "direct_motif_scan_results": scan_results}
    except Exception as e: print(f"  Critical Error in scan_direct_dna_node: {e}"); traceback.print_exc(); return {**state, "error": f"Scan Direct Node Error: {e}"}

def summarize_direct_analysis_node(state: AgentState) -> AgentState:
    print("\n--- Node: Summarize Direct Motif Analysis ---");
    if state.get("error"): return state;
    fetch_results = state.get("direct_motif_fetch_results") or {}; scan_results = state.get("direct_motif_scan_results") or {"hits": [], "status": "unknown"}; motifs_metadata_list = [meta for _, meta in fetch_results.get('motifs_metadata', [])]; reliability_str = _calculate_cisbp_motif_reliability_for_prompt(motifs_metadata_list); all_significant_hits = scan_results.get("hits", []); motif_found_in_dna = len(all_significant_hits) > 0; best_hit = all_significant_hits[0] if motif_found_in_dna else None; best_score = best_hit.get('score') if best_hit else None; best_pvalue = best_hit.get('pvalue') if best_hit else None; scan_status = scan_results.get('status', 'unknown'); scan_message = scan_results.get('message');
    summary = {"p_value_threshold": DEFAULT_P_VALUE_THRESHOLD, "motif_found_in_dna": motif_found_in_dna, "motif_reliability_source_category": reliability_str, "highest_hit_score": best_score, "lowest_hit_pvalue": best_pvalue, "number_of_hits": len(all_significant_hits), "best_hit_details": best_hit, "protein_motif_info": fetch_results, "significant_hits": all_significant_hits, "scan_status": scan_status, "scan_message": scan_message}; print(f"  Direct Analysis Summary: Hits Found={summary['motif_found_in_dna']}, Reliability={summary['motif_reliability_source_category']}, Scan Status={scan_status}"); return {**state, "direct_motif_analysis_summary": summary}

def query_string_node(state: AgentState) -> AgentState:
    print("\n--- Node: Query STRING DB ---");
    if state.get("error"): return state
    try: results = run_string_query(protein_name=state["protein_name"], species_id=TARGET_SPECIES_TAXONOMY_ID, min_score=STRING_MIN_INTERACTION_SCORE, max_interactors=MAX_STRING_INTERACTORS_TO_CHECK); return {**state, "string_interactors": results.get('interactors', [])}
    except Exception as e: print(f"  Critical Error in query_string_node: {e}"); traceback.print_exc(); return {**state, "error": f"STRING Node Error: {e}"}

def filter_string_interactors_node(state: AgentState, cisbp_tf_info_df: pd.DataFrame) -> AgentState:
    print("\n--- Node: Filter STRING Interactors for TFs ---");
    if state.get("error"): return state;
    string_interactors = state.get("string_interactors") or []; tf_interactors = []; tfs_to_analyze = []
    if cisbp_tf_info_df is None or cisbp_tf_info_df.empty: print("  WARN: CisBP TF Info DataFrame not available for filtering.", file=sys.stderr)
    else:
        known_tf_names = set(cisbp_tf_info_df['TF_Name'].unique()); tf_interactors = [i for i in string_interactors if i.get("name") in known_tf_names]; print(f"  Found {len(tf_interactors)} known TFs among {len(string_interactors)} STRING interactors.")
        tfs_to_analyze = tf_interactors
        if len(tf_interactors) > MAX_INTERACTING_TFS_TO_ANALYZE: print(f"  Limiting analysis to top {MAX_INTERACTING_TFS_TO_ANALYZE} TFs based on STRING score."); tfs_to_analyze.sort(key=lambda x: x.get('score', 0), reverse=True); tfs_to_analyze = tfs_to_analyze[:MAX_INTERACTING_TFS_TO_ANALYZE]
    return {**state, "tf_interactors_cisbp": tf_interactors, "tfs_analyzed_indirectly": tfs_to_analyze}

def fetch_indirect_motifs_node(state: AgentState, cisbp_tf_info_df: pd.DataFrame, cisbp_pwm_dir: str) -> AgentState:
    print("\n--- Node: Fetch Indirect Motifs ---");
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
    print("\n--- Node: Scan DNA for Indirect Motifs ---");
    if state.get("error"): return state;
    fetch_results = state.get("indirect_motif_fetch_results") or {}; motifs_meta = fetch_results.get('motifs_metadata')
    if not motifs_meta: print("  Skipping scan: No indirect motifs fetched or available."); return {**state, "indirect_motif_scan_results": {"status": "success_skipped", "hits": []}}
    try:
        scan_results = run_motif_scan(dna_sequence=state["dna_sequence"], motifs_with_metadata=motifs_meta, pvalue_threshold=INDIRECT_MOTIF_P_VALUE_THRESHOLD)
        metadata_map_indirect = {meta['Motif_ID']: meta for _, meta in motifs_meta if meta.get('Motif_ID')}
        for hit in scan_results.get("hits", []):
            if 'interacting_tf' not in hit or hit['interacting_tf'] is None:
                meta = metadata_map_indirect.get(hit.get('motif_id'));
                if meta: hit['interacting_tf'] = meta.get('interacting_tf_name'); hit['interacting_tf_string_score'] = meta.get('interacting_tf_string_score');
        return {**state, "indirect_motif_scan_results": scan_results}
    except Exception as e: print(f"  Critical Error in scan_indirect_dna_node: {e}"); traceback.print_exc(); return {**state, "error": f"Scan Indirect Node Error: {e}"}

def summarize_indirect_analysis_node(state: AgentState) -> AgentState:
    print("\n--- Node: Summarize Indirect Motif Analysis ---");
    if state.get("error"): return state;
    string_interactors = state.get("string_interactors") or []; tf_interactors_cisbp = state.get("tf_interactors_cisbp") or []; tfs_analyzed_indirectly = state.get("tfs_analyzed_indirectly") or []; scan_results = state.get("indirect_motif_scan_results") or {"hits": [], "status": "unknown"}; scan_status = scan_results.get('status', 'unknown'); scan_message = scan_results.get('message'); all_significant_hits = scan_results.get("hits", []); any_indirect_motif_found = len(all_significant_hits) > 0; all_significant_hits.sort(key=lambda x: x.get('pvalue', float('inf'))); best_hit = all_significant_hits[0] if any_indirect_motif_found else None;
    summary = {"search_performed": True, "skipped_reason": None, "interactors_found_string": string_interactors, "interactors_tf_cisbp": tf_interactors_cisbp, "interactors_tf_analyzed": tfs_analyzed_indirectly, "interacting_tf_hits": all_significant_hits, "best_interacting_tf_hit": best_hit, "any_indirect_motif_found": any_indirect_motif_found, "scan_status": scan_status, "scan_message": scan_message, "error_message": None}
    # Use pre-set summary if indirect path was skipped
    if state.get("indirect_motif_analysis_summary") and not state["indirect_motif_analysis_summary"].get("search_performed"):
        summary = state["indirect_motif_analysis_summary"]
        print(f"  Indirect Analysis Summary: Skipped - {summary.get('skipped_reason')}")
    else: print(f"  Indirect Analysis Summary: Search Performed, Hits Found={summary['any_indirect_motif_found']}, #Interactors={len(summary.get('interactors_found_string',[]))}, #TF_Interactors={len(summary.get('interactors_tf_cisbp',[]))}, #TFs_Analyzed={len(summary.get('interactors_tf_analyzed',[]))}, Scan Status={scan_status}");
    return {**state, "indirect_motif_analysis_summary": summary}

def mark_indirect_skipped_node(state: AgentState) -> AgentState:
    print("\n--- Node: Mark Indirect Analysis as Skipped (Ablation) ---")
    if state.get("error"): return state # Propagate errors

    # Populate indirect_motif_analysis_summary with a "skipped" status
    # This ensures downstream nodes (prompt generation, confidence score)
    # have a consistent structure to check.
    summary = {
        "search_performed": False,
        "skipped_reason": "Indirect analysis ablated by user flag.",
        "interactors_found_string": [],
        "interactors_tf_cisbp": [],
        "interactors_tf_analyzed": [],
        "interacting_tf_hits": [],
        "best_interacting_tf_hit": None,
        "any_indirect_motif_found": False,
        "scan_status": "skipped_ablation",
        "scan_message": "Indirect analysis ablated by --no-indirect-analysis flag.",
        "error_message": None # No error, just skipped
    }
    print(f"  Indirect Analysis Summary: Skipped - {summary.get('skipped_reason')}")
    return {**state, "indirect_motif_analysis_summary": summary}

def generate_concise_llm_prompt(state: AgentState) -> str:
    """Generates the concise LLM prompt matching the fine-tuning format, now with multiple model predictions."""
    protein_name = state.get('protein_name', 'N/A')
    direct_summary = state.get("direct_motif_analysis_summary") or {}
    indirect_summary = state.get("indirect_motif_analysis_summary") or {}
    all_predictions = state.get("all_model_predictions", [])

    # Determine overall novelty based on all run models for the general protein info string
    # This is a simplified view for the concise prompt; detailed novelty is per-model below.
    overall_novelty_str = "Novel (Absent from all/most Training Sets or models failed)"
    known_by_any_model = False
    if all_predictions:
        for pred_res in all_predictions:
            if pred_res.get("is_known") and not pred_res.get("status","").startswith("error"):
                known_by_any_model = True
                break
    if known_by_any_model:
        overall_novelty_str = "Known (Present in at least one prediction model's Training Set)"

    protein_info_str = f"{protein_name} ({overall_novelty_str})"


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
    indirect_p_val = f"{direct_summary.get('p_value_threshold', INDIRECT_MOTIF_P_VALUE_THRESHOLD):.1e}" 
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

    # --- Format Prediction Model Summaries ---
    model_prediction_summary_lines = []
    if not all_predictions:
        model_prediction_summary_lines.append("* Prediction Models: No models run or all failed.")
    for pred in all_predictions:
        model_name_upper = pred.get("model_name", "N/A").upper()
        prob_val = pred.get("probability")
        is_known_to_model = pred.get("is_known", False)
        status = pred.get("status", "unknown")

        prob_str = "N/A"
        interpretation = "N/A"
        trust_context = "N/A"

        if status.startswith("error") or status.startswith("skipped"):
            prob_str = f"Failed/Skipped ({status})"
            trust_context = f"{status}"
        elif prob_val is not None:
            prob_str = f"{prob_val:.4f}"
            if prob_val > TRANSFORMER_CONFIDENT_HIGH: interpretation = "High Probability"
            elif prob_val > 0.6: interpretation = "Moderate Probability"
            elif prob_val > 0.4: interpretation = "Uncertain"
            elif prob_val > TRANSFORMER_CONFIDENT_LOW: interpretation = "Low Probability"
            else: interpretation = "Very Low Probability"

            trust_context = "Known Protein" if is_known_to_model else "Novel Protein"

        model_prediction_summary_lines.append(
            f"* {model_name_upper} Prediction: Probability: {prob_str} ({interpretation}) - [Trustworthiness vs {model_name_upper} set: {trust_context}]"
        )

    all_model_predictions_str = "\n".join(model_prediction_summary_lines)


    # --- Construct the Concise Prompt ---
    prompt = f"""Below is evidence regarding a DNA sequence and protein interaction. Provide a step-by-step thinking process synthesizing this evidence to reach a final label (0 or 1).

### Instruction:
You are a computational biology expert evaluating potential DNA-protein interactions based on multiple evidence sources. Synthesize the provided evidence step-by-step within the `<think>` tags and provide the final interaction label.

### Protein:
{protein_info_str}

### Retrieved Evidence:
* Direct Motif Scan (p<={direct_p_val}): {direct_motif_summary}
* Indirect Motif Scan (p<={indirect_p_val}): {indirect_motif_summary}
{all_model_predictions_str}

### Analysis Task:
Generate a step-by-step reasoning process within <think> tags, synthesizing the 'Retrieved Evidence' above to determine the interaction label. Conclude with '### Interaction:\nThe interaction label is [0 or 1]'.

### Analysis:
<think>
"""
    return prompt

def _format_common_evidence_for_prompt(state: AgentState) -> Dict[str, str]:
    protein_name = state.get('protein_name', 'N/A')
    dna_sequence = state.get('dna_sequence', '')
    direct_summary = state.get("direct_motif_analysis_summary") or {}
    indirect_summary = state.get("indirect_motif_analysis_summary") or {}
    all_predictions = state.get("all_model_predictions", []) # Get all model predictions

    evidence_parts = {}

    # --- Format DNA info (unchanged) ---
    dna_len = len(dna_sequence)
    max_dna_len_full_display = 1024
    disp_dna = dna_sequence if dna_len <= max_dna_len_full_display else f"{dna_sequence[:60]}...{dna_sequence[-60:]}"
    evidence_parts['disp_dna_str'] = disp_dna
    evidence_parts['dna_info_str'] = f"(Length: {dna_len})"

    # --- Format Direct Motif Evidence (largely unchanged) ---
    direct_motif_info_summary = direct_summary.get('protein_motif_info', {})
    direct_motif_status = direct_motif_info_summary.get('status', 'unknown')
    evidence_parts['direct_protein_has_motifs_str'] = 'Yes' if direct_motif_status.startswith('success') else 'No/Error'
    evidence_parts['direct_reliability_str'] = direct_summary.get('motif_reliability_source_category', 'Unknown')
    evidence_parts['direct_motif_found_dna_str'] = str(direct_summary.get('motif_found_in_dna', False))

    direct_hits = direct_summary.get('significant_hits', [])
    p_thresh = direct_summary.get('p_value_threshold', DEFAULT_P_VALUE_THRESHOLD)
    p_thresh_print = f"{p_thresh:.1e}" if isinstance(p_thresh, float) else 'N/A'

    direct_cisbp_search_details_list = [f"Status: {direct_motif_status}. Reliability: {evidence_parts['direct_reliability_str']}."]
    if direct_motif_status.startswith('success'):
        num_direct = direct_motif_info_summary.get('num_direct_selected', 0); num_inferred = direct_motif_info_summary.get('num_inferred_selected', 0)
        num_selected = direct_motif_info_summary.get('num_selected', 0); num_total = direct_motif_info_summary.get('num_total_found', 0)
        filtering_applied = direct_motif_info_summary.get('filtering_applied', False)
        filter_msg = f"(Top {num_selected} selected from {num_total})" if filtering_applied else f"({num_selected} total)"
        direct_cisbp_search_details_list.append(f" Found {num_direct} Direct, {num_inferred} Inferred motifs {filter_msg}.")
        direct_motif_detail_list_meta = [m for _, m in direct_motif_info_summary.get('motifs_metadata', [])]
        max_direct_motifs_to_list = 3; listed_motifs = 0
        for m_meta in direct_motif_detail_list_meta:
            if listed_motifs < max_direct_motifs_to_list: direct_cisbp_search_details_list.append(f" | {m_meta.get('Motif_ID', '?')} ({m_meta.get('TF_Status', '?')}/{m_meta.get('MSource_Identifier','?')})"); listed_motifs += 1
            else: direct_cisbp_search_details_list.append(f" | ..."); break
    elif direct_motif_status.startswith('error'): direct_cisbp_search_details_list.append(f" (Error during fetch: {direct_motif_status})")
    evidence_parts['direct_motif_cisbp_search_details_str'] = "".join(direct_cisbp_search_details_list)

    direct_scan_info_list = []
    max_direct_scan_hits_to_list = 2
    direct_scan_status = direct_summary.get('scan_status', 'unknown')
    if direct_scan_status.startswith('error'): direct_scan_info_list.append(f"  Scan Error: {direct_scan_status} ({direct_summary.get('scan_message', 'No details')})")
    elif direct_hits:
        direct_scan_info_list.append(f"  Hits in DNA (p<={p_thresh_print}, Top {min(len(direct_hits), max_direct_scan_hits_to_list)} by p-value):")
        for i, h in enumerate(direct_hits):
            if i < max_direct_scan_hits_to_list: hit_info = f"Mot:{h.get('motif_id','?')} @Pos:{h.get('position','?')}({h.get('strand','?')}) Score:{h.get('score',float('nan')):.2f} p:{h.get('pvalue', float('nan')):.2e}"; direct_scan_info_list.append(f"    - {hit_info}")
        if len(direct_hits) > max_direct_scan_hits_to_list: direct_scan_info_list.append("    ...")
    else: direct_scan_info_list.append(f"  No significant direct motif hits found (p <= {p_thresh_print}). Scan status: {direct_scan_status}")
    evidence_parts['direct_motif_scan_details_str'] = "\n".join(direct_scan_info_list)

    # --- Format Indirect Motif Evidence (largely unchanged) ---
    indirect_search_performed = indirect_summary.get("search_performed", False)
    evidence_parts['indirect_search_performed_str'] = str(indirect_search_performed)
    evidence_parts['indirect_skipped_reason_str'] = indirect_summary.get("skipped_reason", "")
    evidence_parts['indirect_tfs_analyzed_count_str'] = str(len(indirect_summary.get("interactors_tf_analyzed", [])))
    evidence_parts['indirect_motifs_found_dna_str'] = str(indirect_summary.get("any_indirect_motif_found", False))

    indirect_info_list = []
    if not indirect_search_performed and evidence_parts['indirect_skipped_reason_str']: indirect_info_list.append(f"  Indirect search skipped: {evidence_parts['indirect_skipped_reason_str']}")
    elif not indirect_search_performed: indirect_info_list.append("  Indirect search was not performed.")
    else:
        # ... (indirect search details generation as before) ...
        string_interactors_count = len(indirect_summary.get("interactors_found_string", [])); tf_interactors = indirect_summary.get("interactors_tf_cisbp", []); tfs_analyzed = indirect_summary.get("interactors_tf_analyzed", [])
        indirect_info_list.append(f"  STRING DB Search: Found {string_interactors_count} interactors (Score > {STRING_MIN_INTERACTION_SCORE}).")
        if not tf_interactors: indirect_info_list.append("    - None of these interactors are known TFs in our CisBP database.")
        else:
            indirect_info_list.append(f"    - {len(tf_interactors)} interactors are known TFs. Analyzed top {len(tfs_analyzed)}:")
            listed_tfs = 0; max_indirect_interactors_to_list = 5
            for tf_i in tfs_analyzed:
                if listed_tfs < max_indirect_interactors_to_list: indirect_info_list.append(f"      - {tf_i.get('name','?')} (STRING Score: {tf_i.get('score','?')})"); listed_tfs += 1
            if len(tfs_analyzed) > max_indirect_interactors_to_list: indirect_info_list.append(f"      ...")
    evidence_parts['indirect_motif_search_details_str'] = "\n".join(indirect_info_list)

    indirect_scan_hits_list = []
    if indirect_search_performed:
        indirect_hits_data = indirect_summary.get("interacting_tf_hits", []); indirect_scan_status_val = indirect_summary.get('scan_status', 'unknown'); max_indirect_scan_hits_to_list = 3
        if indirect_scan_status_val.startswith('error'): indirect_scan_hits_list.append(f"  Scan Error for Indirect Motifs: {indirect_scan_status_val} ({indirect_summary.get('scan_message', 'No details')})")
        elif indirect_hits_data:
            indirect_scan_hits_list.append(f"  Hits in DNA from Interacting TFs (p<={p_thresh_print}, Top {min(len(indirect_hits_data), max_indirect_scan_hits_to_list)} by p-value):")
            for i, h_indirect in enumerate(indirect_hits_data):
                if i < max_indirect_scan_hits_to_list: hit_info_indirect = f"TF:{h_indirect.get('interacting_tf','?')} (Mot:{h_indirect.get('motif_id','?')}) @Pos:{h_indirect.get('position','?')}({h_indirect.get('strand','?')}) Score:{h_indirect.get('score',float('nan')):.2f} p:{h_indirect.get('pvalue', float('nan')):.2e}"; indirect_scan_hits_list.append(f"    - {hit_info_indirect}")
            if len(indirect_hits_data) > max_indirect_scan_hits_to_list: indirect_scan_hits_list.append("    ...")
        elif indirect_summary.get("interactors_tf_cisbp"): indirect_scan_hits_list.append(f"  No significant indirect motif hits found (p <= {p_thresh_print}) for analyzed interacting TFs. Scan status: {indirect_scan_status_val}")
    evidence_parts['indirect_motif_scan_details_str'] = "\n".join(indirect_scan_hits_list) if indirect_scan_hits_list else "  (Indirect scan not applicable or no hits/errors to report based on prior steps)"

    # --- Format Prediction Model Evidence (Iterate through all_predictions) ---
    # This section replaces the singular prediction model formatting.
    all_models_detail_str_list = []
    all_models_summary_for_header_list = [] # For the top-level summary

    if not all_predictions:
        all_models_detail_str_list.append("  - No prediction models were run or all failed before producing output.")
        all_models_summary_for_header_list.append("- Prediction Models: None run/failed.")

    for pred_result in all_predictions:
        model_name_upper = pred_result.get("model_name", "UnknownModel").upper()
        prob = pred_result.get("probability")
        is_known = pred_result.get("is_known", False)
        status = pred_result.get("status", "unknown")
        message = pred_result.get("message", "")

        prob_str_detail = f"{prob:.4f}" if prob is not None else 'N/A'
        prob_str_summary = prob_str_detail # Can be same or simpler for header

        if status.startswith('error') or status.startswith('skipped'):
            status_msg_short = f" (Status: {status})"
            status_msg_detail = status_msg_short
            if message: status_msg_detail += f" - {message[:100]}" # Truncate long messages
            prob_str_detail += status_msg_detail
            prob_str_summary += status_msg_short


        protein_status_str = f"Known (Present in {model_name_upper} Known Set)" if is_known else f"Novel (Absent from {model_name_upper} Known Set)"

        model_confidence_str = "Unknown"
        if prob is not None and not status.startswith('error') and not status.startswith('skipped'):
            if prob < 0.1 or prob > 0.9: model_confidence_str = "High"
            elif 0.4 <= prob <= 0.6: model_confidence_str = "Low (Uncertain)"
            else: model_confidence_str = "Moderate"

        model_specific_explanation = ""
        if model_name_upper == "DPI":
            model_specific_explanation = f"  - Model Info: DPI Transformer, trained on ~1500 ChIP-seq datasets (~800 TFs). Generally high accuracy on known TFs, potentially less certain on novel ones."
        elif model_name_upper == "DEEPSEA":
            model_specific_explanation = f"  - Model Info: DeepSEA CNN model. Trained on ~1500 ChIP-seq datasets (~800 TFs). Generally high accuracy on known TFs. Performance on novel proteins depends on the mean score across the trained proteins."
        elif model_name_upper == "SEI": # MODIFIED explanation for SEI to match DeepSEA's if it also uses mean for novel
            sei_handling_novel_message = "Performance on novel proteins depends on the mean score across the trained proteins if mean embedding is used, or on default ID behavior if not."
            # Check if the prediction used mean embedding or a default ID for novel protein to refine message.
            if not is_known and message: # message often contains protein input type
                 if "embedding" in message.lower() or "mean" in message.lower() :
                     sei_handling_novel_message = "Performance on novel proteins (like this one) relies on the mean embedding of known proteins."
                 elif "default_0" in message.lower() or "id 0" in message.lower():
                     sei_handling_novel_message = "Performance on novel proteins (like this one) relies on a default ID (e.g., 0)."

            model_specific_explanation = (f"  - Model Info: SEI CNN model. Trained on ~1500 ChIP-seq datasets (~800 TFs). "
                                          f"Generally high accuracy on known TFs. {sei_handling_novel_message}")
        else:
            model_specific_explanation = f"  - Model Info: {model_name_upper} model. Details on training/scope may vary."

        # For the "Supporting Information Details" section
        all_models_detail_str_list.append(
            f"3.{len(all_models_detail_str_list)+1} {model_name_upper} Prediction Details:\n"
            f"    - Probability: {prob_str_detail}\n"
            f"    - Protein Status (vs {model_name_upper} Known Set): {protein_status_str}\n"
            f"    - Model Confidence (derived): {model_confidence_str}\n"
            f"{model_specific_explanation}"
        )
        # For the "Evidence Summary" header section
        all_models_summary_for_header_list.append(
            f"- {model_name_upper} Prediction: Prob: {prob_str_summary}, Known (vs {model_name_upper}): {is_known}, Confidence: {model_confidence_str}"
        )

    evidence_parts['all_prediction_models_detailed_str'] = "\n\n".join(all_models_detail_str_list)
    evidence_parts['all_prediction_models_summary_header_str'] = "\n".join(all_models_summary_for_header_list)

    # Remove old singular prediction model fields that might have been populated by mistake or from old logic
    # This ensures the prompt uses the new multi-model strings.
    keys_to_remove_from_evidence = [
        'prediction_model_prob_str', 'protein_status_wrt_prediction_model_str',
        'prediction_model_confidence_str', 'prediction_model_specific_explanation_str',
        'prediction_model_name_for_prompt_str'
    ]
    for key_to_remove in keys_to_remove_from_evidence:
        evidence_parts.pop(key_to_remove, None)

    return evidence_parts

def generate_verbose_llm_prompt(state: AgentState) -> str:
    protein_name = state.get('protein_name', 'N/A')
    evidence = _format_common_evidence_for_prompt(state)

    prompt = f"""Analyze the potential interaction between the DNA sequence and the Protein '{protein_name}' ({TARGET_SPECIES}). Your goal is to predict if an interaction occurs (1) or not (0), based *only* on the evidence provided below.

### Input Data:
Protein: {protein_name}
DNA {evidence['dna_info_str']}: {evidence['disp_dna_str']}

### Evidence Summary:
{evidence['all_prediction_models_summary_header_str']}
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

{evidence['all_prediction_models_detailed_str']}

### Analysis Task:
Please perform a step-by-step analysis based *strictly* on the provided evidence:
1.  **Direct Evidence:** Evaluate the direct motif evidence. Does the protein have known binding motifs? Are they reliable? Were any significant hits found in the DNA sequence? Assess the strength of this evidence (strong, moderate, weak, none, or error).
2.  **Indirect Evidence:** Evaluate the indirect motif evidence. Was the search performed? Did the protein interact with known TFs? Were any motifs belonging to these interacting TFs found in the DNA sequence? Assess the strength of this evidence, considering it's less direct than the protein's own motifs.
3.  **Prediction Model Evidence (All Models):** For each prediction model's output provided: Evaluate its probability. How confident does the model seem (High/Moderate/Low)? Does the protein's novelty (relative to that specific model's training/known set) affect your trust in its prediction?
4.  **Synthesis:** Synthesize all evidence streams (direct motifs, indirect motifs, and *all* model predictions). Do they agree or conflict? Which evidence seems most compelling or decisive in this specific case? For example, is strong direct evidence sufficient? Is indirect evidence relevant if direct evidence is missing or weak? How do the various model predictions align with each other and with motif data?
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
{evidence['all_prediction_models_summary_header_str']}
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

{evidence['all_prediction_models_detailed_str']}

### Analysis Task:
Please perform a step-by-step analysis based *strictly* on the provided evidence:
1.  **Direct Evidence:** Evaluate the direct motif evidence. Strength?
2.  **Indirect Evidence:** Evaluate the indirect motif evidence. Strength?
3.  **Prediction Model Evidence (All Models):** For each model: probability, confidence, novelty impact.
4.  **Synthesis (Prioritizing Prediction Models):** Synthesize the evidence. **Crucially, give significant weight to the prediction model(s), especially if their confidence is High or Moderate.** If multiple models agree with high confidence, their consensus is very strong. If direct/indirect motif evidence is weak, absent, or contradicts confident model prediction(s), the model evidence should generally take precedence, particularly for proteins known to those models. Explicitly state how you are weighing the evidence. Note any conflicts between models or with motif data.
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
{evidence['all_prediction_models_summary_header_str']}
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

{evidence['all_prediction_models_detailed_str']}

### Analysis Task:
Please perform a step-by-step analysis based *strictly* on the provided evidence:
1.  **Direct Evidence:** Evaluate direct motif evidence (reliability, hits, quality). Strength?
2.  **Indirect Evidence:** Evaluate indirect motif evidence (TFs, hits, quality). Strength?
3.  **Prediction Model Evidence (All Models):** For each model: probability, confidence, novelty impact.
4.  **Synthesis (Prioritizing Motif Evidence):** Synthesize all evidence. **Crucially, give significant weight to direct and then indirect motif evidence.** If any model prediction conflicts with strong or clear motif evidence (e.g., multiple high-quality direct hits), the motif evidence should generally take precedence. If motif evidence is weak, absent, or ambiguous, the model prediction(s) can be more influential. Explicitly state how you are weighing the evidence. Note any conflicts between models or with motif data.
5.  **Conclusion:** Based on your prioritized synthesis, state your final prediction (0 or 1).

### MANDATORY Output Format:
Explanation: [Your detailed step-by-step reasoning following points 1-5 above, clearly showing the prioritized weighing]
The interaction label is: [0 or 1]"""
    return prompt

def generate_llm_prompt_node(state: AgentState,
                             requested_prompt_style: str,
                             llm_identifier: str,
                             finetuned_model_id: str
                             ) -> AgentState:
    if state.get("error"): return state

    protein_name = state.get('protein_name', 'N/A')

    # --- Determine overall 'known' status for 'auto' prompt selection based on all_model_predictions ---
    overall_is_known_for_auto_prompt = False
    all_predictions_for_auto = state.get("all_model_predictions", [])
    if all_predictions_for_auto:
        for pred_res in all_predictions_for_auto:
            # Check if model ran successfully and protein is known to it
            if not pred_res.get("status", "").startswith("error") and \
               not pred_res.get("status", "").startswith("skipped") and \
               pred_res.get("is_known"):
                overall_is_known_for_auto_prompt = True
                break
    # If no models ran successfully or none reported "known", it's treated as novel for auto-style.

    effective_prompt_style = requested_prompt_style
    prompt_selection_reason = f"User selected '{requested_prompt_style}'."

    if requested_prompt_style == "auto":
        if overall_is_known_for_auto_prompt:
            effective_prompt_style = "transformer-priority" # "transformer-priority" is a misnomer, now "model-priority"
            prompt_selection_reason = f"Auto: Protein '{protein_name}' is KNOWN by at least one run model. Switched to 'transformer-priority' (model-priority)."
        else:
            effective_prompt_style = "motif-priority"
            prompt_selection_reason = f"Auto: Protein '{protein_name}' is NOVEL according to all run models (or no models successfully reported 'known'). Switched to 'motif-priority'."
        print(f"  {prompt_selection_reason}")

    is_finetuned_model = False
    if llm_identifier.startswith("hf/"):
        model_path_or_id = llm_identifier.split("hf/", 1)[1]
        if model_path_or_id == finetuned_model_id:
            is_finetuned_model = True
            if effective_prompt_style != "concise":
                original_style_before_finetune_override = effective_prompt_style
                effective_prompt_style = "concise"
                print(f"  WARN: Overriding style to 'concise' for fine-tuned model '{finetuned_model_id}' (was '{original_style_before_finetune_override}').")
                prompt_selection_reason += f" Overridden to 'concise' for fine-tuned model."

    print(f"\n--- Node: Generate LLM Prompt (Effective Style: {effective_prompt_style}) ---")
    state_update_for_prompt_info = {"llm_prompt_style_selected": effective_prompt_style,
                                    "llm_prompt_selection_reason": prompt_selection_reason}

    try:
        if effective_prompt_style == "verbose": prompt = generate_verbose_llm_prompt(state)
        elif effective_prompt_style == "concise": prompt = generate_concise_llm_prompt(state)
        elif effective_prompt_style == "transformer-priority": prompt = generate_transformer_priority_llm_prompt(state)
        elif effective_prompt_style == "motif-priority": prompt = generate_motif_priority_llm_prompt(state)
        else: raise ValueError(f"Invalid effective prompt style determined: {effective_prompt_style}")

        print(f"  Generated {effective_prompt_style} LLM Prompt (Length: {len(prompt)} chars).")
        return {**state, **state_update_for_prompt_info, "llm_prompt": prompt}
    except Exception as e:
        print(f"  Critical Error generating {effective_prompt_style} prompt: {e}")
        traceback.print_exc()
        return {**state, **state_update_for_prompt_info, "error": f"LLM Prompt Generation Error ({effective_prompt_style}): {e}"}

def get_llm_pred_node(state: AgentState,
                      llm_model_name: str,
                      ollama_api_url: str,
                      hf_model, hf_tokenizer, device,
                      api_delay: float
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
        return {**state, "error": f"LLM Prediction Node Error: {e}", "llm_vote": None, "llm_explanation": f"Error during LLM call: {e}", "raw_llm_response": ""}

def calculate_final_confidence_node(state: AgentState, use_transformer_prob_as_baseline_arg: bool) -> AgentState:
    """Node to calculate and store the final confidence score."""
    if state.get("error"):
        print("--- Skipping Confidence Calculation due to previous error ---")
        return {**state, "final_confidence": 0.0}

    prompt_style_used = state.get("llm_prompt_style_selected", "verbose") # Default to verbose if not set

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
        elif "[your detailed reasoning]" in llm_explanation.lower() or \
             "analysis task:" in llm_explanation.lower() or \
             "### supporting information details:" in llm_explanation.lower(): # Common template bleed-through
            is_garbage = True
            print("  WARN: Detected template placeholder text or structure bleed-through. Flagging as garbage.")

        # Add more checks if needed (e.g., for specific repetitive phrases)
        if is_garbage:
            consistency_warning = "[WARNING: LLM explanation appears incomplete, corrupted, or is template bleed-through] "
            print(f"  {consistency_warning}")
            final_confidence = 0.01 # Very low confidence
            updated_explanation = consistency_warning + (llm_explanation or "")
            original_llm_vote = state.get("llm_vote")
            return {**state, "final_confidence": final_confidence, "llm_vote": original_llm_vote, "llm_explanation": updated_explanation}

        confidence = calculate_confidence_score(state, prompt_style_used, use_transformer_prob_as_baseline_arg)
        return {**state, "final_confidence": confidence}
    except Exception as e:
        print(f"  ERROR calculating confidence score: {e}", file=sys.stderr)
        traceback.print_exc()
        # Assign baseline confidence if calculation fails
        error_baseline = CONFIDENCE_BASELINE # Default
        # Try to use model-derived baseline even on error if flag is set and data available
        if use_transformer_prob_as_baseline_arg and state.get('all_model_predictions'):
            derived_baselines_on_error = []
            for pred_res_err in state.get('all_model_predictions', []):
                prob_err = pred_res_err.get("probability")
                status_err = pred_res_err.get("status", "")
                if prob_err is not None and not status_err.startswith('error') and not status_err.startswith('skipped'):
                    derived_baselines_on_error.append(0.5 + abs(prob_err - 0.5))
            if derived_baselines_on_error:
                error_baseline = sum(derived_baselines_on_error) / len(derived_baselines_on_error)

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
        if hasattr(obj, 'any') and np.isnan(obj).any(): # Handle NaN specifically for arrays/numbers
            warnings.warn(f"Converting numpy NaN to string 'NaN' for JSON serialization.")
            if isinstance(obj, np.ndarray):
                 return np.where(np.isnan(obj), "NaN", obj).tolist()
            else: return "NaN" # Single numpy NaN
        elif hasattr(obj, 'any') and np.isinf(obj).any():
             warnings.warn(f"Converting numpy Inf to string 'Infinity'/' -Infinity' for JSON.")
             if isinstance(obj, np.ndarray):
                 return np.where(np.isposinf(obj), "Infinity", np.where(np.isneginf(obj), "-Infinity", obj)).tolist()
             else: return "Infinity" if np.isposinf(obj) else "-Infinity" # Single numpy Inf
        return obj.tolist() if isinstance(obj, np.ndarray) else obj.item() # Convert numpy numbers to standard types
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        # Convert NaN/Inf within pandas objects before converting to dict
        obj_copy = obj.copy() # Avoid modifying original df/series if passed by reference
        obj_copy = obj_copy.replace([np.inf, -np.inf], ["Infinity", "-Infinity"]).fillna("NaN_pandas")
        if isinstance(obj_copy, pd.DataFrame):
            return obj_copy.to_dict(orient='records')
        else: # Series
            return obj_copy.to_dict()
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
def should_run_direct_analysis(state: AgentState) -> str:
    print("\n--- Edge Logic: should_run_direct_analysis ---")
    if state.get("error"):
        print("  --> Error in state, routing to error_path.")
        return "error_path"
    
    if state.get("ablate_direct_analysis", False): # Check the flag from AgentState
        print("  --> Direct analysis is ABLATED by flag. Routing to skip_direct_path.")
        return "skip_direct_path"
    else:
        print("  --> Direct analysis is ENABLED. Routing to do_direct_path.")
        return "do_direct_path"
    
def decide_direct_motif_path(state: AgentState) -> str:
    print("\n--- Edge Logic: decide_direct_motif_path ---");
    if state.get("error"): return "error_path";
    fetch_results = state.get("direct_motif_fetch_results", {})
    if fetch_results.get('status', '').startswith('success') and fetch_results.get('motifs_metadata'): print("  --> Direct motifs found, proceed to scan DNA."); return "scan_direct"
    else: print(f"  --> No direct motifs found/fetched (status: {fetch_results.get('status', 'unknown')}), skip direct scan."); return "summarize_direct"

def decide_indirect_motif_path(state: AgentState) -> str:
    print("\n--- Edge Logic: decide_indirect_motif_path ---"); # ... (rest of implementation unchanged) ...
    if state.get("error"): return "error_path";
    tfs_to_analyze = state.get("tfs_analyzed_indirectly", [])
    if not tfs_to_analyze: print("  --> No relevant interacting TFs found to analyze. Skip indirect scan."); return "summarize_indirect"
    fetch_results = state.get("indirect_motif_fetch_results", {})
    if fetch_results.get('motifs_metadata'): print("  --> Indirect TFs identified and motifs fetched, proceed to scan DNA."); return "scan_indirect"
    else: print(f"  --> Indirect TFs identified, but failed/found no motifs (status: {fetch_results.get('status', 'unknown')}). Skip indirect scan."); return "summarize_indirect"

def should_run_indirect_analysis(state: AgentState) -> str:
    print("\n--- Edge Logic: should_run_indirect_analysis ---")
    if state.get("error"):
        print("  --> Error in state, routing to error_path.")
        return "error_path"
    
    if state.get("ablate_indirect_analysis", False): # Check the flag from AgentState
        print("  --> Indirect analysis is ABLATED by flag. Routing to skip_indirect_path.")
        return "skip_indirect_path"
    else:
        print("  --> Indirect analysis is ENABLED. Routing to do_indirect_path.")
        return "do_indirect_path"
    
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
        default=f"hf/{DEFAULT_HF_MODEL}" if (TRANSFORMERS_AVAILABLE or UNSLOTH_AVAILABLE) else f"gemini/{DEFAULT_GEMINI_MODEL}",
        help=f"LLM ID. Use prefix 'gemini/', 'ollama/', or 'hf/'. For 'hf/', provide Hub ID or LOCAL PATH. Default: HF if available, else Gemini."
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
        "--prompt-style", type=str, default="verbose", choices=["verbose", "concise", "transformer-priority", "motif-priority", "auto"],
        help="Style of prompt to generate for the LLM. 'concise' for fine-tuned. 'auto' switches based on protein novelty relative to run models."
    )
    parser.add_argument(
        "--use-transformer-prob-as-baseline",
        action="store_true",
        help="If set, use the (average if multiple) model probability (converted to a certainty measure) as the baseline for confidence score calculation, overriding style-based fixed baselines."
    )
    parser.add_argument("--random-state", type=int, default=13, help="random state for randomization of the evaluation file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N samples from the evaluation file.")
    parser.add_argument(
        "--api-delay", type=float, default=1.5,
        help="Delay (in seconds) between Gemini API calls to avoid rate limits. Default: 1.5"
    )
    parser.add_argument(
        "--prediction-model-types", type=str, default="dpi",
        help="Comma-separated list of prediction models to run (e.g., 'dpi,sei'). Choices: dpi, deepsea, sei. Default: dpi"
    )
    parser.add_argument(
        "--no-indirect-analysis",
        action="store_true",
        help="Disable indirect motif analysis (STRING query, interactor motifs) for ablation study."
    )
    parser.add_argument(
        "--no-direct-analysis",
        action="store_true",
        help="Disable direct motif analysis (protein's own motifs) for ablation study."
    )    
    # DeepSEA specific arguments
    parser.add_argument("--deepsea-model-path", type=str, default=None, help="Path to the DeepSEA model .pkl state_dict file.")
    parser.add_argument("--deepsea-protein-map-path", type=str, default=None, help="Path to DeepSEA protein_to_id_map.json file.")
    parser.add_argument("--deepsea-dna-len", type=int, default=512, help="DNA sequence length for DeepSEA model.")
    parser.add_argument("--deepsea-protein-emb-dim", type=int, default=50, help="Protein embedding dimension for DeepSEA.")
    parser.add_argument("--deepsea-cnn-l-out-factor", type=int, default=None, help="CNN L_out factor for DeepSEA. Auto-calculated if None.")

    # SEI specific arguments ---
    parser.add_argument("--sei-model-path", type=str, default=None, help="Path to the SEI model .pkl state_dict file.")
    parser.add_argument("--sei-protein-map-path", type=str, default=None, help="Path to SEI protein_to_id_map.json file.")
    parser.add_argument("--sei-dna-len", type=int, default=512, help="DNA sequence length for SEI model.")
    parser.add_argument("--sei-protein-emb-dim", type=int, default=50, help="Protein embedding dimension for SEI model.")
    parser.add_argument("--verbose-debug-unknown-handling", action="store_true", help="Enable verbose logging from ProteinDNADatasetInference during unknown handling.") # Add this arg

    args = parser.parse_args()

    # --- Parse --prediction-model-types ---
    selected_model_types_str = args.prediction_model_types
    if not selected_model_types_str:
        print("ERROR: --prediction-model-types cannot be empty. Must specify at least one model. Exiting.", file=sys.stderr)
        sys.exit(1)

    valid_model_choices = ["dpi", "deepsea", "sei"]
    models_to_run_arg = [model.strip().lower() for model in selected_model_types_str.split(',') if model.strip()] # Filter out empty strings

    for model_type in models_to_run_arg:
        if model_type not in valid_model_choices:
            print(f"ERROR: Invalid model type '{model_type}' in --prediction-model-types. Choices are {valid_model_choices}. Exiting.", file=sys.stderr)
            sys.exit(1)

    if not models_to_run_arg:
        print("ERROR: No valid model types parsed from --prediction-model-types. Must specify at least one model (e.g., 'dpi'). Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Selected prediction models to run: {models_to_run_arg}")


    # --- Create Output Directory ---
    selected_llm_safe = re.sub(r'[^\w\-_\.]', '_', args.llm_model.replace('/', '_'))
    # Include selected prediction models in subdir name
    models_run_str_safe = "_".join(sorted(models_to_run_arg))

    direct_ablation_subdir_suffix = "_no_direct" if args.no_direct_analysis else ""
    indirect_ablation_subdir_suffix = "_no_indirect" if args.no_indirect_analysis else ""
    if args.no_direct_analysis and args.no_indirect_analysis:
        ablation_subdir_suffix = "_no_motif"
    else:
        ablation_subdir_suffix = f"{direct_ablation_subdir_suffix}{indirect_ablation_subdir_suffix}"

    output_subdir_name_for_run = f"{selected_llm_safe}_prompt_{args.prompt_style}_models_{models_run_str_safe}{ablation_subdir_suffix}"
    output_dir_for_run = os.path.join(args.output_dir, output_subdir_name_for_run) # This is where individual sample JSONs go
    try:
        os.makedirs(output_dir_for_run, exist_ok=True)
        # print(f"Output directory for individual samples: {output_dir_for_run}") # Already prints
    except OSError as e:
        print(f"ERROR: Could not create output directory '{output_dir_for_run}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Determine Device ---
    if args.force_cpu:
        DEVICE = torch.device('cpu')
        print("Forcing CPU usage.")
    else:
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Load DNABERT Model & Tokenizer ---
    print("--- Loading DNABERT Resources for Embeddings ---")
    dnabert_model = None
    dnabert_tokenizer = None
    # Only load DNABERT if DPI model is selected, as it's primarily for DPI's on-the-fly DNA embedding
    if "dpi" in models_to_run_arg:
        try:
            print(f"Loading DNABERT tokenizer from Hugging Face Hub: zhihan1996/DNA_bert_6")
            dnabert_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)

            print(f"Loading DNABERT model (BertModel) from Hugging Face Hub: zhihan1996/DNA_bert_6")
            dnabert_model = BertModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)

            dnabert_model.to(DEVICE)
            dnabert_model.eval()
            print(f"DNABERT {DNABERT_KMER}-mer model and tokenizer for embeddings loaded successfully from Hub to {DEVICE}.")
        except Exception as e:
            print(f"ERROR loading DNABERT model/tokenizer for embeddings from Hugging Face Hub: {e}", file=sys.stderr)
            traceback.print_exc()
            dnabert_model = None; dnabert_tokenizer = None
            print("WARN: DNABERT (for DPI on-the-fly DNA embeddings) failed to load. DPI predictions requiring on-the-fly DNA embedding will fail.")
    else:
        print("DPI model not selected, skipping DNABERT resource loading for on-the-fly embeddings.")

    # --- NEW: Load shared resources for unknown protein handling ---
    print("--- Loading Shared Resources for Unknown Protein Handling ---")
    if not Path(UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG).exists():
        Path(UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG).mkdir(parents=True, exist_ok=True)

    # 1. Parse global FASTA
    if Path(FASTA_FILE_PATH_CONFIG).exists():
        GLOBAL_PROTEIN_SEQUENCES_MAP = parse_fasta_for_gene_names(FASTA_FILE_PATH_CONFIG)
        # Convert all keys to uppercase immediately
        GLOBAL_PROTEIN_SEQUENCES_MAP = {k.upper(): v for k, v in GLOBAL_PROTEIN_SEQUENCES_MAP.items()}
    else:
        print(f"WARNING: FASTA file not found at {FASTA_FILE_PATH_CONFIG}. Sequence-based similarity will be limited.")

    # 2. Load TF Family Data (shared part)
    if Path(TF_FAMILY_FILE_PATH_CONFIG).exists():
        try:
            tf_family_df = pd.read_csv(TF_FAMILY_FILE_PATH_CONFIG)
            tf_family_df['protein'] = tf_family_df['protein'].astype(str).str.upper()
            tf_family_df['family'] = tf_family_df['family'].astype(str).str.upper()
            PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED = pd.Series(
                tf_family_df['family'].values, index=tf_family_df['protein']
            ).to_dict()
            print(f"Loaded {len(PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED)} protein-to-family mappings.")
        except Exception as e_tf:
            print(f"ERROR loading or processing TF family file {TF_FAMILY_FILE_PATH_CONFIG}: {e_tf}")
    else:
        print(f"WARNING: TF Family file {TF_FAMILY_FILE_PATH_CONFIG} not found. TF family fallback inactive.")

    # --- Load DeepSEA Model & Resources (if selected) ---
    # --- Load DeepSEA Model & Resources (if selected) ---
    deepsea_model_instance = None
    deepsea_protein_to_id_map = None
    deepsea_avg_embedding = None
    NUM_PROTEINS_DEEPSEA = 0

    if "deepsea" in models_to_run_arg:
        if not args.deepsea_model_path or not args.deepsea_protein_map_path:
            print("ERROR: --deepsea-model-path and --deepsea-protein-map-path are required when 'deepsea' is in --prediction-model-types. Exiting.", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.deepsea_protein_map_path, 'r') as f: deepsea_protein_to_id_map = json.load(f)
            NUM_PROTEINS_DEEPSEA = len(deepsea_protein_to_id_map)
            if NUM_PROTEINS_DEEPSEA == 0: raise ValueError("DeepSEA protein map is empty.")
            print(f"Loaded DeepSEA protein_to_id_map with {NUM_PROTEINS_DEEPSEA} proteins from {args.deepsea_protein_map_path}.")

            cnn_l_out_ds = args.deepsea_cnn_l_out_factor
            if cnn_l_out_ds is None: _l = args.deepsea_dna_len; _l = _l-7; _l = _l//4; _l = _l-7; _l = _l//4; _l = _l-7; cnn_l_out_ds = _l; print(f"  Auto-calculated DeepSEA cnn_l_out: {cnn_l_out_ds}")

            deepsea_model_instance = DeepSEAProteinInteraction(num_proteins=NUM_PROTEINS_DEEPSEA, protein_emb_dim=args.deepsea_protein_emb_dim, l_in=args.deepsea_dna_len, cnn_l_out_factor=cnn_l_out_ds).to(DEVICE)
            deepsea_model_instance.load_state_dict(torch.load(args.deepsea_model_path, map_location=DEVICE)); deepsea_model_instance.eval()
            print(f"DeepSEA model loaded from {args.deepsea_model_path} to {DEVICE}.")

            with torch.no_grad(): all_known_protein_embeddings_ds = deepsea_model_instance.protein_embedding.weight.data.clone(); deepsea_avg_embedding = torch.mean(all_known_protein_embeddings_ds, dim=0).to(DEVICE)
            print(f"  Calculated DeepSEA average embedding for unknown proteins (shape: {deepsea_avg_embedding.shape})")
        except Exception as e: print(f"FATAL ERROR loading DeepSEA resources: {e}", file=sys.stderr); traceback.print_exc(); sys.exit(1)

        MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"] = {
            "protein_id_to_model_id_map": deepsea_protein_to_id_map, # gene_name_UC -> int_id
            "known_protein_ids_for_model": list(deepsea_protein_to_id_map.keys()), # list of gene_name_UC
            "protein_id_to_embedding_map_for_fallback": {}, # gene_name_UC -> cpu_tensor
            "fallback_cpu_embedding": None, # cpu_tensor
            "tf_family_to_model_known_protein_ids_map": {},
            "mmseqs_db_path_prefix": Path(UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG) / "known_deepsea_mmseqs_db",
            "blast_db_path_prefix": Path(UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG) / "known_deepsea_blast_db",
        }
        if deepsea_model_instance and hasattr(deepsea_model_instance, 'protein_embedding'):
            all_embs = deepsea_model_instance.protein_embedding.weight.data.cpu().clone()
            MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["fallback_cpu_embedding"] = torch.mean(all_embs, dim=0)
            inv_map = {v: k for k,v in deepsea_protein_to_id_map.items()}
            for i in range(all_embs.shape[0]):
                gene_name = inv_map.get(i)
                if gene_name:
                    MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["protein_id_to_embedding_map_for_fallback"][gene_name] = all_embs[i]
        # Populate TF_FAMILY_TO_MODEL_KNOWN_PROTEIN_IDS_MAP_DEEPSEA
        for prot_id_uc, fam_name_uc in PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED.items():
            if prot_id_uc in MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["known_protein_ids_for_model"]:
                if fam_name_uc not in MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["tf_family_to_model_known_protein_ids_map"]:
                    MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["tf_family_to_model_known_protein_ids_map"][fam_name_uc] = []
                MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["tf_family_to_model_known_protein_ids_map"][fam_name_uc].append(prot_id_uc)

    else:
        print("DeepSEA model not selected, skipping its resource loading.")


    # --- Load SEI Model & Resources (if selected) ---
    sei_model_instance = None
    sei_protein_to_id_map = None
    NUM_PROTEINS_SEI = 0
    sei_avg_embedding = None # MODIFIED: Initialize sei_avg_embedding

    if "sei" in models_to_run_arg:
        if not args.sei_model_path or not args.sei_protein_map_path:
            print("ERROR: --sei-model-path and --sei-protein-map-path are required when 'sei' is in --prediction-model-types. Exiting.", file=sys.stderr)
            sys.exit(1)

        print("--- Loading SEI Resources ---")
        try:
            with open(args.sei_protein_map_path, 'r') as f: sei_protein_to_id_map = json.load(f)
            NUM_PROTEINS_SEI = len(sei_protein_to_id_map)
            if NUM_PROTEINS_SEI == 0: raise ValueError("SEI protein map is empty.")
            print(f"Loaded SEI protein_to_id_map with {NUM_PROTEINS_SEI} proteins from {args.sei_protein_map_path}.")

            sei_model_instance = SeiProteinInteractionWithMeanEmbedding(
                sequence_length=args.sei_dna_len,
                num_proteins=NUM_PROTEINS_SEI,
                protein_emb_dim=args.sei_protein_emb_dim,
                n_genomic_features_output=1
            ).to(DEVICE)            
            sei_model_instance.load_state_dict(torch.load(args.sei_model_path, map_location=DEVICE)); sei_model_instance.eval()
            print(f"SEI model loaded from {args.sei_model_path} to {DEVICE}.")

            # MODIFIED: Calculate average embedding for SEI
            if hasattr(sei_model_instance, 'protein_embedding') and \
               isinstance(sei_model_instance.protein_embedding, torch.nn.Embedding):
                with torch.no_grad():
                    all_known_protein_embeddings_sei = sei_model_instance.protein_embedding.weight.data.clone()
                    if all_known_protein_embeddings_sei.numel() > 0 and all_known_protein_embeddings_sei.shape[0] > 0:
                        sei_avg_embedding = torch.mean(all_known_protein_embeddings_sei, dim=0).to(DEVICE)
                        print(f"  Calculated SEI average embedding for unknown proteins (shape: {sei_avg_embedding.shape})")
                    else:
                        print(f"  WARN: SEI model's protein_embedding layer has no weights or is empty. Cannot calculate average embedding.", file=sys.stderr)
                        sei_avg_embedding = None
            else:
                print("  WARN: SEI model does not have an accessible 'protein_embedding' layer or it's not nn.Embedding. "
                      "Cannot calculate average embedding for unknown proteins. SEI will use fallback for novel proteins.", file=sys.stderr)
                sei_avg_embedding = None

        except Exception as e: print(f"FATAL ERROR loading SEI resources: {e}", file=sys.stderr); traceback.print_exc(); sys.exit(1)
        # ... (sei_model_instance, sei_protein_to_id_map loaded) ...
        MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"] = {
            "protein_id_to_model_id_map": sei_protein_to_id_map,
            "known_protein_ids_for_model": list(sei_protein_to_id_map.keys()),
            "protein_id_to_embedding_map_for_fallback": {},
            "fallback_cpu_embedding": None,
            "tf_family_to_model_known_protein_ids_map": {},
            "mmseqs_db_path_prefix": Path(UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG) / "known_sei_mmseqs_db",
            "blast_db_path_prefix": Path(UNKNOWN_HANDLING_TEMP_DIR_BASE_CONFIG) / "known_sei_blast_db",
        }
        if sei_model_instance and hasattr(sei_model_instance, 'protein_embedding'):
            all_embs_sei = sei_model_instance.protein_embedding.weight.data.cpu().clone()
            MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["fallback_cpu_embedding"] = torch.mean(all_embs_sei, dim=0)
            inv_map_sei = {v:k for k,v in sei_protein_to_id_map.items()}
            for i in range(all_embs_sei.shape[0]):
                gene_name = inv_map_sei.get(i)
                if gene_name:
                     MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["protein_id_to_embedding_map_for_fallback"][gene_name] = all_embs_sei[i]
        # Populate TF_FAMILY_TO_MODEL_KNOWN_PROTEIN_IDS_MAP_SEI
        for prot_id_uc, fam_name_uc in PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED.items():
            if prot_id_uc in MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["known_protein_ids_for_model"]:
                if fam_name_uc not in MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["tf_family_to_model_known_protein_ids_map"]:
                    MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["tf_family_to_model_known_protein_ids_map"][fam_name_uc] = []
                MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["tf_family_to_model_known_protein_ids_map"][fam_name_uc].append(prot_id_uc)
    
    else:
        print("SEI model not selected, skipping its resource loading.")

    # --- Fetch missing sequences for all relevant proteins (from model maps) ---
    # This should be done after all model-specific protein maps are loaded
    all_model_protein_ids_to_fetch = set()
    if "deepsea" in models_to_run_arg:
        all_model_protein_ids_to_fetch.update(MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["deepsea"]["known_protein_ids_for_model"])
    if "sei" in models_to_run_arg:
        all_model_protein_ids_to_fetch.update(MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING["sei"]["known_protein_ids_for_model"])

    if FETCH_MISSING_SEQUENCES_FROM_UNIPROT_CONFIG and all_model_protein_ids_to_fetch:
        print(f"Checking/Fetching sequences for {len(all_model_protein_ids_to_fetch)} known model protein IDs...")
        fetched_count_shared = 0
        for prot_id_uc in tqdm(list(all_model_protein_ids_to_fetch), desc="Fetching Known Model Protein Sequences"):
            if not prot_id_uc or not isinstance(prot_id_uc, str): continue
            if prot_id_uc.upper() not in GLOBAL_PROTEIN_SEQUENCES_MAP or not GLOBAL_PROTEIN_SEQUENCES_MAP[prot_id_uc.upper()]:
                seq = fetch_uniprot_sequence_by_gene_or_id(prot_id_uc, organism_id=UNIPROT_ORGANISM_ID_FOR_FETCH_CONFIG)
                if seq: GLOBAL_PROTEIN_SEQUENCES_MAP[prot_id_uc.upper()] = seq; fetched_count_shared += 1
        print(f"Fetched {fetched_count_shared} new sequences for known model proteins.")
    # --- Finished loading shared and model-specific resources for unknown handling ---

    # --- Validate LLM Choice & Dependencies ---
    selected_llm = args.llm_model
    ollama_api_url_arg = args.ollama_url
    hf_model_id = None; hf_model = None; hf_tokenizer = None
    print(f"Selected LLM Identifier: {selected_llm}")
    if selected_llm.startswith("gemini/"):
        if not GOOGLE_API_KEY_CONFIGURED: print("ERROR: Gemini model selected, but GOOGLE_API_KEY is not set. Exiting.", file=sys.stderr); sys.exit(1)
    elif selected_llm.startswith("ollama/"): print(f"Using Ollama API URL: {ollama_api_url_arg}")
    elif selected_llm.startswith("hf/"):
        if not TRANSFORMERS_AVAILABLE and not UNSLOTH_AVAILABLE: print("ERROR: HF model selected, but neither 'transformers' nor 'unsloth' library is available.", file=sys.stderr); sys.exit(1)
        hf_model_id = selected_llm.split('/', 1)[1]
        print(f"Attempting to load HF model/path: {hf_model_id}")
        try:
            print("Loading tokenizer..."); hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=args.hf_token); print("Tokenizer loaded.")
            print("Loading model..."); load_start_time = time.time(); model_loaded_by_unsloth = False
            if UNSLOTH_AVAILABLE and DEVICE.type == 'cuda':
                print("Attempting optimized loading with Unsloth...")
                try:
                    hf_model, hf_tokenizer = unsloth.FastLanguageModel.from_pretrained(model_name = hf_model_id, dtype = None, load_in_4bit = True, token = args.hf_token, device_map = "auto")
                    print("Successfully loaded model with Unsloth."); model_loaded_by_unsloth = True
                except Exception as e_unsloth: print(f"WARN: Unsloth loading failed ({e_unsloth}). Will attempt standard Transformers loading."); hf_model = None
            if not model_loaded_by_unsloth:
                if not TRANSFORMERS_AVAILABLE: print("ERROR: Standard Transformers library not available, cannot load model.", file=sys.stderr); sys.exit(1)
                print("Using standard Hugging Face transformers loading...")
                quantization_config = None
                if DEVICE.type == 'cuda': print("Configuring standard 4-bit quantization..."); quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                elif DEVICE.type == 'cpu': print("WARN: Loading HF model on CPU.")
                hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.bfloat16 if DEVICE.type == 'cuda' and quantization_config is None else None, token=args.hf_token, trust_remote_code=True)
                print("Standard Hugging Face model loaded.")
            load_end_time = time.time(); print(f"Model loading took {load_end_time - load_start_time:.2f} seconds."); hf_model.eval()
        except Exception as e: print(f"ERROR loading Hugging Face model '{hf_model_id}': {e}", file=sys.stderr); traceback.print_exc(); print("Cannot proceed without the LLM. Exiting."); sys.exit(1)
    else: print(f"ERROR: Invalid --llm-model format: '{selected_llm}'. Must start with 'gemini/', 'ollama/', or 'hf/'. Exiting.", file=sys.stderr); sys.exit(1)

    # --- Resource Loading (Domain Specific - CisBP, Embeddings, DPI Model) ---
    print("--- Loading Domain Specific Resources (CisBP, DPI-related Embeddings & Model) ---")
    TRANSFORMER_MODEL_PATH = '/new-stg/home/cong/DPI/scripts/model2_Transformer/v5/output/model/main_singletask_Encode3and4_all_847_proteins-lr=0.05,epoch=2,dropout=0.2:0,hid_dim=240,n_layer=2,n_heads=6,batch=128,input=train_min10,max_dna=512,max_protein=768,mixed_precision=False.pt'
    DNA_EMB_DB_PATH = '/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/embeddings/valid_min10/dna_embeddings.duckdb'
    PRO_EMB_PATH = '/new-stg/home/cong/DPI/dataset/all_human_tfs_protein_embedding_mean_deduplicated.pkl'
    TRAINING_PROTEIN_LIST_PATH = '/new-stg/home/cong/DPI/dataset/Encode3and4/proteins.txt'
    CISBP_BASE_DIR = '/new-stg/home/cong/DPI/scripts/deepseek/motif_database/'
    CISBP_TF_INFO_VARIANT = 'standard'

    cisbp_tf_info_df = None; cisbp_pwm_dir = None
    try:
        if os.path.isdir(CISBP_BASE_DIR):
             cisbp_pwm_dir = os.path.join(CISBP_BASE_DIR, 'pwms_all_motifs');
             if not os.path.isdir(cisbp_pwm_dir): raise FileNotFoundError(f"PWM directory not found: {cisbp_pwm_dir}")
             tf_info_filename = 'TF_Information.txt';
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

    pro_embs_dict = None; pro_embs_df = None;
    if "dpi" in models_to_run_arg: # DPI protein embeddings
        try:
            if os.path.exists(PRO_EMB_PATH):
                print(f"Loading protein embeddings file (for DPI): {PRO_EMB_PATH}"); pro_embs_df = pd.read_pickle(PRO_EMB_PATH); required_cols = ['protein', 'protein_embedding']
                if not all(col in pro_embs_df.columns for col in required_cols): print(f"ERROR: Protein embedding file {PRO_EMB_PATH} missing required columns: {required_cols}", file=sys.stderr)
                else:
                    protein_counts = pro_embs_df['protein'].value_counts(); duplicate_proteins = protein_counts[protein_counts > 1].index.tolist()
                    if duplicate_proteins:
                        print(f"WARN: Found {len(duplicate_proteins)} protein names with multiple entries in DPI embeddings. Removing ALL entries for these proteins.");
                        filtered_df = pro_embs_df[~pro_embs_df['protein'].isin(duplicate_proteins)].copy();
                        if not filtered_df.empty: pro_embs_dict = filtered_df.set_index('protein')['protein_embedding'].to_dict(); print(f"Loaded {len(pro_embs_dict)} DPI protein embeddings after removing duplicates.")
                        else: print("WARN: No DPI embeddings remaining after removing all duplicates."); pro_embs_dict = {}
                    else:
                        if pro_embs_df['protein'].is_unique: pro_embs_dict = pro_embs_df.set_index('protein')['protein_embedding'].to_dict(); print(f"Loaded {len(pro_embs_dict)} DPI protein embeddings.")
                        else: print("ERROR: Inconsistency in DPI embeddings - duplicates list empty but column not unique.", file=sys.stderr); pro_embs_dict = None
            else: print(f"WARN: DPI Protein embeddings file not found: {PRO_EMB_PATH}", file=sys.stderr)
        except Exception as e: print(f"ERROR loading DPI protein embeddings: {e}", file=sys.stderr); traceback.print_exc(); pro_embs_dict = None

    dna_db_con = None;
    if "dpi" in models_to_run_arg: # DPI DNA embeddings DB
        try:
            if os.path.exists(DNA_EMB_DB_PATH): dna_db_con = duckdb.connect(database=DNA_EMB_DB_PATH, read_only=True); print(f"Connected to DNA embeddings DB (for DPI): {DNA_EMB_DB_PATH}")
            else: print(f"WARN: DNA embedding DB (for DPI) not found: {DNA_EMB_DB_PATH}", file=sys.stderr)
        except Exception as e: print(f"ERROR connecting to DNA embeddings DB (for DPI): {e}", file=sys.stderr)

    training_protein_set = set();
    if "dpi" in models_to_run_arg: # DPI training protein list
        try:
            if os.path.exists(TRAINING_PROTEIN_LIST_PATH):
                 with open(TRAINING_PROTEIN_LIST_PATH, 'r') as f: training_proteins = [line.strip() for line in f if line.strip()]
                 training_protein_set = set(training_proteins); print(f"Loaded {len(training_protein_set)} DPI training proteins.")
            else: print(f"WARN: DPI Training protein list not found: {TRAINING_PROTEIN_LIST_PATH}", file=sys.stderr)
        except Exception as e: print(f"ERROR loading DPI training protein list: {e}", file=sys.stderr)

    transformer_model = None; transformer_config = None;
    if "dpi" in models_to_run_arg: # DPI model itself
        try:
            transformer_config = eval_get_config(); keys_to_remove = ['warmup', 'iteration_per_split', 'files_per_split', 'valid_dna_size', 'lr'];
            for key in keys_to_remove: transformer_config.pop(key, None)
            transformer_config['return_attention'] = False; transformer_config['device'] = DEVICE;
            matches = re.findall(r'([a-zA-Z_]+)=([\d.]+)', os.path.basename(TRANSFORMER_MODEL_PATH)); parsed = {key: (int(v) if v.isdigit() else float(v) if '.' in v else v) for key, v in matches}
            if 'max_dna' in parsed and 'max_dna_seq' not in parsed: parsed['max_dna_seq'] = parsed.pop('max_dna')
            if 'max_protein' in parsed and 'max_protein_seq' not in parsed: parsed['max_protein_seq'] = parsed.pop('max_protein')
            transformer_config.update(parsed); print(f"Using DPI Transformer Config: {transformer_config}")
            transformer_model = Predictor(**transformer_config)
            state_dict = torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE)
            if any(key.startswith('module.') for key in state_dict.keys()): print("Adjusting DPI state_dict keys (removing 'module.' prefix)..."); new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()]); transformer_model.load_state_dict(new_state_dict)
            else: transformer_model.load_state_dict(state_dict)
            transformer_model.to(DEVICE); transformer_model.eval(); print("DPI Transformer model loaded.")
        except Exception as e: print(f"ERROR loading DPI Transformer model: {e}", file=sys.stderr); traceback.print_exc(); transformer_model = None # Ensure it's None on failure

    print("--- Domain Resource Loading Complete ---")

    # --- Check essential resources for selected models ---
    if "dpi" in models_to_run_arg and (pro_embs_dict is None or transformer_model is None): # dna_db_con can be None if only on-the-fly is used
         print("\nFATAL ERROR: Essential DPI resources (protein embeddings or model) failed to load. Cannot proceed.", file=sys.stderr); sys.exit(1)
    if "deepsea" in models_to_run_arg and (deepsea_model_instance is None or deepsea_protein_to_id_map is None):
         print("\nFATAL ERROR: Essential DeepSEA resources failed to load. Cannot proceed.", file=sys.stderr); sys.exit(1)
    if "sei" in models_to_run_arg and (sei_model_instance is None or sei_protein_to_id_map is None): # sei_avg_embedding can be None if calculation failed, handled by run_sei_prediction
         print("\nFATAL ERROR: Essential SEI resources (model or protein map) failed to load. Cannot proceed.", file=sys.stderr); sys.exit(1)

    if selected_llm.startswith("hf/") and (hf_model is None or hf_tokenizer is None):
        print("\nFATAL ERROR: Failed to load the selected Hugging Face LLM. Cannot proceed.", file=sys.stderr); sys.exit(1)
    if cisbp_tf_info_df is None or not cisbp_pwm_dir:
         print("\nFATAL ERROR: Essential CisBP resources failed to load. Cannot proceed.", file=sys.stderr); sys.exit(1)


    # --- Load, RENAME, Shuffle, and Prepare Evaluation Data ---
    print(f"--- Loading Evaluation Data from: {args.evaluation_file} ---")
    eval_inputs = []
    try:
        if args.evaluation_file.endswith('.pkl'): df_eval_raw = pd.read_pickle(args.evaluation_file); print(f"  Loaded {len(df_eval_raw)} raw samples from pickle.")
        elif args.evaluation_file.endswith('.tsv'): df_eval_raw = pd.read_csv(args.evaluation_file, sep='\t'); print(f"  Loaded {len(df_eval_raw)} raw samples from tsv.")
        else: raise ValueError("Unsupported evaluation file format. Use .pkl or .tsv")
        rename_map = {'dna': 'dna_sequence', 'protein': 'protein_name', 'label': 'ground_truth_label'}
        required_original_cols = list(rename_map.keys())
        if not all(col in df_eval_raw.columns for col in required_original_cols): raise ValueError(f"Evaluation file missing required original columns: {required_original_cols}")
        df_eval = df_eval_raw.rename(columns=rename_map); print(f"  Renamed columns to: {list(df_eval.columns)}")
        print(f"  Shuffling dataset with random_state={args.random_state}..."); df_eval = df_eval.sample(frac=1, random_state=args.random_state).reset_index(drop=True); print("  Dataset shuffled.")
        if args.limit is not None and args.limit >= 0:
             if args.limit < len(df_eval): df_eval = df_eval.head(args.limit); print(f"  Limiting processing to first {args.limit} shuffled samples.")
             else: print(f"  Limit ({args.limit}) is >= dataset size ({len(df_eval)}), processing all shuffled samples.")
        else: print(f"  Processing all {len(df_eval)} shuffled samples.")
        final_required_cols = list(rename_map.values()); eval_inputs = df_eval[final_required_cols].to_dict('records')
    except FileNotFoundError: print(f"ERROR: Evaluation file not found: {args.evaluation_file}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"ERROR loading or parsing evaluation file: {e}", file=sys.stderr); sys.exit(1)
    if not eval_inputs and args.limit != 0: print("ERROR: No evaluation samples loaded after processing. Exiting.", file=sys.stderr); sys.exit(1)
    elif args.limit == 0: print("Limit set to 0. No samples will be processed.")

    # --- Build Graph ---
    workflow = StateGraph(AgentState)

    # --- Add nodes, passing resources via partial ---
    # Start node now gets the list of models to run
    start_node_partial = partial(start_analysis,
                                 models_to_run_arg=models_to_run_arg,
                                 ablate_indirect_analysis_flag=args.no_indirect_analysis,
                                 ablate_direct_analysis_flag=args.no_direct_analysis) # <<< MODIFIED HERE
    workflow.add_node("start", start_node_partial)
    workflow.add_node("mark_direct_skipped_node", mark_direct_skipped_node)

    # Motif analysis nodes
    workflow.add_node("fetch_direct_motifs", partial(fetch_direct_motifs_node, cisbp_tf_info_df=cisbp_tf_info_df, cisbp_pwm_dir=cisbp_pwm_dir))
    workflow.add_node("scan_direct_dna", scan_direct_dna_node)
    workflow.add_node("summarize_direct_analysis", summarize_direct_analysis_node)

    workflow.add_node("mark_indirect_skipped_node", mark_indirect_skipped_node)
    workflow.add_node("query_string", query_string_node)
    workflow.add_node("filter_string_interactors", partial(filter_string_interactors_node, cisbp_tf_info_df=cisbp_tf_info_df))
    workflow.add_node("fetch_indirect_motifs", partial(fetch_indirect_motifs_node, cisbp_tf_info_df=cisbp_tf_info_df, cisbp_pwm_dir=cisbp_pwm_dir))
    workflow.add_node("scan_indirect_dna", scan_indirect_dna_node)
    workflow.add_node("summarize_indirect_analysis", summarize_indirect_analysis_node)

    # Prediction Model Nodes
    dpi_pred_partial = partial(
        get_dpi_pred_node_wrapper, dpi_transformer_model=transformer_model, dpi_pro_embs_dict=pro_embs_dict,
        dpi_dna_db_con=dna_db_con, dpi_transformer_config=transformer_config, dpi_device=DEVICE,
        dpi_training_protein_set=training_protein_set, dnabert_model=dnabert_model, dnabert_tokenizer=dnabert_tokenizer,
        dnabert_kmer=DNABERT_KMER, dnabert_max_len=DNABERT_MAX_LEN, dnabert_device=DEVICE
    )
    workflow.add_node("get_dpi_pred", dpi_pred_partial)

    deepsea_pred_partial = partial(
        get_deepsea_pred_node_wrapper,
        deepsea_model_instance=deepsea_model_instance,
        deepsea_model_specific_resources_uh=MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING.get("deepsea"), # Pass the dict
        global_sequences_map_uh=GLOBAL_PROTEIN_SEQUENCES_MAP,
        tf_family_map_uh=PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED,
        deepsea_dna_seq_len_param=args.deepsea_dna_len,
        deepsea_device_param=DEVICE
    )
    workflow.add_node("get_deepsea_pred", deepsea_pred_partial)

    sei_pred_partial = partial(
        get_sei_pred_node_wrapper,
        sei_model_instance_param=sei_model_instance,
        sei_model_specific_resources_uh=MODEL_SPECIFIC_RESOURCES_FOR_UNKNOWN_HANDLING.get("sei"), # Pass the dict
        global_sequences_map_uh=GLOBAL_PROTEIN_SEQUENCES_MAP,
        tf_family_map_uh=PROTEIN_ID_TO_TF_FAMILY_MAP_SHARED,
        sei_dna_seq_len_param=args.sei_dna_len,
        sei_device_param=DEVICE
    )
    workflow.add_node("get_sei_pred", sei_pred_partial)

    # LLM and Confidence nodes
    workflow.add_node("generate_llm_prompt", partial(generate_llm_prompt_node, requested_prompt_style=args.prompt_style, llm_identifier=selected_llm, finetuned_model_id=FINETUNED_LOCAL_MODEL_PATH_ID))
    workflow.add_node("get_llm_pred", partial(get_llm_pred_node, llm_model_name=selected_llm, ollama_api_url=ollama_api_url_arg, hf_model=hf_model, hf_tokenizer=hf_tokenizer, device=DEVICE, api_delay=args.api_delay))
    workflow.add_node("calculate_final_confidence", partial(calculate_final_confidence_node, use_transformer_prob_as_baseline_arg=args.use_transformer_prob_as_baseline))
    workflow.add_node("handle_error", lambda state: {**state, "llm_vote": None, "final_confidence": 0.0, "llm_explanation": f"Processing stopped due to error: {state.get('error', 'Unknown error')}"})

    # --- Define Edges ---
    workflow.set_entry_point("start")
    workflow.add_conditional_edges(
        "start",
        should_run_direct_analysis, # New conditional function for direct analysis
        {
            "do_direct_path": "fetch_direct_motifs",        # Normal path for direct
            "skip_direct_path": "mark_direct_skipped_node", # Path if ablating direct
            "error_path": "handle_error"
        }
    )
    workflow.add_conditional_edges("fetch_direct_motifs", decide_direct_motif_path, {"scan_direct": "scan_direct_dna", "summarize_direct": "summarize_direct_analysis", "error_path": "handle_error"})
    workflow.add_edge("scan_direct_dna", "summarize_direct_analysis")

    common_indirect_decision_map = {
        "do_indirect_path": "query_string",  # Normal path to start indirect analysis
        "skip_indirect_path": "mark_indirect_skipped_node", # Path if ablating indirect (checks state.ablate_indirect_analysis)
        "error_path": "handle_error"
    }
    # If direct analysis ran, its summary node is the source for indirect decision
    workflow.add_conditional_edges("summarize_direct_analysis", should_run_indirect_analysis, common_indirect_decision_map)
    # If direct analysis was skipped, its skip_node is the source for indirect decision
    workflow.add_conditional_edges("mark_direct_skipped_node", should_run_indirect_analysis, common_indirect_decision_map)

    workflow.add_edge("query_string", "filter_string_interactors")
    workflow.add_edge("filter_string_interactors", "fetch_indirect_motifs")
    workflow.add_conditional_edges("fetch_indirect_motifs", decide_indirect_motif_path, {"scan_indirect": "scan_indirect_dna", "summarize_indirect": "summarize_indirect_analysis", "error_path": "handle_error"})
    workflow.add_edge("scan_indirect_dna", "summarize_indirect_analysis")

    # Chain the prediction model nodes. Each node internally decides if it runs.
    workflow.add_edge("summarize_indirect_analysis", "get_dpi_pred")
    workflow.add_edge("mark_indirect_skipped_node", "get_dpi_pred")
    workflow.add_edge("get_dpi_pred", "get_deepsea_pred")
    workflow.add_edge("get_deepsea_pred", "get_sei_pred")
    workflow.add_edge("get_sei_pred", "generate_llm_prompt") # Last model in chain connects to LLM prompt

    workflow.add_edge("generate_llm_prompt", "get_llm_pred")
    workflow.add_edge("get_llm_pred", "calculate_final_confidence")
    workflow.add_edge("calculate_final_confidence", END)
    workflow.add_edge("handle_error", END)

    # Compile the graph
    print("--- Compiling LangGraph Workflow ---")
    app = workflow.compile()
    print("--- Workflow Compiled ---")

    # --- Run Evaluation Loop & Save Outputs ---
    print(f"\n--- Running Evaluation (LLM: {selected_llm}, Prompt: {args.prompt_style}, Models: {models_to_run_arg}) ---")

    results_summary = []
    if not eval_inputs: print("No samples to process due to limit=0.")
    else:
        for i, input_data in enumerate(eval_inputs):
            protein_name = input_data.get("protein_name", f"unknown_protein_{i+1}")
            try: input_data['ground_truth_label'] = int(input_data['ground_truth_label'])
            except (ValueError, TypeError): print(f"WARN: Invalid ground truth '{input_data.get('ground_truth_label')}' for {protein_name}. Skipping sample {i+1}.", file=sys.stderr); continue

            print(f"\n--- Processing Sample {i+1}/{len(eval_inputs)} [{protein_name}] ---")
            start_time = time.time(); final_state = None

            safe_protein_name = re.sub(r'[^\w\-_\.]', '_', protein_name)
            run_identifier = f"sample{i+1:04d}"
            llm_name_safe = re.sub(r'[^\w\-_\.]', '_', selected_llm.replace('/', '_'))
            simple_output_path = os.path.join(output_dir_for_run, f"{run_identifier}_{safe_protein_name}_{llm_name_safe}_simple.json")
            comp_output_path = os.path.join(output_dir_for_run, f"{run_identifier}_{safe_protein_name}_{llm_name_safe}_comp.json")

            try:
                final_state = app.invoke(input_data, {"recursion_limit": 50})

                # --- MODIFIED simple_output to reflect multiple model probabilities ---
                # We can store all model probs in simple output or a primary one if defined, or average
                # For now, let's store them all if possible, or just indicate number of models contributing
                model_probs_for_simple_output = []
                if final_state.get("all_model_predictions"):
                    for pred_res_simple in final_state["all_model_predictions"]:
                        if pred_res_simple.get("probability") is not None and not pred_res_simple.get("status","").startswith("error"):
                             model_probs_for_simple_output.append({
                                 "model": pred_res_simple.get("model_name"),
                                 "prob": pred_res_simple.get("probability")
                             })

                simple_output = {
                    "protein_name": final_state.get("protein_name"),
                    "dna_sequence_length": len(final_state.get("dna_sequence", "")),
                    "ground_truth_label": final_state.get("ground_truth_label"),
                    "predicted_label": final_state.get("llm_vote"),
                    "confidence_score": final_state.get("final_confidence"),
                    "model_probabilities_run": model_probs_for_simple_output, # List of dicts
                    "llm_explanation": final_state.get("llm_explanation"),
                    "error": final_state.get("error")
                }
                results_summary.append(simple_output)

                try:
                    with open(simple_output_path, 'w') as f: json.dump(simple_output, f, indent=4)
                except Exception as e_save: print(f"  ERROR saving simple output JSON for {protein_name}: {e_save}", file=sys.stderr)

                try:
                    serializable_state = make_serializable(final_state)
                    with open(comp_output_path, 'w') as f: json.dump(serializable_state, f, indent=2)
                except Exception as e_save: print(f"  ERROR saving comprehensive output JSON for {protein_name}: {e_save}", file=sys.stderr); traceback.print_exc(file=sys.stderr)

            except Exception as e_invoke:
                error_msg = f"Agent invocation failed: {e_invoke}"
                print(f"\nFATAL ERROR during agent invocation {i+1}: {e_invoke}", file=sys.stderr); traceback.print_exc();
                final_state = {"error": error_msg, **input_data, "llm_vote": None, "final_confidence": 0.0, "llm_explanation": f"Error: {error_msg}."};

                simple_output_error = {
                    "protein_name": input_data.get("protein_name"),
                    "dna_sequence_length": len(input_data.get("dna_sequence", "")),
                    "ground_truth_label": input_data.get("ground_truth_label"),
                    "predicted_label": None, "confidence_score": 0.0,
                    "model_probabilities_run": [], # No successful model runs
                    "llm_explanation": final_state.get("llm_explanation"),
                    "error": final_state.get("error")
                }
                results_summary.append(simple_output_error)
                try:
                    with open(simple_output_path, 'w') as f: json.dump(simple_output_error, f, indent=4)
                except Exception as e_save: print(f"  ERROR saving simple error output JSON for {protein_name}: {e_save}", file=sys.stderr)

            end_time = time.time()
            print(f"--- Sample {i+1} Finished (Took {end_time - start_time:.2f} seconds) ---")
            if (i + 1) % 50 == 0: print(f"*** Progress: Processed {i+1}/{len(eval_inputs)} samples ***")

        print(f"\nAgent evaluation run complete for {selected_llm} (Prompt: {args.prompt_style}, Models: {models_to_run_arg}).")
        print(f"Individual outputs saved in: {output_dir_for_run}")

    # +++ START OF MODIFIED NAMING FOR AGGREGATED JSON +++
    agg_filename_base = f"{selected_llm_safe}_prompt_{args.prompt_style}"
    if models_to_run_arg and models_run_str_safe != '': # Only add _models_ if there are models
        agg_filename_base += f"_models_{models_run_str_safe}"

    # Determine the correct ablation suffix for the aggregated file name
    if args.no_direct_analysis and args.no_indirect_analysis:
        agg_ablation_suffix = "_no_motif"
    elif args.no_direct_analysis:
        agg_ablation_suffix = "_no_direct"
    elif args.no_indirect_analysis:
        agg_ablation_suffix = "_no_indirect"
    else:
        agg_ablation_suffix = "" # Baseline, no suffix

    agg_filename = f"{agg_filename_base}{agg_ablation_suffix}_aggregated_simple_results.json"
    # Ensure the main output directory (args.output_dir) exists for aggregated files
    try:
        os.makedirs(args.output_dir, exist_ok=True) 
    except OSError as e:
        print(f"ERROR: Could not create main output directory '{args.output_dir}' for aggregated results: {e}", file=sys.stderr)
        # Decide if to exit or try to save in current dir or subdir
    
    agg_simple_path = os.path.join(args.output_dir, agg_filename)
    # +++ END OF MODIFIED NAMING FOR AGGREGATED JSON +++

    try:
        with open(agg_simple_path, 'w') as f: json.dump(results_summary, f, indent=2) # results_summary is collected during the loop
        print(f"Aggregated simple results for this run saved to: {agg_simple_path}")
    except Exception as e_save: print(f"ERROR saving aggregated simple results for this run: {e_save}", file=sys.stderr)

    if dna_db_con:
        try: dna_db_con.close(); print("DNA DB Connection closed.")
        except Exception as e: print(f"Error closing DNA DB: {e}")

    if hf_model: del hf_model
    if hf_tokenizer: del hf_tokenizer
    if transformer_model: del transformer_model
    if deepsea_model_instance: del deepsea_model_instance
    if sei_model_instance: del sei_model_instance

    print("\n--- Overall Unknown Protein Handling Stats ---")
    for model_key, stats in _unknown_handling_stats_cache.items():
        print(f"\n--- Stats for Model: {model_key} ---")
        if stats.get('total_unknown', 0) > 0:
            pct = lambda k_stat, s_stat=stats: f"{s_stat.get(k_stat,0)} ({s_stat.get(k_stat,0)/s_stat['total_unknown']:.1%})" if s_stat['total_unknown'] > 0 else f"{s_stat.get(k_stat,0)} (N/A)"
            print(f"  Total unknown processed for this model: {stats['total_unknown']}")
            print(f"  Ensemble Logits Match: {pct('ensemble_logits_match')}")
        else:
            print(f"  No unknown proteins processed for model {model_key} via this mechanism.")

    if DEVICE.type == 'cuda': print("Clearing CUDA cache..."); torch.cuda.empty_cache(); print("CUDA cache cleared.")
    print("Forcing garbage collection..."); gc.collect(); print("Garbage collection finished.")