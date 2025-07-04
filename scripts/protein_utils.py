# protein_utils.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import random
import json
import re
import requests
import time
import subprocess
import tempfile
import shutil

# --- Configuration for External Tools & Availability Checks ---
MMSEQS_EXECUTABLE = "mmseqs"
BLASTP_EXECUTABLE = "blastp"
MAKEBLASTDB_EXECUTABLE = "makeblastdb"

def is_tool_available(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

LEVENSHTEIN_AVAILABLE = False
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
    print("INFO (protein_utils): Levenshtein library found.")
except ImportError:
    print("WARNING (protein_utils): Levenshtein library not found. Levenshtein similarity will be unavailable.")

MMSEQS_AVAILABLE = is_tool_available(MMSEQS_EXECUTABLE)
BLAST_AVAILABLE = is_tool_available(BLASTP_EXECUTABLE) and is_tool_available(MAKEBLASTDB_EXECUTABLE)

if MMSEQS_AVAILABLE: print(f"INFO (protein_utils): MMseqs2 found at '{shutil.which(MMSEQS_EXECUTABLE)}'.")
else: print("WARNING (protein_utils): MMseqs2 not found. MMseqs2 similarity will be unavailable.")

if BLAST_AVAILABLE: print(f"INFO (protein_utils): BLAST+ tools found (blastp: '{shutil.which(BLASTP_EXECUTABLE)}').")
else: print("WARNING (protein_utils): BLAST+ tools not found. BLAST similarity will be unavailable.")


def dna_to_one_hot(dna_sequence, seq_len=None):
    # ... (Exact same implementation as in eval_deepsea.py) ...
    dna_sequence_processed = str(dna_sequence)
    if seq_len is None: seq_len = len(dna_sequence_processed)
    if len(dna_sequence_processed) < seq_len: dna_sequence_processed += 'N' * (seq_len - len(dna_sequence_processed))
    elif len(dna_sequence_processed) > seq_len: dna_sequence_processed = dna_sequence_processed[:seq_len]
    mapping = {'A': [1,0,0,0], 'G': [0,1,0,0], 'C': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    one_hot_encoding = np.array([mapping.get(nuc.upper(), [0,0,0,0]) for nuc in dna_sequence_processed], dtype=np.float32)
    return one_hot_encoding.T

# --- FASTA and UniProt Utilities ---
def parse_fasta_for_gene_names(fasta_file_path: Path) -> dict:
    print(f"Parsing FASTA: {fasta_file_path}...")
    sequences = {}
    current_gene_id = None
    current_seq_parts = []
    protein_count = 0
    try:
        with open(fasta_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    protein_count += 1
                    if current_gene_id and current_seq_parts:
                        sequences[current_gene_id] = "".join(current_seq_parts)
                    
                    header = line[1:]
                    extracted_id = None
                    
                    # Try to get GN= field first
                    gn_match = re.search(r"GN=([A-Z0-9\-_]+)", header, re.I)
                    if gn_match:
                        extracted_id = gn_match.group(1).upper()
                    
                    # If no GN=, try UniProt ID format like |P12345| or |P12345_HUMAN|
                    if not extracted_id:
                        id_match = re.search(r"\|([A-Z0-9\-_]+?)(?:_[A-Z0-9]+)?\|", header) or \
                                   re.match(r"([A-Z0-9\-_]+?)(?:_[A-Z0-9]+)?\s", header) # Also check if ID is at start of header
                        if id_match:
                            full_match_part = id_match.group(1)
                            # Take part before first underscore if it looks like P12345_HUMAN
                            extracted_id = (full_match_part.split('_')[0] if "_" in full_match_part else full_match_part).upper()

                    # If still no ID, try more general UniProt accession regex
                    if not extracted_id:
                        uniprot_match = re.search(r"\|([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\|", header)
                        if uniprot_match:
                            extracted_id = uniprot_match.group(1).upper()
                            
                    # Ultimate fallback: take the first word after '>' or '|'
                    if not extracted_id:
                        extracted_id = header.split(' ')[0].upper().lstrip('>|')
                        
                    current_gene_id = extracted_id
                    current_seq_parts = []
                elif current_gene_id: # Only append if we have a current_gene_id
                    current_seq_parts.append(line)
            
            # Add the last sequence
            if current_gene_id and current_seq_parts:
                sequences[current_gene_id] = "".join(current_seq_parts)
        
        # Basic filter for valid keys (non-empty and reasonable length, e.g., >1)
        sequences = {k: v for k, v in sequences.items() if k and len(k) > 1}
        print(f"Parsed {len(sequences)} sequences from FASTA (processed {protein_count} headers).")
        return sequences
    except FileNotFoundError:
        print(f"ERROR: FASTA file not found at {fasta_file_path}")
        return {}
    except Exception as e:
        print(f"ERROR parsing FASTA {fasta_file_path}: {e}")
        return {}

def fetch_uniprot_sequence_by_gene_or_id(identifier: str, organism_id: str = "9606", max_retries: int = 2, retry_delay: int = 2) -> str | None:
    time.sleep(0.1) # Basic rate limiting before any request
    query = identifier
    is_uniprot_id = (re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", identifier, re.I) or 
                     re.match(r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$", identifier, re.I) or
                     re.match(r"^[A-Z0-9]{6,10}$", identifier)) # For longer IDs like A0A...

    if not is_uniprot_id and organism_id: # If not a clear ID, assume gene name and use organism filter
        query = f"gene:{identifier} AND organism_id:{organism_id}"
    # else, if it looks like a UniProt ID or no organism_id, query with identifier directly

    url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=fasta&size=1"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15) # Increased timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            fasta_content = response.text.strip()
            if not fasta_content or not fasta_content.startswith('>'):
                # print(f"DEBUG (fetch_uniprot): No valid FASTA content for {identifier} (Query: {query}). Response: {fasta_content[:200]}")
                return None
            return "".join(fasta_content.splitlines()[1:]) # Return sequence part
        except requests.exceptions.RequestException as e:
            print(f"Warning (fetch_uniprot): Request failed for {identifier} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None # Failed after all retries
    return None


class ProteinDNADatasetInference(torch.utils.data.Dataset): # Inherit from Dataset
    def __init__(self, data_df, protein_id_to_model_id_map, dna_seq_len,
                 global_protein_sequences_map,
                 protein_id_to_embedding_map=None,
                 known_protein_ids_for_model=None,
                 protein_id_to_tf_family_map=None,
                 tf_family_to_known_protein_ids_map=None,
                 fallback_embedding_for_unknown=None,
                 verbose_unknown_matching=False, # Changed default to False
                 num_top_similar_for_avg_logits=1,
                 similarity_engine='levenshtein',
                 max_levenshtein_dist=float('inf'),
                 min_mmseqs_blast_bitscore=0,
                 max_mmseqs_blast_evalue=10.0,
                 mmseqs_db_path_prefix=None,
                 blast_db_path_prefix=None,
                 temp_dir_base="./tmp_protein_sim"
                ):
        self.df = data_df
        self.protein_id_to_model_id_map = protein_id_to_model_id_map
        self.dna_seq_len = dna_seq_len
        self.global_protein_sequences_map = global_protein_sequences_map
        self.protein_id_to_embedding_map_for_fallback = protein_id_to_embedding_map if protein_id_to_embedding_map is not None else {}
        self.known_protein_ids_for_model_set = set(known_protein_ids_for_model) if known_protein_ids_for_model else set()
        self.protein_id_to_tf_family_map = protein_id_to_tf_family_map if protein_id_to_tf_family_map is not None else {}
        self.tf_family_to_known_protein_ids_map = tf_family_to_known_protein_ids_map if tf_family_to_known_protein_ids_map is not None else {}
        self.fallback_embedding = fallback_embedding_for_unknown
        self.verbose_unknown_matching = verbose_unknown_matching
        self.num_top_similar_for_avg_logits = num_top_similar_for_avg_logits
        self.similarity_engine = similarity_engine
        self.max_levenshtein_dist = max_levenshtein_dist
        self.min_mmseqs_blast_bitscore = min_mmseqs_blast_bitscore
        self.max_mmseqs_blast_evalue = max_mmseqs_blast_evalue
        self.temp_dir_base = Path(temp_dir_base); self.temp_dir_base.mkdir(parents=True, exist_ok=True)
        self.mmseqs_db_path_prefix = Path(mmseqs_db_path_prefix) if mmseqs_db_path_prefix else self.temp_dir_base/"known_mmseqs_db"
        self.blast_db_path_prefix = Path(blast_db_path_prefix) if blast_db_path_prefix else self.temp_dir_base/"known_blast_db"

        self.unknown_handling_stats = {
            "ensemble_logits_match": 0, "tf_family_avg_emb_match": 0, "fallback_emb_used": 0, "id0_used": 0,
            "total_unknown": 0, "ensemble_match_count_dist": {}, "threshold_filtered_out": 0,
            "seq_found_in_global_map": 0, "seq_missing_for_unknown": 0,
            "seq_missing_for_known_in_search":0, "similarity_engine_error":0,
            "unknown_not_in_family_map":0, "family_found_no_known_members_with_emb":0,
            "no_valid_similar_for_ensemble": 0
        }
        if self.df is not None and not self.df.empty and self.similarity_engine in ['mmseqs', 'blast'] and self.known_protein_ids_for_model_set:
            self._prepare_similarity_database()
        elif self.df is None or self.df.empty: # If used without a df (e.g. just for _handle_unknown_protein)
            if self.similarity_engine in ['mmseqs', 'blast'] and self.known_protein_ids_for_model_set:
                 self._prepare_similarity_database() # Still prepare DB if needed by engine

    def __len__(self):
        return len(self.df) if self.df is not None else 0


    def _prepare_similarity_database(self):
        # ... (Identical to the one in eval_deepsea.py, ensure it uses self.attributes)
        known_fasta_path = self.temp_dir_base / "known_proteins_for_db.fasta"; db_exists = False
        db_file_to_check = None
        if self.similarity_engine == 'mmseqs':
            db_file_to_check = self.mmseqs_db_path_prefix.with_suffix(".dbtype")
            db_exists = db_file_to_check.exists() if MMSEQS_AVAILABLE else False
        elif self.similarity_engine == 'blast':
            db_file_to_check = self.blast_db_path_prefix.with_suffix(".pin")
            db_exists = db_file_to_check.exists() if BLAST_AVAILABLE else False

        if db_exists:
            if not known_fasta_path.exists():
                if self.verbose_unknown_matching: print(f"INFO (protein_utils): {self.similarity_engine} DB found, not rebuilding (source FASTA for comparison absent).")
                return
            elif db_file_to_check and db_file_to_check.exists() and known_fasta_path.stat().st_mtime < db_file_to_check.stat().st_mtime:
                if self.verbose_unknown_matching: print(f"INFO (protein_utils): {self.similarity_engine} DB seems up-to-date relative to {known_fasta_path}.")
                return
            else:
                if self.verbose_unknown_matching: print(f"WARN (protein_utils): {self.similarity_engine} DB might be stale or FASTA newer, rebuilding.")
        elif self.verbose_unknown_matching:
             print(f"INFO (protein_utils): {self.similarity_engine} DB not found or tool unavailable, attempting to build.")

        if self.verbose_unknown_matching: print(f"INFO (protein_utils): Preparing {self.similarity_engine} database for known model proteins...")
        with open(known_fasta_path, "w") as f_known:
            count_written = 0
            for prot_id in self.known_protein_ids_for_model_set:
                seq = self.global_protein_sequences_map.get(prot_id.upper())
                if seq: f_known.write(f">{prot_id}\n{seq}\n"); count_written+=1
                else: self.unknown_handling_stats["seq_missing_for_known_in_search"] += 1
        if count_written == 0:
            if self.verbose_unknown_matching: print("WARNING (protein_utils): No known protein sequences found to build similarity database.")
            if known_fasta_path.exists(): known_fasta_path.unlink(missing_ok=True)
            return
        try:
            if self.similarity_engine == 'mmseqs' and MMSEQS_AVAILABLE:
                cmd = [MMSEQS_EXECUTABLE, "createdb", str(known_fasta_path), str(self.mmseqs_db_path_prefix), "--dbtype", "1"]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                if self.verbose_unknown_matching: print(f"INFO (protein_utils): MMseqs2 DB created: {self.mmseqs_db_path_prefix}")
            elif self.similarity_engine == 'blast' and BLAST_AVAILABLE:
                cmd = [MAKEBLASTDB_EXECUTABLE, "-in", str(known_fasta_path), "-dbtype", "prot", "-out", str(self.blast_db_path_prefix)]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                if self.verbose_unknown_matching: print(f"INFO (protein_utils): BLAST DB created: {self.blast_db_path_prefix}")
            elif self.verbose_unknown_matching:
                 print(f"WARN (protein_utils): Similarity engine {self.similarity_engine} selected but tool is not available. DB not built.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR (protein_utils): creating DB: {e.stderr if e.stderr else e.stdout}");
            self.unknown_handling_stats["similarity_engine_error"] +=1
        finally:
            if known_fasta_path.exists(): known_fasta_path.unlink(missing_ok=True)

    def _run_similarity_search(self, query_protein_id, query_sequence):
        # ... (Identical to the one in eval_deepsea.py, ensure it uses self.attributes)
        with tempfile.TemporaryDirectory(dir=self.temp_dir_base, prefix=f"search_{query_protein_id}_") as tmp_search_dir_str:
            tmp_search_dir=Path(tmp_search_dir_str); query_fasta_path=tmp_search_dir/"query.fasta"
            with open(query_fasta_path,"w") as fq: fq.write(f">{query_protein_id}\n{query_sequence}\n")
            hits=[]; lower_score_is_better_flag = True
            try:
                if self.similarity_engine == 'levenshtein' and LEVENSHTEIN_AVAILABLE:
                    lower_score_is_better_flag = True
                    for known_pid in self.known_protein_ids_for_model_set:
                        known_seq = self.global_protein_sequences_map.get(known_pid.upper())
                        if known_seq:
                            dist = Levenshtein.distance(query_sequence, known_seq)
                            if dist <= self.max_levenshtein_dist: hits.append((float(dist), known_pid))
                        # else: self.unknown_handling_stats["seq_missing_for_known_in_search"] +=1 # Already counted in _prepare_db
                elif self.similarity_engine == 'mmseqs' and MMSEQS_AVAILABLE:
                    lower_score_is_better_flag = False
                    results_m8 = tmp_search_dir/"results.m8"
                    cmd = [MMSEQS_EXECUTABLE,"easy-search",str(query_fasta_path),str(self.mmseqs_db_path_prefix),str(results_m8),str(tmp_search_dir/"tmp_mm"),"--threads","1","-s","1.0", "--min-seq-id", "0.0", "--format-output","query,target,pident,evalue,bits", "--max-seqs", str(self.num_top_similar_for_avg_logits * 10) ]
                    subprocess.run(cmd,check=True,capture_output=True, text=True)
                    if results_m8.exists():
                        with open(results_m8,'r') as fr:
                            for line in fr:
                                p=line.strip().split('\t')
                                if len(p)>=5 and float(p[3])<=self.max_mmseqs_blast_evalue and float(p[4])>=self.min_mmseqs_blast_bitscore:
                                    hits.append((float(p[4]),p[1]))
                elif self.similarity_engine == 'blast' and BLAST_AVAILABLE:
                    lower_score_is_better_flag = False
                    results_blast = tmp_search_dir/"results_blast.txt"
                    cmd = [BLASTP_EXECUTABLE,"-query",str(query_fasta_path),"-db",str(self.blast_db_path_prefix),"-out",str(results_blast),"-outfmt","6 sseqid evalue bitscore","-max_target_seqs",str(self.num_top_similar_for_avg_logits*10)]
                    subprocess.run(cmd,check=True,capture_output=True, text=True)
                    if results_blast.exists():
                        with open(results_blast,'r') as fr:
                            for line in fr:
                                p=line.strip().split('\t')
                                if len(p)==3 and float(p[1])<=self.max_mmseqs_blast_evalue and float(p[2])>=self.min_mmseqs_blast_bitscore:
                                    hits.append((float(p[2]),p[0]))
                else:
                    if self.verbose_unknown_matching: print(f"WARN (protein_utils): Sim engine '{self.similarity_engine}' not available/configured for {query_protein_id}")
                    self.unknown_handling_stats["similarity_engine_error"]+=1; return [], True
                hits.sort(key=lambda x: x[0], reverse=(not lower_score_is_better_flag))
                return hits, lower_score_is_better_flag
            except subprocess.CalledProcessError as e_sub:
                print(f"ERROR (protein_utils): Subprocess failed for {self.similarity_engine} on {query_protein_id}: {e_sub.stderr if e_sub.stderr else e_sub.stdout}")
                self.unknown_handling_stats["similarity_engine_error"]+=1; return [], True
            except Exception as e:
                print(f"ERROR (protein_utils): in _run_similarity_search for {query_protein_id}: {e}")
                self.unknown_handling_stats["similarity_engine_error"]+=1; return [], True

    def _handle_unknown_protein(self, unknown_protein_id_from_df):
        self.unknown_handling_stats["total_unknown"] += 1
        unknown_protein_sequence = self.global_protein_sequences_map.get(unknown_protein_id_from_df)
        if unknown_protein_sequence: self.unknown_handling_stats["seq_found_in_global_map"] += 1
        else: self.unknown_handling_stats["seq_missing_for_unknown"] += 1

        # Tier 0: Ensemble Logits from Similar Known Proteins
        if unknown_protein_sequence:
            all_hits_with_scores, _ = self._run_similarity_search(unknown_protein_id_from_df, unknown_protein_sequence)
            valid_similar_protein_model_ids = []
            similar_protein_gene_names = [] # NEW: Store gene names of similar proteins
            formatted_scores_logs = []

            for score, hit_protein_id_uc in all_hits_with_scores: # hit_protein_id_uc is gene name from DB
                if len(valid_similar_protein_model_ids) >= self.num_top_similar_for_avg_logits: break
                model_internal_id = self.protein_id_to_model_id_map.get(hit_protein_id_uc)
                if model_internal_id is not None:
                    valid_similar_protein_model_ids.append(model_internal_id)
                    similar_protein_gene_names.append(hit_protein_id_uc) # Store the gene name
                    score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                    formatted_scores_logs.append(f"{hit_protein_id_uc}(s:{score_str})")

            if valid_similar_protein_model_ids:
                self.unknown_handling_stats["ensemble_logits_match"] += 1
                num_averaged = len(valid_similar_protein_model_ids)
                self.unknown_handling_stats["ensemble_match_count_dist"][num_averaged] = \
                    self.unknown_handling_stats["ensemble_match_count_dist"].get(num_averaged, 0) + 1
                if self.verbose_unknown_matching:
                    print(f"INFO (protein_utils): Unknown '{unknown_protein_id_from_df}' (engine:{self.similarity_engine}) -> "
                          f"Ensemble Logits from N={num_averaged} similar: [{', '.join(formatted_scores_logs)}].")
                # MODIFIED: Return tuple of (IDs_tensor, gene_names_list)
                return (torch.LongTensor(valid_similar_protein_model_ids), similar_protein_gene_names), "ensemble_similar_ids"
            else:
                self.unknown_handling_stats["no_valid_similar_for_ensemble"] += 1
                if self.verbose_unknown_matching and unknown_protein_sequence:
                     print(f"INFO (protein_utils): For '{unknown_protein_id_from_df}', sim search ran, but no valid similar proteins (known to model) found/kept.")
        elif self.verbose_unknown_matching:
            print(f"INFO (protein_utils): Unknown '{unknown_protein_id_from_df}' missing sequence. Cannot use ensemble logits strategy.")

        # ... (Tier 1, 2, 3 fallbacks remain the same, they return (tensor, type_string) ) ...
        unknown_tf_family = self.protein_id_to_tf_family_map.get(unknown_protein_id_from_df)
        if unknown_tf_family:
            known_proteins_in_family = self.tf_family_to_known_protein_ids_map.get(unknown_tf_family, [])
            family_member_embeddings = []
            for pid in known_proteins_in_family:
                emb = self.protein_id_to_embedding_map_for_fallback.get(pid)
                if emb is not None: family_member_embeddings.append(emb)
            if family_member_embeddings:
                family_avg_embedding = torch.stack(family_member_embeddings).mean(dim=0)
                self.unknown_handling_stats["tf_family_avg_emb_match"] += 1
                if self.verbose_unknown_matching: print(f"INFO (protein_utils): Unknown '{unknown_protein_id_from_df}' (Family: {unknown_tf_family}) -> TF Family Avg Emb from {len(family_member_embeddings)} members.")
                return family_avg_embedding.clone().detach(), "embedding"
            else: self.unknown_handling_stats["family_found_no_known_members_with_emb"] += 1
        else: self.unknown_handling_stats["unknown_not_in_family_map"] += 1

        if self.fallback_embedding is not None:
            if self.verbose_unknown_matching: print(f"INFO (protein_utils): For '{unknown_protein_id_from_df}', no ensemble/family. Using global fallback embedding.")
            self.unknown_handling_stats["fallback_emb_used"] += 1
            return self.fallback_embedding.clone().detach(), "embedding"

        default_id_for_model = 0
        if self.verbose_unknown_matching: print(f"WARNING (protein_utils): For '{unknown_protein_id_from_df}', all fallbacks exhausted. Using model ID {default_id_for_model}.")
        self.unknown_handling_stats["id0_used"] += 1
        return torch.LongTensor([default_id_for_model]), "id"


    def __getitem__(self, idx):
        if self.df is None or self.df.empty:
            raise IndexError("Dataset DataFrame is not initialized or empty.")
        row = self.df.iloc[idx]; dna_seq_str = str(row['dna'])
        protein_id_from_df = str(row['protein']).upper(); label = row['label']
        dna_tensor = torch.FloatTensor(dna_to_one_hot(dna_seq_str, self.dna_seq_len)) # dna_to_one_hot is global in protein_utils
        label_tensor = torch.FloatTensor([label])
        model_internal_protein_id = self.protein_id_to_model_id_map.get(protein_id_from_df)

        if model_internal_protein_id is not None:
            protein_payload_for_item = torch.LongTensor([model_internal_protein_id]);
            protein_input_type_for_item = "id"
        else:
            # _handle_unknown_protein returns (payload, type_string)
            # payload is a tensor for "id"/"embedding", or a tuple (ids_tensor, gene_names_list) for "ensemble_similar_ids"
            protein_payload_for_item, protein_input_type_for_item = self._handle_unknown_protein(protein_id_from_df)
        
        # The item returned by __getitem__ is:
        # (dna_tensor, protein_payload, label_tensor, protein_input_type_string, original_protein_id_string)
        return dna_tensor, protein_payload_for_item, label_tensor, protein_input_type_for_item, protein_id_from_df
    