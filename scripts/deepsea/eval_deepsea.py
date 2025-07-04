import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, roc_curve,
    confusion_matrix, precision_score, recall_score, matthews_corrcoef
)
from tqdm import tqdm
from pathlib import Path
import random # For sampling
import json

def dna_to_one_hot(dna_sequence, seq_len=None):
    if seq_len is None:
        seq_len = len(dna_sequence)
    if len(dna_sequence) < seq_len:
        dna_sequence += 'N' * (seq_len - len(dna_sequence))
    elif len(dna_sequence) > seq_len:
        dna_sequence = dna_sequence[:seq_len]
    mapping = {'A': [1,0,0,0], 'G': [0,1,0,0], 'C': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    one_hot_encoding = np.array([mapping.get(nuc.upper(), [0,0,0,0]) for nuc in dna_sequence], dtype=np.float32)
    return one_hot_encoding.T

def evaluate_binary(y_pred_probs, y_label):
    try:
        roc_auc = roc_auc_score(y_label, y_pred_probs)
        pr_auc = average_precision_score(y_label, y_pred_probs)
        fpr, tpr, thresholds = roc_curve(y_label, y_pred_probs)
        J = tpr - fpr
        if len(J) == 0 or len(thresholds) == 0: thred_optim = 0.5
        else:
            if len(thresholds) > 1 : thred_optim = thresholds[np.argmax(J)]
            else: thred_optim = 0.5
        thred_optim = 0.5
        print(f"Using threshold: {thred_optim}")
        y_pred_binary = (y_pred_probs >= thred_optim).astype(int)
        cm = confusion_matrix(y_label, y_pred_binary)
        if cm.size == 1:
             if y_label[0] == 0 and y_pred_binary[0] == 0 : tn, fp, fn, tp = len(y_label),0,0,0
             elif y_label[0] == 1 and y_pred_binary[0] == 1: tn, fp, fn, tp = 0,0,0,len(y_label)
             elif y_label[0] == 0 and y_pred_binary[0] == 1: tn, fp, fn, tp = 0,len(y_label),0,0
             elif y_label[0] == 1 and y_pred_binary[0] == 0: tn, fp, fn, tp = 0,0,len(y_label),0
             else: tn, fp, fn, tp = 0,0,0,0
        elif cm.size == 4: tn, fp, fn, tp = cm.ravel()
        else: tn, fp, fn, tp = 0,0,0,0
        total_samples = tn + fp + fn + tp
        accuracy = (tn + tp) / total_samples if total_samples > 0 else 0
        precision = precision_score(y_label, y_pred_binary, zero_division=0)
        recall = recall_score(y_label, y_pred_binary, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_label, y_pred_binary, zero_division=0)
        mcc = matthews_corrcoef(y_label, y_pred_binary)
        return accuracy, precision, recall, specificity, roc_auc, pr_auc, mcc, f1
    except ValueError as e: return 0,0,0,0,0,0,0,0


# --- Dataset Class for Inference (Handles unknown proteins) ---
def custom_collate_fn_batched(batch):
    # batch is a list of (dna_tensor, protein_tensor, label_tensor, protein_input_type_str)

    # For samples with protein IDs
    dna_id_list = []
    protein_id_list = []
    label_id_list = []
    original_indices_id = [] # To map results back to original batch order

    # For samples with protein embeddings
    dna_emb_list = []
    protein_emb_list = []
    label_emb_list = []
    original_indices_emb = [] # To map results back

    for i, sample in enumerate(batch):
        dna_tensor, protein_tensor, label_tensor, protein_type = sample
        if protein_type == "id":
            dna_id_list.append(dna_tensor)
            protein_id_list.append(protein_tensor) # Should be LongTensor([id])
            label_id_list.append(label_tensor)
            original_indices_id.append(i)
        elif protein_type == "embedding":
            dna_emb_list.append(dna_tensor)
            protein_emb_list.append(protein_tensor) # Should be FloatTensor([emb_dim])
            label_emb_list.append(label_tensor)
            original_indices_emb.append(i)
        else:
            raise ValueError(f"Unknown protein type in batch: {protein_type}")

    # Prepare batches for ID-based inputs
    dna_id_batch, protein_id_batch, label_id_batch = None, None, None
    if dna_id_list:
        dna_id_batch = torch.stack(dna_id_list, 0)
        # protein_id_list contains tensors of shape [1], stack and squeeze
        protein_id_batch = torch.stack(protein_id_list, 0).squeeze(1) # Becomes (N_ids,)
        label_id_batch = torch.stack(label_id_list, 0)

    # Prepare batches for embedding-based inputs
    dna_emb_batch, protein_emb_batch, label_emb_batch = None, None, None
    if dna_emb_list:
        dna_emb_batch = torch.stack(dna_emb_list, 0)
        protein_emb_batch = torch.stack(protein_emb_list, 0) # Should be (N_emb, emb_dim)
        label_emb_batch = torch.stack(label_emb_list, 0)
        
    # We also need all original labels in order for final concatenation
    all_original_labels_sorted = [None] * len(batch)
    for i, sample in enumerate(batch):
        all_original_labels_sorted[i] = sample[2] # label_tensor
    all_original_labels_batch = torch.stack(all_original_labels_sorted, 0)


    return {
        "dna_id": dna_id_batch, "protein_id": protein_id_batch, "label_id": label_id_batch, "indices_id": original_indices_id,
        "dna_emb": dna_emb_batch, "protein_emb": protein_emb_batch, "label_emb": label_emb_batch, "indices_emb": original_indices_emb,
        "all_labels_in_order": all_original_labels_batch,
        "batch_size": len(batch)
    }

class ProteinDNADatasetInference(Data.Dataset):
    def __init__(self, data_df, protein_to_id_map, dna_seq_len, unknown_protein_embedding=None):
        self.df = data_df # Expects a pandas DataFrame
        self.protein_to_id_map = protein_to_id_map
        self.dna_seq_len = dna_seq_len
        self.unknown_protein_embedding = unknown_protein_embedding # Pre-calculated average embedding

        # Create a reverse map for debugging or inspection if needed
        self.id_to_protein_map = {v: k for k, v in protein_to_id_map.items()}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dna_seq_str = row['dna']
        protein_name = row['protein']
        label = row['label'] # Will be present in test data, used for evaluation

        dna_one_hot = dna_to_one_hot(dna_seq_str, self.dna_seq_len)
        dna_tensor = torch.FloatTensor(dna_one_hot)

        protein_id = self.protein_to_id_map.get(protein_name)
        protein_input_type = "id" # "id" or "embedding"

        if protein_id is not None:
            protein_tensor = torch.LongTensor([protein_id])
        elif self.unknown_protein_embedding is not None:
            # Use the pre-calculated average embedding for unknown proteins
            protein_tensor = self.unknown_protein_embedding.clone().detach() # ensure it's a new tensor
            protein_input_type = "embedding"
            # print(f"Using unknown embedding for protein: {protein_name}") # For debugging
        else:
            # Fallback if unknown_protein_embedding is not provided (should not happen with new logic)
            # Or raise an error
            print(f"ERROR: Protein '{protein_name}' not in map and no unknown_protein_embedding provided. Using ID 0.")
            protein_tensor = torch.LongTensor([0]) # Default to a known ID, e.g., the first one

        label_tensor = torch.FloatTensor([label])

        # Return protein_tensor and an indicator of its type
        return dna_tensor, protein_tensor, label_tensor, protein_input_type


# --- Model Definition (Modified to accept pre-calculated embeddings) ---
class DeepSEAProteinInteraction(nn.Module):
    def __init__(self, num_proteins, protein_emb_dim, l_in=512, cnn_l_out_factor=None):
        super(DeepSEAProteinInteraction, self).__init__()
        self.l_in = l_in
        self.protein_emb_dim = protein_emb_dim # Store for use with direct embeddings

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)

        if cnn_l_out_factor is not None: self.cnn_l_out = cnn_l_out_factor
        else:
            _l = self.l_in; _l = _l-7; _l = _l//4; _l = _l-7; _l = _l//4; _l = _l-7
            self.cnn_l_out = _l
            print(f"Calculated CNN output sequence length (l_out): {self.cnn_l_out}")

        self.protein_embedding = nn.Embedding(num_proteins, protein_emb_dim) # Renamed for clarity
        fc1_in_features = self.cnn_l_out * 960 + protein_emb_dim
        self.Linear1 = nn.Linear(fc1_in_features, 925)
        self.Linear2 = nn.Linear(925, 1)

    def forward(self, dna_input, protein_input, protein_input_type="id"):
        x_dna = self.Conv1(dna_input); x_dna = F.relu(x_dna); x_dna = self.Maxpool(x_dna); x_dna = self.Drop1(x_dna)
        x_dna = self.Conv2(x_dna); x_dna = F.relu(x_dna); x_dna = self.Maxpool(x_dna); x_dna = self.Drop1(x_dna)
        x_dna = self.Conv3(x_dna); x_dna = F.relu(x_dna); x_dna = self.Drop2(x_dna)
        x_dna = x_dna.view(-1, self.cnn_l_out * 960)

        if protein_input_type == "id":
            # protein_input is LongTensor of IDs
            x_protein = self.protein_embedding(protein_input)
        elif protein_input_type == "embedding":
            # protein_input is already a FloatTensor (batch_size, protein_emb_dim)
            x_protein = protein_input
            if x_protein.ndim == 1: # If a single unknown embedding was passed for a batch_size=1 sample
                x_protein = x_protein.unsqueeze(0)
        else:
            raise ValueError(f"Unknown protein_input_type: {protein_input_type}")


        x_combined = torch.cat((x_dna, x_protein), dim=1)
        x = self.Linear1(x_combined); x = F.relu(x); x = self.Linear2(x)
        return x

# --- Main Script Logic (Focus on Inference Part) ---
if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1) # For reproducible sampling
    if torch.cuda.is_available(): torch.cuda.manual_seed(1)

    L_IN_DNA = 512
    PROTEIN_EMB_DIM = 50
    BATCH_SIZE = 64 # For inference, can be larger if memory allows
    NUM_WORKERS = 0 # For inference, 0 is often fine or a small number
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    output_dir = Path("./benchmark/deepsea")
    best_model_path = output_dir / 'best_model_params.pkl'
    protein_map_load_path = output_dir / "protein_to_id_map.json" # Or .pkl
    
    # test_data_full_path = Path('/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/valid_min10.pkl')
    # test_data_full_path = Path('/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/test_min10.pkl')
    # test_data_full_path = Path('/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/data/test.pkl')
    test_data_full_path = Path('/new-stg/home/cong/DPI/dataset/ChIP_690/deepsea/data/valid.pkl')
    
    # dataset_name = 'enc34_valid_min10'
    # dataset_name = 'enc34_test_min10'
    # dataset_name = 'chip690_test'
    dataset_name = 'chip690_valid'
    
    num_test_samples_subset = 500
    random_state_for_sampling = 13

    # Path to the original PKL file from which protein_to_id_map was built
    # This is needed to reconstruct the map. If you saved the map, load it directly.
    # Assuming train_min10.pkl was used to build the map for training
    data_folder = '/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/' #MODIFY AS NEEDED
    
    # --- Reconstruct or Load protein_to_id_map ---
    # It's crucial that this map is IDENTICAL to the one used during training
    if protein_map_load_path.exists():
        print(f"Loading protein_to_id_map from {protein_map_load_path}...")
        try:
            with open(protein_map_load_path, 'r') as f:
                protein_to_id_map = json.load(f)
            NUM_PROTEINS = len(protein_to_id_map)
            print(f"Loaded protein_to_id_map with {NUM_PROTEINS} proteins.")
        except Exception as e:
            print(f"Error loading protein_to_id_map from {protein_map_load_path}: {e}. Will attempt to reconstruct.")
            protein_to_id_map = None # Flag to reconstruct
    else:    
        print("Reconstructing protein_to_id map (ensure this matches training)...")
        protein_map_source_files = [
            Path(data_folder) / 'train_min10.pkl',
            Path(data_folder) / 'valid_min10.pkl',
            Path(data_folder) / 'test_min10.pkl' # Include test to be comprehensive for map
        ]
        all_proteins_list = []
        for f_path in protein_map_source_files:
            if f_path.exists():
                try:
                    all_proteins_list.append(pd.read_pickle(f_path)['protein'])
                except Exception as e:
                    print(f"Warning: Could not read {f_path} for protein map: {e}")
            else:
                print(f"Warning: Protein map source file not found: {f_path}")

        if not all_proteins_list:
            print("ERROR: No protein data found to build protein_to_id_map. Exiting.")
            # As a fallback for script structure, create a dummy map
            protein_to_id_map = {"Protein_0": 0, "Protein_1": 1}
            NUM_PROTEINS = len(protein_to_id_map)
            print(f"USING DUMMY protein_to_id_map with {NUM_PROTEINS} proteins.")
            # exit(1)
        else:
            all_proteins_series = pd.concat(all_proteins_list)
            unique_proteins = sorted([p for p in all_proteins_series.unique() if pd.notna(p)])
            protein_to_id_map = {protein: i for i, protein in enumerate(unique_proteins)}
            NUM_PROTEINS = len(protein_to_id_map)
            print(f"Reconstructed protein_to_id_map with {NUM_PROTEINS} unique proteins.")

        if NUM_PROTEINS == 0:
            print("ERROR: NUM_PROTEINS is 0. Cannot proceed.")
            exit()

        if NUM_PROTEINS > 0:
            protein_map_save_path = output_dir / "protein_to_id_map.json" # Or .pkl
            try:
                import json
                with open(protein_map_save_path, 'w') as f:
                    json.dump(protein_to_id_map, f, indent=4)
                print(f"Saved protein_to_id_map to {protein_map_save_path}")
            except Exception as e:
                print(f"Error saving protein_to_id_map: {e}")
        else:
            print("Skipping saving protein_to_id_map as NUM_PROTEINS is 0.")

    # --- Load Model ---
    cnn_l_out_deepsea_calc = int((int((L_IN_DNA - 7) / 4) - 7) / 4) - 7
    model = DeepSEAProteinInteraction(
        num_proteins=NUM_PROTEINS, # Must match the number of proteins used for training
        protein_emb_dim=PROTEIN_EMB_DIM,
        l_in=L_IN_DNA,
        cnn_l_out_factor=cnn_l_out_deepsea_calc
    ).to(DEVICE)

    if not best_model_path.exists():
        print(f"ERROR: Best model not found at {best_model_path}. Exiting.")
        exit(1)
    
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # --- Calculate Average Embedding for Unknown Proteins ---
    # This uses the *trained* embeddings from the loaded model
    with torch.no_grad():
        all_known_protein_embeddings = model.protein_embedding.weight.data
        average_unknown_protein_embedding = torch.mean(all_known_protein_embeddings, dim=0)
        average_unknown_protein_embedding = average_unknown_protein_embedding.to(DEVICE) # Ensure it's on the right device
    print(f"Calculated average embedding for unknown proteins (shape: {average_unknown_protein_embedding.shape})")


    # --- Prepare Test Data Subset ---
    if not test_data_full_path.exists():
        print(f"ERROR: Full test data file not found: {test_data_full_path}")
        # Create dummy test data for unknown protein testing
        print("Creating dummy test data with known and unknown proteins...")
        dummy_test_data = []
        known_proteins_sample = list(protein_to_id_map.keys())[:2] if protein_to_id_map else ["Protein_0"]
        
        for i in range(num_test_samples_subset // 2) :
             dummy_test_data.append({
                'dna': "ATGC" * (L_IN_DNA//4 +1 )[:L_IN_DNA],
                'protein': random.choice(known_proteins_sample) if known_proteins_sample else "Protein_Known_Dummy",
                'label': random.randint(0,1)
            })
        for i in range(num_test_samples_subset // 2, num_test_samples_subset) :
            dummy_test_data.append({
                'dna': "CGTA" * (L_IN_DNA//4 +1 )[:L_IN_DNA],
                'protein': f"UnknownProtein_{i}", # These will not be in protein_to_id_map
                'label': random.randint(0,1)
            })
        test_df_full = pd.DataFrame(dummy_test_data)
        # exit(1)
    else:
        test_df_full = pd.read_pickle(test_data_full_path)

    if len(test_df_full) < num_test_samples_subset:
        print(f"Warning: Full test set ({len(test_df_full)}) is smaller than requested subset ({num_test_samples_subset}). Using full test set.")
        test_df_subset = test_df_full
    else:
        test_df_subset = test_df_full.sample(n=num_test_samples_subset, random_state=random_state_for_sampling) # For reproducibility
    
    print(f"Using a subset of {len(test_df_subset)} test samples.")

    # Add some artificial unknown proteins to the subset for testing this feature
    # (Optional: if your test_min10.pkl already contains proteins not in train/valid)
    if "UnknownProtein_Test1" not in protein_to_id_map and len(test_df_subset) > 2 :
        print("Adding artificial unknown proteins to test subset...")
        num_to_replace = min(5, len(test_df_subset) // 10) # Replace up to 5 samples or 10%
        indices_to_replace = random.sample(list(test_df_subset.index), num_to_replace)
        for i, idx in enumerate(indices_to_replace):
            test_df_subset.loc[idx, 'protein'] = f"Artificially_Unknown_Protein_{i+1}"
        print(f"Replaced {num_to_replace} protein names with artificial unknowns.")


    subset_test_dataset = ProteinDNADatasetInference(
        data_df=test_df_subset,
        protein_to_id_map=protein_to_id_map,
        dna_seq_len=L_IN_DNA,
        unknown_protein_embedding=average_unknown_protein_embedding
    )
    subset_test_loader = Data.DataLoader(
        subset_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn_batched
    )

    # --- Evaluation on Subset ---
    print(f"\nStarting evaluation on {len(test_df_subset)} test samples (subset)...")
    all_subset_preds_collected, all_subset_labels_collected = [], [] # Use different names
    model.eval()
    with torch.no_grad():
        progress_bar_test = tqdm(subset_test_loader, desc="Subset Test Eval", unit="batch")
        for batch_data in progress_bar_test:
            # batch_data is the dictionary returned by custom_collate_fn_batched
            
            batch_predictions = [None] * batch_data["batch_size"]
            
            # Process ID-based samples
            if batch_data["dna_id"] is not None:
                dna_id_b = batch_data["dna_id"].to(DEVICE)
                protein_id_b = batch_data["protein_id"].to(DEVICE) # Already (N_ids,)
                
                # Model expects protein_id_b to be (N_ids) for nn.Embedding
                output_logits_id = model(dna_id_b, protein_id_b, protein_input_type="id")
                
                pred_probs_id = torch.sigmoid(output_logits_id).cpu()
                for i, orig_idx in enumerate(batch_data["indices_id"]):
                    batch_predictions[orig_idx] = pred_probs_id[i]

            # Process embedding-based samples
            if batch_data["dna_emb"] is not None:
                dna_emb_b = batch_data["dna_emb"].to(DEVICE)
                protein_emb_b = batch_data["protein_emb"].to(DEVICE) # Already (N_emb, emb_dim)
                
                output_logits_emb = model(dna_emb_b, protein_emb_b, protein_input_type="embedding")

                pred_probs_emb = torch.sigmoid(output_logits_emb).cpu()
                for i, orig_idx in enumerate(batch_data["indices_emb"]):
                    batch_predictions[orig_idx] = pred_probs_emb[i]
            
            # Ensure all predictions were filled (sanity check)
            if any(p is None for p in batch_predictions):
                raise RuntimeError("Not all predictions were filled in the batch. Check collate_fn logic.")

            # Stack predictions for the current batch and collect
            batch_predictions_tensor = torch.stack(batch_predictions) # (batch_size, 1)
            all_subset_preds_collected.append(batch_predictions_tensor.numpy())
            
            # Collect labels (which are already in original batch order)
            all_subset_labels_collected.append(batch_data["all_labels_in_order"].cpu().numpy())

    # Concatenate all collected predictions and labels
    all_subset_preds = np.concatenate(all_subset_preds_collected).flatten()
    all_subset_labels = np.concatenate(all_subset_labels_collected).flatten()

    print(f"\n--- Subset Test Set Performance ({len(test_df_subset)} samples) ---")
    if len(all_subset_labels) > 0 and len(np.unique(all_subset_labels)) > 1 : # Check for valid evaluation scenario
        accuracy, precision, recall, specificity, roc_auc, pr_auc, mcc, f1 = evaluate_binary(all_subset_preds, all_subset_labels)
        print(f"Accuracy:    {accuracy:.5f}")
        print(f"Precision:   {precision:.5f}")
        print(f"Recall:      {recall:.5f}")
        print(f"Specificity: {specificity:.5f}")
        print(f"F1-score:    {f1:.5f}")
        print(f"MCC:         {mcc:.5f}")
        print(f"ROC AUC:     {roc_auc:.5f}")
        print(f"PR AUC:      {pr_auc:.5f}")
        
        subset_results_filename = f'performance_{dataset_name}_rs{random_state_for_sampling}.txt'
        subset_results_file = output_dir / subset_results_filename
        with open(subset_results_file, 'w') as f:
            f.write(f"Evaluated on {len(test_df_subset)} random samples from {test_data_full_path.name}\n")
            f.write(f"Dataset: {dataset_name}\n") # Added dataset name
            f.write(f"Random seed for sampling: {random_state_for_sampling}\n") # Added random seed
            f.write("Metric\tValue\n")
            f.write(f"Accuracy\t{accuracy:.5f}\n")
            f.write(f"Precision\t{precision:.5f}\n")
            f.write(f"Recall\t{recall:.5f}\n")
            f.write(f"Specificity\t{specificity:.5f}\n")
            f.write(f"F1-score\t{f1:.5f}\n")
            f.write(f"MCC\t{mcc:.5f}\n")
            f.write(f"ROC AUC\t{roc_auc:.5f}\n")
            f.write(f"PR AUC\t{pr_auc:.5f}\n")
        print(f"Subset test results summary saved to {subset_results_file}")
    else:
        print("Not enough data or only one class in subset labels for full evaluation.")
        if len(all_subset_labels) > 0:
            print(f"Labels: {np.unique(all_subset_labels, return_counts=True)}")

    print("Inference with subset and unknown protein handling complete.")