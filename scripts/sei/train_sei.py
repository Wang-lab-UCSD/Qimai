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
from scipy.interpolate import splev # For B-spline calculation

def dna_to_one_hot(dna_sequence, seq_len=None):
    if seq_len is None:
        seq_len = len(dna_sequence)
    
    if len(dna_sequence) < seq_len:
        dna_sequence += 'N' * (seq_len - len(dna_sequence)) 
    elif len(dna_sequence) > seq_len:
        dna_sequence = dna_sequence[:seq_len] 

    mapping = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0,0,0,0]}
    one_hot_encoding = np.array([mapping.get(nucleotide.upper(), [0,0,0,0]) for nucleotide in dna_sequence], dtype=np.float32)
    return one_hot_encoding.T


def evaluate_binary(y_pred_probs, y_label):
    try:
        # ROC AUC and PR AUC are calculated on probabilities
        roc_auc = roc_auc_score(y_label, y_pred_probs)
        pr_auc = average_precision_score(y_label, y_pred_probs)

        # Determine optimal threshold for binary metrics
        fpr, tpr, thresholds = roc_curve(y_label, y_pred_probs)
        J = tpr - fpr
        if len(J) == 0 or len(thresholds) == 0:
            print("Warning: Could not determine optimal threshold (J or thresholds is empty). Using 0.5.")
            thred_optim = 0.5
        else:
            if len(thresholds) > 1 :
                 ix = np.argmax(J)
                 thred_optim = thresholds[ix]
            else: 
                 thred_optim = 0.5 
                 print(f"Warning: Only one threshold found in ROC curve. Using {thred_optim} as optimal threshold.")

        print(f"Optimal threshold for binary metrics: {thred_optim:.4f}")
        y_pred_binary = (y_pred_probs >= thred_optim).astype(int)
        
        cm = confusion_matrix(y_label, y_pred_binary)
        
        if cm.size == 1: 
             if y_label[0] == 0 and y_pred_binary[0] == 0 : # All TN
                 tn, fp, fn, tp = len(y_label), 0, 0, 0
             elif y_label[0] == 1 and y_pred_binary[0] == 1: # All TP
                 tn, fp, fn, tp = 0, 0, 0, len(y_label)
             elif y_label[0] == 0 and y_pred_binary[0] == 1: # All FP
                 tn, fp, fn, tp = 0, len(y_label), 0, 0
             elif y_label[0] == 1 and y_pred_binary[0] == 0: # All FN
                 tn, fp, fn, tp = 0, 0, len(y_label), 0
             else: 
                 tn, fp, fn, tp = 0,0,0,0 
        elif cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else: 
            print(f"Warning: Unexpected confusion matrix shape: {cm.shape}. Metrics might be incorrect.")
            tn, fp, fn, tp = 0,0,0,0

        total_samples = tn + fp + fn + tp
        
        print("Confusion Matrix (tn, fp, fn, tp):")
        print(f'{tn} | {fp}')
        print(f'--- ---')
        print(f'{fn} | {tp}')
        
        accuracy = (tn + tp) / total_samples if total_samples > 0 else 0
        precision = precision_score(y_label, y_pred_binary, zero_division=0)
        recall = recall_score(y_label, y_pred_binary, zero_division=0) 
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_label, y_pred_binary, zero_division=0)
        mcc = matthews_corrcoef(y_label, y_pred_binary)
        
        return accuracy, precision, recall, specificity, roc_auc, pr_auc, mcc, f1

    except ValueError as e:
        print(f"Evaluation error: {e}. Returning zeros for all metrics.")
        return 0, 0, 0, 0, 0, 0, 0, 0


# --- B-Spline Helper Functions (from sei.py) ---
def bs(x, df=None, knots=None, degree=3, intercept=False):
    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            # print("df was too small; have used %d" % (order - (1 - intercept)))
        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])
    elif knots is not None:
        inner_knots = knots
    
    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))
    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    # Handle cases where x might be a single point or very few points
    if len(x) == 0:
        return np.empty((0, n_basis if intercept else max(0, n_basis -1)), dtype=float)
    if n_basis <= 0: # Not enough knots to form a basis
        if intercept:
            # Return a constant basis if intercept is true and no other basis can be formed
            return np.ones((x.shape[0], 1), dtype=float) if n_basis == 0 and order == 1 else np.empty((x.shape[0], 0), dtype=float)
        else:
            return np.empty((x.shape[0], 0), dtype=float)


    basis = np.empty((x.shape[0], n_basis), dtype=float)
    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        try:
            basis[:, i] = splev(x, (all_knots, coefs, degree))
        except ValueError as e: # Can happen if x is outside knot range due to insufficient unique values
            # print(f"Splev warning/error: {e}. This might happen with very short sequences or non-diverse data for splines.")
            # Fallback or re-raise depending on strictness. For now, fill with 0.
            basis[:,i] = 0 


    if not intercept and basis.shape[1] > 0:
        basis = basis[:, 1:]
    return basis

def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * ( 2*(dist>0)-1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist),np.max(dist),n_knots+2)[1:-1] if n_knots > 0 else None
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)).float()

class BSplineTransformation(nn.Module):
    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input_tensor): # Renamed from 'input' to 'input_tensor'
        if self._spline_tr is None:
            spatial_dim = input_tensor.size()[-1]
            if spatial_dim == 0 : # Should not happen if input is valid
                 # This case needs to be handled: either raise error or return input_tensor
                 print("Warning: BSplineTransformation received input with spatial_dim 0.")
                 return input_tensor # Or an appropriately shaped zero tensor
            
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input_tensor.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        
        if self._spline_tr.shape[0] != input_tensor.size(-1): # Recompute if length changed
            spatial_dim = input_tensor.size()[-1]
            if spatial_dim == 0 :
                 print("Warning: BSplineTransformation (recompute) received input with spatial_dim 0.")
                 return input_tensor
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input_tensor.is_cuda:
                self._spline_tr = self._spline_tr.cuda()

        return torch.matmul(input_tensor, self._spline_tr)

# --- Dataset Class (from original script) ---
class ProteinDNADataset(Data.Dataset):
    def __init__(self, pkl_file_path, protein_to_id_map, dna_seq_len):
        self.dna_seq_len = dna_seq_len
        print(f"Reading data index from {pkl_file_path}...")
        try:
            self.df = pd.read_pickle(pkl_file_path)[['dna', 'protein', 'label']]
        except Exception as e:
            print(f"Error reading pickle file {pkl_file_path}: {e}")
            raise e
        self.protein_to_id_map = protein_to_id_map
        print(f"Dataset initialized with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dna_seq_str = row['dna']
        protein_name = row['protein']
        label = row['label']

        dna_one_hot = dna_to_one_hot(dna_seq_str, self.dna_seq_len)
        dna_tensor = torch.FloatTensor(dna_one_hot)

        protein_id = self.protein_to_id_map.get(protein_name)
        if protein_id is None:
            print(f"Warning: Protein '{protein_name}' at index {idx} not in map. Using ID 0.")
            protein_id = 0 
        protein_id_tensor = torch.LongTensor([protein_id])
        label_tensor = torch.FloatTensor([label])
        return dna_tensor, protein_id_tensor.squeeze(), label_tensor
    
# --- Model Definition (Sei Adapted for Protein Interaction) ---
class SeiProteinInteraction(nn.Module):
    def __init__(self, sequence_length, num_proteins, protein_emb_dim, n_genomic_features_output=1):
        super(SeiProteinInteraction, self).__init__()
        # Sei Backbone
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4), # padding = (kernel_size-1)*dilation / 2
            nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50), # Original padding for k=5, d=25 would be 50 for 'same'
            nn.ReLU(inplace=True))

        self._spline_df = int(128/8) # 16       
        self.spline_tr_module = nn.Sequential( # Renamed to avoid conflict if self.spline_tr is used as tensor
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False))

        # Protein Embedding
        self.protein_embedding = nn.Embedding(num_proteins, protein_emb_dim)

        # Classifier
        # Input features: DNA features (960 channels * spline_df) + protein_emb_dim
        fc1_in_features = 960 * self._spline_df + protein_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fc1_in_features, 925), # Using a hidden dim similar to DeepSEA for fc1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), # Added dropout, common in classifiers
            nn.Linear(925, n_genomic_features_output) 
        )

    def forward(self, dna_input, protein_id_input):
        # DNA processing (Sei backbone)
        lout1 = self.lconv1(dna_input)
        out1 = self.conv1(lout1)
        
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        dna_features = cat_out4 + dconv_out5 # Shape: (batch, 960, reduced_seq_len)
        
        spline_out = self.spline_tr_module(dna_features) # Shape: (batch, 960, _spline_df)
        # Reshape DNA features
        dna_features_flat = spline_out.view(spline_out.size(0), 960 * self._spline_df)

        # Protein processing
        protein_emb = self.protein_embedding(protein_id_input)
        protein_emb = protein_emb.squeeze(1) if protein_emb.ndim == 3 else protein_emb

        # Combine features
        combined_features = torch.cat((dna_features_flat, protein_emb), dim=1)
        
        # Classifier
        output_logits = self.classifier(combined_features)
        return output_logits

# --- Main Script Logic ---
if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    L_IN_DNA = 512 
    PROTEIN_EMB_DIM = 50 
    EPOCH = 10 
    BATCH_SIZE = 64 
    NUM_WORKERS = 8 
    LR = 0.001 # This LR was for Adam in DeepSEA, might need tuning for SGD
    TQDM_MININTERVAL = 300.0 # Minimum update interval for tqdm in seconds (e.g., 10 seconds)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    data_folder = '/new-stg/home/cong/DPI/dataset/Encode3and4/deepsea/data/' #MODIFY AS NEEDED
    train_pkl_path = Path(data_folder) / 'train_min10.pkl' 
    valid_pkl_path = Path(data_folder) / 'valid_min10.pkl'
    test_pkl_path  = Path(data_folder) / 'test_min10.pkl'

    print("Building protein_to_id map...")
    try:
        df_train_proteins = pd.read_pickle(train_pkl_path)['protein'] if train_pkl_path.exists() else pd.Series(dtype=str)
        df_valid_proteins = pd.read_pickle(valid_pkl_path)['protein'] if valid_pkl_path.exists() else pd.Series(dtype=str)
        df_test_proteins = pd.read_pickle(test_pkl_path)['protein'] if test_pkl_path.exists() else pd.Series(dtype=str)
        
        all_proteins = pd.concat([df_train_proteins, df_valid_proteins, df_test_proteins]).unique()
        all_proteins = sorted([p for p in all_proteins if pd.notna(p)])
        
        protein_to_id_map = {protein: i for i, protein in enumerate(all_proteins)}
        NUM_PROTEINS = len(protein_to_id_map)
        print(f"Found {NUM_PROTEINS} unique proteins.")
        if NUM_PROTEINS == 0:
            raise ValueError("No proteins found. Check your .pkl files and paths.")
        
        output_base_dir = Path("./benchmark/sei/") # CHANGED
        output_base_dir.mkdir(parents=True, exist_ok=True)
        protein_map_save_path = output_base_dir / "protein_to_id_map.json" # CHANGED

        try:
            import json
            with open(protein_map_save_path, 'w') as f:
                json.dump(protein_to_id_map, f, indent=4)
            print(f"Saved protein_to_id_map to {protein_map_save_path}")
        except Exception as e:
            print(f"Error saving protein_to_id_map: {e}")

    except FileNotFoundError as e:
        print(f"Error: pkl file not found: {e.filename}. Cannot build protein map. Exiting.")
        exit()
    except Exception as e:
        print(f"An error occurred building protein map: {e}")
        exit()

    print("Loading training data...")
    train_dataset = ProteinDNADataset(train_pkl_path, protein_to_id_map, L_IN_DNA)
    train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    print("Loading validation data...")
    valid_dataset = ProteinDNADataset(valid_pkl_path, protein_to_id_map, L_IN_DNA)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)

    print("Loading test data...")
    test_dataset = ProteinDNADataset(test_pkl_path, protein_to_id_map, L_IN_DNA)
    test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)

    model = SeiProteinInteraction(
        sequence_length=L_IN_DNA,
        num_proteins=NUM_PROTEINS, 
        protein_emb_dim=PROTEIN_EMB_DIM,
        n_genomic_features_output=1 # For binary classification
    ).to(DEVICE)
    
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in SeiProteinInteraction model: {total_params}")

    loss_func = nn.BCEWithLogitsLoss()
    # Optimizer changed to SGD as per sei.py suggestion style
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-7, momentum=0.9)
    
    output_dir = Path("./benchmark/sei/") # CHANGED
    output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_metric = -1 

    for epoch in range(EPOCH):
        model.train()
        train_loss_accum = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} [T]", unit="batch", mininterval=TQDM_MININTERVAL)
        for dna_batch, protein_id_batch, label_batch in progress_bar:
            dna_batch, protein_id_batch, label_batch = dna_batch.to(DEVICE), protein_id_batch.to(DEVICE), label_batch.to(DEVICE)
            optimizer.zero_grad()
            output_logits = model(dna_batch, protein_id_batch)
            loss = loss_func(output_logits, label_batch)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()
            progress_bar.set_postfix(loss=loss.item(), refresh=False)
        avg_train_loss = train_loss_accum / len(train_loader)

        model.eval()
        valid_loss_accum = 0
        all_valid_preds, all_valid_labels = [], []
        with torch.no_grad():
            progress_bar_valid = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCH} [V]", unit="batch", mininterval=TQDM_MININTERVAL)
            for dna_batch, protein_id_batch, label_batch in progress_bar_valid:
                dna_batch, protein_id_batch, label_batch = dna_batch.to(DEVICE), protein_id_batch.to(DEVICE), label_batch.to(DEVICE)
                output_logits = model(dna_batch, protein_id_batch)
                loss = loss_func(output_logits, label_batch)
                valid_loss_accum += loss.item()
                all_valid_preds.append(torch.sigmoid(output_logits).cpu().numpy())
                all_valid_labels.append(label_batch.cpu().numpy())
                progress_bar_valid.set_postfix(loss=loss.item(), refresh=False)
        avg_valid_loss = valid_loss_accum / len(valid_loader)
        all_valid_preds = np.concatenate(all_valid_preds).flatten()
        all_valid_labels = np.concatenate(all_valid_labels).flatten()
        
        acc, precis, recal, specif, roc_auc_val, pr_auc_val, mcc_val, f1_val = evaluate_binary(all_valid_preds, all_valid_labels)
        print(f"Epoch {epoch+1}/{EPOCH} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        print(f"Valid Metrics: Acc: {acc:.4f}, Prc: {precis:.4f}, Rec: {recal:.4f}, Spc: {specif:.4f}, ROC: {roc_auc_val:.4f}, PRC: {pr_auc_val:.4f}, MCC: {mcc_val:.4f}, F1: {f1_val:.4f}")

        current_metric_for_best = pr_auc_val 
        if current_metric_for_best > best_valid_metric:
            best_valid_metric = current_metric_for_best
            torch.save(model.state_dict(), output_dir / 'best_model_params.pkl')
            print(f"Saved best model parameters at epoch {epoch+1} (PR AUC: {best_valid_metric:.4f})")
    
    torch.save(model.state_dict(), output_dir / 'final_model_params.pkl')
    print("Saved final model parameters.")

    print("\nStarting testing with SeiProteinInteraction model...")
    best_model_path = output_dir / 'best_model_params.pkl'
    if best_model_path.exists():
        print("Loading best model for testing.")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Best model not found, using final model for testing.")
        # Attempt to load final model if best doesn't exist
        final_model_path = output_dir / 'final_model_params.pkl'
        if final_model_path.exists():
            model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
        else:
            print("Error: Neither best_model_params.pkl nor final_model_params.pkl found. Cannot test.")
            exit()

    model.eval()
    test_loss_accum = 0
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        progress_bar_test = tqdm(test_loader, desc="Testing", unit="batch", mininterval=TQDM_MININTERVAL)
        for dna_batch, protein_id_batch, label_batch in progress_bar_test:
            dna_batch, protein_id_batch, label_batch = dna_batch.to(DEVICE), protein_id_batch.to(DEVICE), label_batch.to(DEVICE)
            output_logits = model(dna_batch, protein_id_batch)
            loss = loss_func(output_logits, label_batch)
            test_loss_accum += loss.item()
            all_test_preds.append(torch.sigmoid(output_logits).cpu().numpy())
            all_test_labels.append(label_batch.cpu().numpy())
            progress_bar_test.set_postfix(loss=loss.item(), refresh=False)
    avg_test_loss = test_loss_accum / len(test_loader)
    all_test_preds = np.concatenate(all_test_preds).flatten()
    all_test_labels = np.concatenate(all_test_labels).flatten()

    print(f"\nTest Loss: {avg_test_loss:.5f}")
    np.save(output_dir / 'test_predictions.npy', all_test_preds)
    np.save(output_dir / 'test_labels.npy', all_test_labels)
    print(f"Saved test predictions and labels to {output_dir}")

    accuracy, precision, recall, specificity, roc_auc, pr_auc, mcc, f1 = evaluate_binary(all_test_preds, all_test_labels)
    
    print("\n--- Test Set Performance (SeiProteinInteraction) ---")
    print(f"Accuracy:    {accuracy:.5f}")
    print(f"Precision:   {precision:.5f}")
    print(f"Recall:      {recall:.5f}")
    print(f"Specificity: {specificity:.5f}")
    print(f"F1-score:    {f1:.5f}")
    print(f"MCC:         {mcc:.5f}")
    print(f"ROC AUC:     {roc_auc:.5f}")
    print(f"PR AUC:      {pr_auc:.5f}")

    results_file = output_dir / 'test_results_summary.txt'
    with open(results_file, 'w') as f:
        f.write("Metric\tValue\n")
        f.write(f"Test Loss\t{avg_test_loss:.5f}\n")
        f.write(f"Accuracy\t{accuracy:.5f}\n")
        f.write(f"Precision\t{precision:.5f}\n")
        f.write(f"Recall\t{recall:.5f}\n")
        f.write(f"Specificity\t{specificity:.5f}\n")
        f.write(f"F1-score\t{f1:.5f}\n")
        f.write(f"MCC\t{mcc:.5f}\n")
        f.write(f"ROC AUC\t{roc_auc:.5f}\n")
        f.write(f"PR AUC\t{pr_auc:.5f}\n")
    print(f"Test results summary saved to {results_file}")
