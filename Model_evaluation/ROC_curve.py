import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lifelines.statistics import logrank_test
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tqdm import tqdm
from matplotlib import font_manager

import argparse

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_MODEL_DIR = _PROJECT_ROOT / 'Model'
_TRAINING_DIR = _PROJECT_ROOT / 'Model_training'
if _MODEL_DIR.exists():
    sys.path.insert(0, str(_MODEL_DIR))
if _TRAINING_DIR.exists():
    sys.path.insert(0, str(_TRAINING_DIR))

from model import DualViewHGNN
from dataset import HGNN_Dataset, collate_fn

# Font configuration
try:
    font_manager.fontManager.addfont('/scratchdata1/users/a1978372/Anxuan/arial.ttf')
    plt.rcParams['font.family'] = 'Arial'
except:
    print("Arial font not found, using default.")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

CLAM_SB = None
TransMIL = None
DSMIL = None
try:
    sys.path.append('/scratchdata1/users/a1978372/Anxuan/framework/try_clam')
    from model_clam import CLAM_SB  # type: ignore
except Exception as e:
    print(f"Warning: CLAM import failed ({e}); CLAM evaluation will be skipped unless paths/modules are fixed.")

try:
    sys.path.append('/scratchdata1/users/a1978372/Anxuan/framework/try_transmil')
    from model_transmil import TransMIL  # type: ignore
except Exception as e:
    print(f"Warning: TransMIL import failed ({e}); TransMIL evaluation will be skipped unless paths/modules are fixed.")

try:
    sys.path.append('/scratchdata1/users/a1978372/Anxuan/framework/try_dsmil')
    from model_dsmil import DSMIL  # type: ignore
except Exception as e:
    print(f"Warning: DSMIL import failed ({e}); DSMIL evaluation will be skipped unless paths/modules are fixed.")

CLINICAL_FILE = None

def load_clinical_data(clinical_file=None):
    """Load clinical data and return dictionary mapping patient_id to time"""
    clinical_file = clinical_file or CLINICAL_FILE
    if not clinical_file:
        return {}
    if not os.path.exists(clinical_file):
        print(f"Warning: Clinical file not found at {clinical_file}")
        return {}
    
    df = pd.read_csv(clinical_file)
    # Ensure patient_id is string and strip whitespace
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    time_dict = dict(zip(df['patient_id'], df['time']))
    return time_dict

def calculate_c_index(y_true, y_pred, pids, time_dict):
    """
    Calculate C-index (Concordance Index) for survival analysis.
    y_true: binary labels (0=censored/no recurrence, 1=event/recurrence)
    y_pred: predicted risk scores (higher probability = higher risk)
    pids: patient ids to look up time
    time_dict: dictionary mapping pid -> time
    """
    if not time_dict:
        return np.nan

    times = []
    events = []
    risks = []
    
    for i in range(len(y_true)):
        pid = str(pids[i]).strip()
        t = time_dict.get(pid)
        if t is None:
            continue
            
        times.append(t)
        events.append(y_true[i])
        risks.append(y_pred[i])
            
    n_subset = len(times)
    if n_subset < 2:
        return np.nan
        
    concordant = 0
    permissible = 0
    
    for i in range(n_subset):
        for j in range(n_subset):
            if i == j: continue
            
            if events[i] == 1 and times[i] < times[j]:
                permissible += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    concordant += 0.5
                    
    if permissible == 0:
        return np.nan
        
    return concordant / permissible

def bootstrap_c_index(y_true, y_pred, pids, time_dict, n_bootstraps=1000, seed=42):
    """
    Bootstrap C-index calculation to get Mean, Std and 95% CI.
    """
    if not time_dict or len(y_true) < 2:
        return np.nan, np.nan, np.nan, np.nan

    rng = np.random.RandomState(seed)
    c_indices = []
    n_samples = len(y_true)
    
    # Pre-validate if we can calculate C-index at all
    point_est = calculate_c_index(y_true, y_pred, pids, time_dict)
    if np.isnan(point_est):
        return np.nan, np.nan, np.nan, np.nan

    for _ in range(n_bootstraps):
        # Bootstrap resampling with replacement
        indices = rng.randint(0, n_samples, n_samples)
        
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        pids_boot = [pids[i] for i in indices]
        
        c = calculate_c_index(y_true_boot, y_pred_boot, pids_boot, time_dict)
        if not np.isnan(c):
            c_indices.append(c)
            
    if not c_indices:
        return np.nan, np.nan, np.nan, np.nan
        
    mean_c = np.mean(c_indices)
    std_c = np.std(c_indices)
    ci_lower = np.percentile(c_indices, 2.5)
    ci_upper = np.percentile(c_indices, 97.5)
    
    return mean_c, std_c, ci_lower, ci_upper

def calculate_logrank_p_value(y_true, y_pred, pids, time_dict, threshold=None):
    """
    Calculate Log-rank test P-value.
    """
    if not time_dict or len(y_true) < 2:
        return np.nan

    times = []
    events = []
    scores = []

    for i, pid in enumerate(pids):
        pid_clean = str(pid).strip()
        if pid_clean in time_dict:
            t = time_dict[pid_clean]
            e = y_true[i]
            times.append(t)
            events.append(e)
            scores.append(y_pred[i])
    
    if len(times) < 2:
        return np.nan

    times = np.array(times)
    events = np.array(events)
    scores = np.array(scores)
    
    if threshold is None:
        threshold = np.median(scores)
        
    group_labels = (scores >= threshold).astype(int)
    
    # If all in one group, return nan
    if len(np.unique(group_labels)) < 2:
        return np.nan
        
    results = logrank_test(times[group_labels==0], times[group_labels==1], 
                           event_observed_A=events[group_labels==0], 
                           event_observed_B=events[group_labels==1])
                           
    return results.p_value

# 设置图表样式
try:
    font_manager.fontManager.addfont('/scratchdata1/users/a1978372/Anxuan/arial.ttf')
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    print("Arial font not found, using default.")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Config (defaults will be overridden by argparse in main())
H5_DIR = str(_PROJECT_ROOT / 'Features')
CSV_PATH = str(_PROJECT_ROOT / 'data' / 'recurrence_information.csv')

# Optional test set paths
TEST_H5_DIR = None
TEST_CSV_PATH = None

# Model paths
HGNN_MODEL_PATH = str(_PROJECT_ROOT / 'Model_training' / 'results' / 'best_model.pth')
CLAM_MODEL_PATH = None
TRANSMIL_MODEL_PATH = None
DSMIL_MODEL_PATH = None

OUTPUT_DIR = str(_PROJECT_ROOT / 'Model_evaluation' / 'results')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DROPOUT = 0.6
SEED = 42
TRAIN_RATIO = 0.8
PID_COL = 'patient_id'
LABEL_COL = 'recurrence'
H5_NAME_FMT = '{pid}.h5'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DvHGNN (and optionally other models)')

    parser.add_argument('--dataset_type', type=str, default='validation', choices=['validation', 'test'],
                        help='Evaluate on validation split or test set')

    parser.add_argument('--h5_dir', type=str, default=H5_DIR)
    parser.add_argument('--csv_path', type=str, default=CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)

    parser.add_argument('--test_h5_dir', type=str, default=TEST_H5_DIR)
    parser.add_argument('--test_csv_path', type=str, default=TEST_CSV_PATH)

    parser.add_argument('--hgnn_model_path', type=str, default=HGNN_MODEL_PATH)
    parser.add_argument('--dropout', type=float, default=DROPOUT)

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO)

    parser.add_argument('--pid_col', type=str, default=PID_COL)
    parser.add_argument('--label_col', type=str, default=LABEL_COL)
    parser.add_argument('--h5_name_fmt', type=str, default=H5_NAME_FMT)

    parser.add_argument('--clinical_file', type=str, default=None,
                        help='Optional clinical CSV with columns patient_id,time (enables C-index & log-rank)')

    parser.add_argument('--all_models', action='store_true',
                        help='Also evaluate CLAM/TransMIL/DSMIL if available (requires their paths/modules)')
    parser.add_argument('--clam_model_path', type=str, default=CLAM_MODEL_PATH)
    parser.add_argument('--transmil_model_path', type=str, default=TRANSMIL_MODEL_PATH)
    parser.add_argument('--dsmil_model_path', type=str, default=DSMIL_MODEL_PATH)

    return parser.parse_args()

def evaluate_hgnn_model(dataset_type='validation'):
    """Load and evaluate HGNN model on validation or test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating DvHGNN Model on {dataset_type.upper()} set")
    print(f"{'='*60}")
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {HGNN_MODEL_PATH}")
    
    if dataset_type == 'test':
        # Load test dataset
        if not TEST_CSV_PATH or not TEST_H5_DIR:
            print("Test paths not provided. Set --test_csv_path and --test_h5_dir.")
            return None, None, None

        test_dataset = HGNN_Dataset(
            TEST_CSV_PATH,
            TEST_H5_DIR,
            training=False,
            pid_col=PID_COL,
            label_col=LABEL_COL,
            h5_name_fmt=H5_NAME_FMT,
        )
        
        if len(test_dataset) == 0:
            print("No test data found! Check paths.")
            return None, None, None
        
        print(f"Test set size: {len(test_dataset)}")
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
    else:  # validation
        # Load full dataset
        full_dataset = HGNN_Dataset(
            CSV_PATH,
            H5_DIR,
            training=False,
            pid_col=PID_COL,
            label_col=LABEL_COL,
            h5_name_fmt=H5_NAME_FMT,
        )
        
        if len(full_dataset) == 0:
            print("No data found! Check paths.")
            return None, None, None
        
        # Split indices (same as in train.py)
        train_size = int(TRAIN_RATIO * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Use the same random seed for reproducibility
        torch.manual_seed(SEED)
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        val_indices = val_subset.indices
        val_dataset = HGNN_Dataset(
            CSV_PATH,
            H5_DIR,
            training=False,
            pid_col=PID_COL,
            label_col=LABEL_COL,
            h5_name_fmt=H5_NAME_FMT,
        )
        val_dataset.data_list = [full_dataset.data_list[i] for i in val_indices]
        
        print(f"Validation set size: {len(val_dataset)}")
        loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load model
    model = DualViewHGNN(dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(HGNN_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Evaluate
    preds = []
    labels = []
    pids = []
    
    print(f"Evaluating DvHGNN model on {dataset_type} set...")
    with torch.no_grad():
        for features, clusters, knn_indices, label, pid in tqdm(loader):
            features = features[0].to(DEVICE)
            clusters = clusters[0].to(DEVICE)
            knn_indices = knn_indices[0].to(DEVICE)
            label = label.to(DEVICE)
            
            logits, A, _ = model(features, clusters, knn_indices)
            probs = torch.softmax(logits, dim=1)
            
            preds.extend(probs[:, 1].cpu().numpy())
            labels.extend(label.cpu().numpy())
            pids.extend(pid)
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    return preds, labels, pids

def evaluate_clam_model(dataset_type='validation'):
    """Load and evaluate CLAM model on validation or test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating CLAM Model on {dataset_type.upper()} set")
    print(f"{'='*60}")
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {CLAM_MODEL_PATH}")
    
    if dataset_type == 'test':
        # Load test dataset
        test_dataset = HGNN_Dataset(TEST_CSV_PATH, TEST_H5_DIR, training=False)
        
        if len(test_dataset) == 0:
            print("No test data found! Check paths.")
            return None, None, None
        
        print(f"Test set size: {len(test_dataset)}")
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
    else:  # validation
        # Load full dataset (same split as HGNN)
        full_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
        
        if len(full_dataset) == 0:
            print("No data found! Check paths.")
            return None, None, None
        
        # Split indices (same as HGNN)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        torch.manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        val_indices = val_subset.indices
        val_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
        val_dataset.data_list = [full_dataset.data_list[i] for i in val_indices]
        
        print(f"Validation set size: {len(val_dataset)}")
        loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load CLAM model
    model = CLAM_SB(gate=True, size_arg="small", dropout=True, k_sample=8, 
                    n_classes=2, instance_loss_fn=torch.nn.CrossEntropyLoss(), 
                    subtyping=False)
    model.to(DEVICE)
    model.load_state_dict(torch.load(CLAM_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Evaluate
    preds = []
    labels = []
    pids = []
    
    print(f"Evaluating CLAM model on {dataset_type} set...")
    with torch.no_grad():
        for features, clusters, knn_indices, label, pid in tqdm(loader):
            features = features[0].to(DEVICE)
            label = label.to(DEVICE)
            
            # CLAM only needs features
            logits, Y_prob, Y_hat, A_raw, results_dict = model(features, label=label, 
                                                                instance_eval=False, 
                                                                return_features=False, 
                                                                attention_only=False)
            
            preds.extend(Y_prob[:, 1].cpu().numpy())
            labels.extend(label.cpu().numpy())
            pids.extend(pid)
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    return preds, labels, pids

def evaluate_transmil_model(dataset_type='validation'):
    """Load and evaluate TransMIL model on validation or test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating TransMIL Model on {dataset_type.upper()} set")
    print(f"{'='*60}")
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {TRANSMIL_MODEL_PATH}")
    
    if dataset_type == 'test':
        test_dataset = HGNN_Dataset(TEST_CSV_PATH, TEST_H5_DIR, training=False)
        if len(test_dataset) == 0:
            print("No test data found!")
            return None, None, None
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    else:
        full_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        torch.manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_indices = val_subset.indices
        val_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
        val_dataset.data_list = [full_dataset.data_list[i] for i in val_indices]
        loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load Model
    model = TransMIL(n_classes=2).to(DEVICE)
    try:
        model.load_state_dict(torch.load(TRANSMIL_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Warning: TransMIL model not found at {TRANSMIL_MODEL_PATH}")
        return None, None, None
        
    model.eval()
    
    preds = []
    labels = []
    pids = []
    
    print(f"Evaluating TransMIL model on {dataset_type} set...")
    with torch.no_grad():
        for features, clusters, knn_indices, label, pid in tqdm(loader):
            features = features[0].to(DEVICE)
            label = label.to(DEVICE)
            
            logits, _ = model(features)
            probs = torch.softmax(logits, dim=1)
            
            preds.extend(probs[:, 1].cpu().numpy())
            labels.extend(label.cpu().numpy())
            pids.extend(pid)
            
    return np.array(preds), np.array(labels), pids

def evaluate_dsmil_model(dataset_type='validation'):
    """Load and evaluate DSMIL model on validation or test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating DSMIL Model on {dataset_type.upper()} set")
    print(f"{'='*60}")
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {DSMIL_MODEL_PATH}")
    
    if dataset_type == 'test':
        test_dataset = HGNN_Dataset(TEST_CSV_PATH, TEST_H5_DIR, training=False)
        if len(test_dataset) == 0:
            return None, None, None
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    else:
        full_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        torch.manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_indices = val_subset.indices
        val_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
        val_dataset.data_list = [full_dataset.data_list[i] for i in val_indices]
        loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load Model
    model = DSMIL(n_classes=2).to(DEVICE)
    try:
        model.load_state_dict(torch.load(DSMIL_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Warning: DSMIL model not found at {DSMIL_MODEL_PATH}")
        return None, None, None
        
    model.eval()
    
    preds = []
    labels = []
    pids = []
    
    print(f"Evaluating DSMIL model on {dataset_type} set...")
    with torch.no_grad():
        for features, clusters, knn_indices, label, pid in tqdm(loader):
            features = features[0].to(DEVICE)
            label = label.to(DEVICE)
            
            logits, _ = model(features)
            probs = torch.softmax(logits, dim=1)
            
            preds.extend(probs[:, 1].cpu().numpy())
            labels.extend(label.cpu().numpy())
            pids.extend(pid)
            
    return np.array(preds), np.array(labels), pids

def plot_combined_roc_curve(models_results, save_path, title_suffix=""):
    """Plot ROC curves for multiple models on the same figure"""
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    colors = ['#fab160', '#62ab64', '#f6796a', '#85ccbf', '#a05195', '#d45087']
    
    for idx, (model_name, y_true, y_pred) in enumerate(models_results):
        if y_true is None or y_pred is None:
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=1.5, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--', 
            label='Random', alpha=0.7)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold', color='black')
    ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold', color='black')
    ax.set_title(f'ROC Curves', fontsize=10, fontweight='bold', color='black', pad=5)
    
    # Legend style from dca_curve.py (fontsize=5, customized location if needed, but 'best' is safe)
    # User asked for "font settings and size 3.5,2.5", legend font smaller.
    ax.legend(loc="lower right", fontsize=5.8, frameon=True, shadow=True, 
              edgecolor='black', facecolor='white', labelcolor='black',handlelength=1  )
              
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    
    # 加粗边框 (Thicken borders)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    ax.tick_params(axis='both', colors='black', direction='out', labelsize=10, width=1, length=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nCombined ROC curve saved to: {save_path}")
    
    # Also save as PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, format='pdf')
    print(f"Combined ROC curve also saved as PDF to: {pdf_path}")
    
    plt.close()

def plot_roc_curve(y_true, y_pred, save_path, model_name):
    """Plot individual ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    # 选择颜色
    model_color = '#f6796a' if 'HGNN' in model_name else '#85ccbf'
    
    ax.plot(fpr, tpr, color=model_color, lw=1.2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--', 
            label='Random Classifier', alpha=0.7)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold', color='black')
    ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold', color='black')
    ax.set_title(f'{model_name} - ROC Curve', fontsize=10, fontweight='bold', color='black')
    ax.legend(loc="lower right", fontsize=6, frameon=True, shadow=True,
              edgecolor='black', facecolor='white', labelcolor='black')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8, color='gray')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    ax.tick_params(axis='both', colors='black', direction='out', labelsize=8, width=1, length=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"{model_name} ROC curve saved to: {save_path}")
    plt.close()
    
    return roc_auc, fpr, tpr, thresholds

def plot_confusion_matrix(y_true, y_pred, save_path, model_name, threshold=0.5):
    """Plot confusion matrix"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    # 自定义颜色映射
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='black',
                annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'},
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold', color='black')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold', color='black')
    ax.set_title(f'{model_name} - Confusion Matrix (Threshold = {threshold:.2f})', 
                 fontsize=10, fontweight='bold', color='black')
    
    ax.set_xticklabels(['No Recurrence (0)', 'Recurrence (1)'], 
                       fontsize=8, fontweight='bold', rotation=0)
    ax.set_yticklabels(['No Recurrence (0)', 'Recurrence (1)'], 
                       fontsize=8, fontweight='bold', rotation=90, va='center')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    ax.tick_params(axis='both', colors='black', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"{model_name} confusion matrix saved to: {save_path}")
    plt.close()
    
    return cm

def find_optimal_threshold(y_true, y_pred, fpr, tpr, thresholds, model_name):
    """Find optimal threshold using Youden's J statistic"""
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n{model_name} - Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"  - TPR: {tpr[optimal_idx]:.4f}")
    print(f"  - FPR: {fpr[optimal_idx]:.4f}")
    
    return optimal_threshold

def save_classification_report(y_true, y_pred, save_path, model_name, threshold=0.5):
    """Save classification report to file"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    report = classification_report(y_true, y_pred_binary, 
                                   target_names=['No Recurrence', 'Recurrence'],
                                   digits=4)
    
    with open(save_path, 'w') as f:
        f.write(f"{model_name} - Classification Report (Threshold = {threshold:.2f})\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"{model_name} classification report saved to: {save_path}")
    print("\n" + report)

def process_model_results(y_pred, y_true, pids, model_name, model_dir, time_dict=None):
    """Process and save results for a single model"""
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Positive samples (Recurrence): {np.sum(y_true)}")
    print(f"Negative samples (No Recurrence): {len(y_true) - np.sum(y_true)}")
    
    # Plot individual ROC curve
    roc_path = os.path.join(model_dir, f"{model_name}_roc_curve.png")
    roc_auc, fpr, tpr, thresholds = plot_roc_curve(y_true, y_pred, roc_path, model_name)
    print(f"\n{model_name} AUC Score: {roc_auc:.4f}")
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, y_pred, fpr, tpr, thresholds, model_name)
    
    # Calculate C-index and Log-rank P
    c_index_mean = np.nan
    c_index_std = np.nan
    c_index_lower = np.nan
    c_index_upper = np.nan
    logrank_p = np.nan
    
    if time_dict:
        c_index_mean, c_index_std, c_index_lower, c_index_upper = bootstrap_c_index(y_true, y_pred, pids, time_dict)
        logrank_p = calculate_logrank_p_value(y_true, y_pred, pids, time_dict, threshold=optimal_threshold)
        
        print(f"C-index: {c_index_mean:.4f} ± {c_index_std:.4f} (95% CI: {c_index_lower:.4f}-{c_index_upper:.4f})")
        print(f"Log-rank P: {logrank_p:.4e}")
    
    # Plot confusion matrices
    cm_path_default = os.path.join(model_dir, f"{model_name}_confusion_matrix_0.5.png")
    plot_confusion_matrix(y_true, y_pred, cm_path_default, model_name, threshold=0.5)
    
    cm_path_optimal = os.path.join(model_dir, f"{model_name}_confusion_matrix_optimal_{optimal_threshold:.2f}.png")
    plot_confusion_matrix(y_true, y_pred, cm_path_optimal, model_name, threshold=optimal_threshold)
    
    # Save classification reports
    report_path_default = os.path.join(model_dir, f"{model_name}_classification_report_0.5.txt")
    save_classification_report(y_true, y_pred, report_path_default, model_name, threshold=0.5)
    
    report_path_optimal = os.path.join(model_dir, f"{model_name}_classification_report_optimal_{optimal_threshold:.2f}.txt")
    save_classification_report(y_true, y_pred, report_path_optimal, model_name, threshold=optimal_threshold)
    
    # Save predictions
    pred_path = os.path.join(model_dir, f"{model_name}_predictions.txt")
    with open(pred_path, 'w') as f:
        f.write("patient_id,true_label,predicted_probability,predicted_label_0.5,predicted_label_optimal\n")
        for pid, true_label, pred_prob in zip(pids, y_true, y_pred):
            pred_binary_default = 1 if pred_prob >= 0.5 else 0
            pred_binary_optimal = 1 if pred_prob >= optimal_threshold else 0
            f.write(f"{pid},{true_label},{pred_prob:.6f},{pred_binary_default},{pred_binary_optimal}\n")
    
    print(f"{model_name} predictions saved to: {pred_path}")
    
    return roc_auc, c_index_mean, c_index_std, c_index_lower, c_index_upper, logrank_p

def evaluate_on_dataset(dataset_type='validation'):
    """Evaluate both models on a specific dataset (validation or test)"""
    print(f"\n{'#'*80}")
    print(f"# EVALUATING ON {dataset_type.upper()} SET")
    print(f"{'#'*80}")
    
    # Create output directory for this dataset
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Evaluate HGNN model
    hgnn_pred, hgnn_true, hgnn_pids = evaluate_hgnn_model(dataset_type)
    
    # Evaluate CLAM model
    clam_pred, clam_true, clam_pids = evaluate_clam_model(dataset_type)
    
    # Evaluate TransMIL model
    trans_pred, trans_true, trans_pids = evaluate_transmil_model(dataset_type)
    
    # Evaluate DSMIL model
    dsmil_pred, dsmil_true, dsmil_pids = evaluate_dsmil_model(dataset_type)
    
    # Create directories
    hgnn_dir = os.path.join(dataset_output_dir, "HGNN")
    clam_dir = os.path.join(dataset_output_dir, "CLAM")
    trans_dir = os.path.join(dataset_output_dir, "TransMIL")
    dsmil_dir = os.path.join(dataset_output_dir, "DSMIL")
    
    # Load clinical data
    time_dict = load_clinical_data()
    
    results_list = []
    
    if hgnn_pred is not None:
        hgnn_auc, hgnn_mean, hgnn_std, hgnn_low, hgnn_high, hgnn_p = process_model_results(hgnn_pred, hgnn_true, hgnn_pids, "DvHGNN", hgnn_dir, time_dict)
        results_list.append(("DvHGNN", hgnn_true, hgnn_pred))
    else:
        hgnn_auc, hgnn_mean, hgnn_std, hgnn_low, hgnn_high, hgnn_p = 0, 0, 0, 0, 0, 0
        
    if clam_pred is not None:
        clam_auc, clam_mean, clam_std, clam_low, clam_high, clam_p = process_model_results(clam_pred, clam_true, clam_pids, "CLAM-SB", clam_dir, time_dict)
        results_list.append(("CLAM-SB", clam_true, clam_pred))
    else:
        clam_auc, clam_mean, clam_std, clam_low, clam_high, clam_p = 0, 0, 0, 0, 0, 0
        
    if trans_pred is not None:
        trans_auc, trans_mean, trans_std, trans_low, trans_high, trans_p = process_model_results(trans_pred, trans_true, trans_pids, "TransMIL", trans_dir, time_dict)
        results_list.append(("TransMIL", trans_true, trans_pred))
    else:
        trans_auc, trans_mean, trans_std, trans_low, trans_high, trans_p = 0, 0, 0, 0, 0, 0
        
    if dsmil_pred is not None:
        dsmil_auc, dsmil_mean, dsmil_std, dsmil_low, dsmil_high, dsmil_p = process_model_results(dsmil_pred, dsmil_true, dsmil_pids, "DSMIL", dsmil_dir, time_dict)
        results_list.append(("DSMIL", dsmil_true, dsmil_pred))
    else:
        dsmil_auc, dsmil_mean, dsmil_std, dsmil_low, dsmil_high, dsmil_p = 0, 0, 0, 0, 0, 0
    
    # Plot combined ROC curve
    combined_roc_path = os.path.join(dataset_output_dir, "combined_roc_curves.png")
    plot_combined_roc_curve(results_list, combined_roc_path, 
                           title_suffix=f" ({dataset_type.capitalize()} Set)")
    
    # Save comparison summary
    summary_path = os.path.join(dataset_output_dir, "model_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model Comparison Summary - {dataset_type.upper()} Set\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"DvHGNN Model AUC: {hgnn_auc:.4f}, C-index: {hgnn_mean:.4f} ± {hgnn_std:.4f} (95% CI: {hgnn_low:.4f}-{hgnn_high:.4f}), Log-rank P: {hgnn_p:.4e}\n")
        f.write(f"CLAM-SB Model AUC: {clam_auc:.4f}, C-index: {clam_mean:.4f} ± {clam_std:.4f} (95% CI: {clam_low:.4f}-{clam_high:.4f}), Log-rank P: {clam_p:.4e}\n")
        f.write(f"TransMIL Model AUC: {trans_auc:.4f}, C-index: {trans_mean:.4f} ± {trans_std:.4f} (95% CI: {trans_low:.4f}-{trans_high:.4f}), Log-rank P: {trans_p:.4e}\n")
        f.write(f"DSMIL Model AUC: {dsmil_auc:.4f}, C-index: {dsmil_mean:.4f} ± {dsmil_std:.4f} (95% CI: {dsmil_low:.4f}-{dsmil_high:.4f}), Log-rank P: {dsmil_p:.4e}\n")
    
    print(f"\n{'='*60}")
    print(f"Model Comparison Summary - {dataset_type.upper()} Set")
    print(f"{'='*60}")
    print(f"DvHGNN AUC: {hgnn_auc:.4f}")
    print(f"CLAM-SB AUC: {clam_auc:.4f}")
    print(f"AUC Difference: {abs(hgnn_auc - clam_auc):.4f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"Results saved in: {dataset_output_dir}")
    
    return hgnn_auc, clam_auc, trans_auc, dsmil_auc

def main():
    args = parse_args()

    global H5_DIR, CSV_PATH, OUTPUT_DIR, DEVICE, DROPOUT, SEED, TRAIN_RATIO
    global PID_COL, LABEL_COL, H5_NAME_FMT
    global TEST_H5_DIR, TEST_CSV_PATH
    global HGNN_MODEL_PATH, CLAM_MODEL_PATH, TRANSMIL_MODEL_PATH, DSMIL_MODEL_PATH
    global CLINICAL_FILE

    H5_DIR = args.h5_dir
    CSV_PATH = args.csv_path
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DEVICE = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    DROPOUT = args.dropout
    SEED = args.seed
    TRAIN_RATIO = args.train_ratio

    PID_COL = args.pid_col
    LABEL_COL = args.label_col
    H5_NAME_FMT = args.h5_name_fmt

    TEST_H5_DIR = args.test_h5_dir
    TEST_CSV_PATH = args.test_csv_path

    HGNN_MODEL_PATH = args.hgnn_model_path
    CLAM_MODEL_PATH = args.clam_model_path
    TRANSMIL_MODEL_PATH = args.transmil_model_path
    DSMIL_MODEL_PATH = args.dsmil_model_path
    CLINICAL_FILE = args.clinical_file

    print(f"\n{'*'*80}")
    print(f"* MODEL EVALUATION SCRIPT")
    if args.all_models:
        print(f"* Evaluating DvHGNN, CLAM-SB, TransMIL, DSMIL")
    else:
        print(f"* Evaluating DvHGNN only")
    print(f"{'*'*80}\n")

    if args.all_models:
        # Keep original behavior
        val_res = evaluate_on_dataset('validation')
        if val_res is None:
            val_res = (0, 0, 0, 0)
        test_res = evaluate_on_dataset('test')
        if test_res is None:
            test_res = (0, 0, 0, 0)

        val_hgnn_auc, val_clam_auc, val_trans_auc, val_dsmil_auc = val_res
        test_hgnn_auc, test_clam_auc, test_trans_auc, test_dsmil_auc = test_res

        overall_summary_path = os.path.join(OUTPUT_DIR, 'overall_comparison_summary.txt')
        with open(overall_summary_path, 'w') as f:
            f.write('Overall Model Comparison Summary\n')
            f.write('=' * 80 + '\n\n')
            f.write('VALIDATION SET RESULTS:\n')
            f.write('-' * 80 + '\n')
            f.write(f"  DvHGNN AUC:    {val_hgnn_auc:.4f}\n")
            f.write(f"  CLAM-SB AUC: {val_clam_auc:.4f}\n")
            f.write(f"  TransMIL AUC:{val_trans_auc:.4f}\n")
            f.write(f"  DSMIL AUC:   {val_dsmil_auc:.4f}\n\n")

            f.write('TEST SET RESULTS:\n')
            f.write('-' * 80 + '\n')
            f.write(f"  DvHGNN AUC:    {test_hgnn_auc:.4f}\n")
            f.write(f"  CLAM-SB AUC: {test_clam_auc:.4f}\n")
            f.write(f"  TransMIL AUC:{test_trans_auc:.4f}\n")
            f.write(f"  DSMIL AUC:   {test_dsmil_auc:.4f}\n\n")

        print(f"\nResults saved in: {OUTPUT_DIR}")
        print(f"Overall summary: {overall_summary_path}")
        return

    # HGNN-only mode
    dataset_type = args.dataset_type
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    hgnn_dir = os.path.join(dataset_output_dir, 'HGNN')

    time_dict = load_clinical_data(args.clinical_file)

    hgnn_pred, hgnn_true, hgnn_pids = evaluate_hgnn_model(dataset_type)
    if hgnn_pred is None:
        raise RuntimeError('HGNN evaluation failed. Please check paths.')

    process_model_results(hgnn_pred, hgnn_true, hgnn_pids, 'DvHGNN', hgnn_dir, time_dict)
    print(f"\nDvHGNN evaluation done. Results in: {hgnn_dir}")

if __name__ == "__main__":
    main()