import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HGNN_Dataset, collate_fn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_MODEL_DIR = _PROJECT_ROOT / "Model"
if _MODEL_DIR.exists():
    sys.path.insert(0, str(_MODEL_DIR))
from model import DualViewHGNN


def parse_args():
    parser = argparse.ArgumentParser(description='Train DualViewHGNN Model')

    # Data paths
    parser.add_argument('--h5_dir', type=str,
                        default=str(_PROJECT_ROOT / 'Features'),
                        help='Directory containing H5 feature files')
    parser.add_argument('--csv_path', type=str,
                        default=str(_PROJECT_ROOT / 'data' / 'recurrence_information.csv'),
                        help='Path to CSV file with labels')
    parser.add_argument('--output_dir', type=str,
                        default=str(_PROJECT_ROOT / 'Model_training' / 'results'),
                        help='Directory to save results and models')

    # CSV schema / filename conventions
    parser.add_argument('--pid_col', type=str, default='patient_id',
                        help='Column name for patient/slide id in CSV')
    parser.add_argument('--label_col', type=str, default='recurrence',
                        help='Column name for label in CSV')
    parser.add_argument('--h5_name_fmt', type=str, default='{pid}.h5',
                        help="Format string to map pid -> h5 filename (supports {pid} or {patient_id})")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate for model')
    parser.add_argument('--train_dropout', type=float, default=0.4,
                        help='Dropout rate for training data augmentation')

    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (this script assumes 1)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Saving
    parser.add_argument('--save_last', action='store_true',
                        help='Also save last_model.pth each epoch (guarantees a saved checkpoint)')

    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print("Training configuration:")
    print(f"  h5_dir: {args.h5_dir}")
    print(f"  csv_path: {args.csv_path}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  pid_col: {args.pid_col}")
    print(f"  label_col: {args.label_col}")
    print(f"  h5_name_fmt: {args.h5_name_fmt}")
    print(f"  epochs: {args.epochs}")
    print(f"  lr: {args.lr}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  dropout: {args.dropout}")
    print(f"  train_dropout: {args.train_dropout}")
    print(f"  train_ratio: {args.train_ratio}")
    print(f"  batch_size: {args.batch_size}")
    
    # Dataset
    # We initialize the full dataset first to get the split, but we need separate instances for train/val 
    # to control the 'training' flag.
    if args.batch_size != 1:
        raise ValueError('This training script currently assumes --batch_size 1 (variable-length bags).')

    full_dataset = HGNN_Dataset(
        args.csv_path,
        args.h5_dir,
        training=False,
        pid_col=args.pid_col,
        label_col=args.label_col,
        h5_name_fmt=args.h5_name_fmt,
    )
    
    if len(full_dataset) == 0:
        print("No data found! Check paths.")
        return

    # Split indices
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create new datasets with correct flags using the indices
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    
    # Re-initialize to ensure clean state
    train_dataset = HGNN_Dataset(
        args.csv_path,
        args.h5_dir,
        training=True,
        dropout_rate=args.train_dropout,
        pid_col=args.pid_col,
        label_col=args.label_col,
        h5_name_fmt=args.h5_name_fmt,
    )
    train_dataset.data_list = [full_dataset.data_list[i] for i in train_indices]
    
    val_dataset = HGNN_Dataset(
        args.csv_path,
        args.h5_dir,
        training=False,
        pid_col=args.pid_col,
        label_col=args.label_col,
        h5_name_fmt=args.h5_name_fmt,
    )
    val_dataset.data_list = [full_dataset.data_list[i] for i in val_indices]
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = DualViewHGNN(dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_auc = float('-inf')
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for features, clusters, knn_indices, label, pid in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Batch size is 1, so we take the first element
            features = features[0].to(device)
            clusters = clusters[0].to(device)
            knn_indices = knn_indices[0].to(device)
            label = label.to(device) # (1,)
            
            optimizer.zero_grad()
            logits, A, _ = model(features, clusters, knn_indices)
            
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            train_labels.extend(label.cpu().numpy())
            
        # Calculate metrics
        try:
            train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0
        except:
            train_auc = 0
            
        print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f} AUC: {train_auc:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for features, clusters, knn_indices, label, pid in val_loader:
                features = features[0].to(device)
                clusters = clusters[0].to(device)
                knn_indices = knn_indices[0].to(device)
                label = label.to(device)

                logits, A, _ = model(features, clusters, knn_indices)
                loss = criterion(logits, label)
                val_loss += loss.item()
                val_preds.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        val_loss = val_loss / max(1, len(val_loader))
        
        val_has_both_classes = len(set([int(x) for x in val_labels])) > 1
        if val_has_both_classes:
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except Exception:
                val_auc = float('nan')
        else:
            val_auc = float('nan')

        if np.isnan(val_auc):
            print(f"Epoch {epoch+1} Val AUC: N/A (single class in val) | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} Val AUC: {val_auc:.4f} | Val Loss: {val_loss:.4f}")

        saved_best = False
        if not np.isnan(val_auc):
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                saved_best = True
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                saved_best = True

        if saved_best:
            print('Saved best model.')

        if args.save_last:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'last_model.pth'))

if __name__ == "__main__":
    args = parse_args()
    train(args)
