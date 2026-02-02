import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HGNN_Dataset, collate_fn
from model import DualViewHGNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

# Config
H5_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/integrated/features"
CSV_PATH = "/scratchdata1/users/a1978372/Anxuan/framework/folder/rec_clean_2_final.csv"
OUTPUT_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/try_gated/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

def train():
    print(f"Using device: {DEVICE}")
    
    # Dataset
    # We initialize the full dataset first to get the split, but we need separate instances for train/val 
    # to control the 'training' flag.
    full_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
    
    if len(full_dataset) == 0:
        print("No data found! Check paths.")
        return

    # Split indices
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create new datasets with correct flags using the indices
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    
    # Re-initialize to ensure clean state
    train_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=True, dropout_rate=0.4)
    train_dataset.data_list = [full_dataset.data_list[i] for i in train_indices]
    
    val_dataset = HGNN_Dataset(CSV_PATH, H5_DIR, training=False)
    val_dataset.data_list = [full_dataset.data_list[i] for i in val_indices]
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = DualViewHGNN(dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for features, clusters, knn_indices, label, pid in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # Batch size is 1, so we take the first element
            features = features[0].to(DEVICE)
            clusters = clusters[0].to(DEVICE)
            knn_indices = knn_indices[0].to(DEVICE)
            label = label.to(DEVICE) # (1,)
            
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
        with torch.no_grad():
            for features, clusters, knn_indices, label, pid in val_loader:
                features = features[0].to(DEVICE)
                clusters = clusters[0].to(DEVICE)
                knn_indices = knn_indices[0].to(DEVICE)
                label = label.to(DEVICE)
                
                logits, A, _ = model(features, clusters, knn_indices)
                val_preds.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                val_labels.extend(label.cpu().numpy())
        
        try:
            val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0
        except:
            val_auc = 0
            
        print(f"Epoch {epoch+1} Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    train()
