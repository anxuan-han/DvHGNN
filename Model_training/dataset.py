import os
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class HGNN_Dataset(Dataset):
    def __init__(self, csv_path, h5_dir, training=False, dropout_rate=0.25):
        """
        Args:
            csv_path (str): Path to the CSV file with labels.
            h5_dir (str): Directory containing H5 feature files.
            training (bool): Whether in training mode (enables augmentation).
            dropout_rate (float): Probability of dropping patches during training.
        """
        self.csv_path = csv_path
        self.h5_dir = h5_dir
        self.training = training
        self.dropout_rate = dropout_rate
        
        # Read CSV
        self.df = pd.read_csv(csv_path)
        
        # Get all h5 files recursively
        self.h5_files = {}
        for root, dirs, files in os.walk(h5_dir):
            for f in files:
                if f.endswith('.h5'):
                    self.h5_files[f] = os.path.join(root, f)
        
        self.data_list = []
        missing_count = 0
        
        print(f"Scanning data... CSV has {len(self.df)} records.")
        
        for idx, row in self.df.iterrows():
            pid = row['patient_id']
            label = int(row['recurrence'])
            
            # Construct expected filename
            # Assuming filename is patient_id.h5 based on previous context
            h5_name = f"{pid}.h5"
            
            if h5_name in self.h5_files:
                self.data_list.append({
                    'patient_id': pid,
                    'h5_path': self.h5_files[h5_name],
                    'label': label
                })
            else:
                missing_count += 1
                
        print(f"Dataset initialized.")
        print(f"  - Found: {len(self.data_list)}")
        print(f"  - Missing H5: {missing_count}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        h5_path = item['h5_path']
        label = item['label']
        pid = item['patient_id']
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Read datasets
                features = torch.from_numpy(f['features'][:]).float() # (N, 1024)
                
                # Check if clusters exist, if not, create dummy or handle error
                if 'clusters' in f:
                    clusters = torch.from_numpy(f['clusters'][:]).long() # (N,)
                else:
                    # Fallback if missing (should not happen based on user desc)
                    print(f"Warning: 'clusters' missing in {pid}, using zeros.")
                    clusters = torch.zeros(features.shape[0], dtype=torch.long)

                # Check if knn_indices exist
                if 'knn_indices' in f:
                    knn_indices = torch.from_numpy(f['knn_indices'][:]).long() # (N, k)
                else:
                    # Fallback
                    # print(f"Warning: 'knn_indices' missing in {pid}, using self-loops.")
                    N = features.shape[0]
                    k = 10 # default
                    knn_indices = torch.arange(N).view(-1, 1).repeat(1, k)

            # --- Patch Dropout Augmentation ---
            if self.training and features.shape[0] > 100:
                # Randomly keep (1 - dropout_rate) to 100% of patches
                # But to be simple and effective, let's just drop a fixed percentage or random percentage
                # Strategy: Keep at least 50% of patches, up to 100%
                keep_prob = np.random.uniform(1.0 - self.dropout_rate, 1.0)
                num_keep = int(features.shape[0] * keep_prob)
                
                if num_keep < features.shape[0]:
                    # Randomly select indices to keep
                    keep_indices = torch.randperm(features.shape[0])[:num_keep].sort()[0]
                    
                    features = features[keep_indices]
                    clusters = clusters[keep_indices]
                    
                    # For KNN, we need to be careful. 
                    # If we drop nodes, the old indices in knn_indices (which point to original indices) become invalid.
                    # Re-computing KNN on the fly is expensive.
                    # A simple approximation: 
                    # 1. Map old indices to new indices.
                    # 2. If a neighbor is dropped, replace it with self (or -1).
                    
                    # Create a mapping: old_index -> new_index (or -1 if dropped)
                    old_to_new = torch.full((knn_indices.max() + 1,), -1, dtype=torch.long)
                    old_to_new[keep_indices] = torch.arange(num_keep)
                    
                    # Filter knn_indices rows corresponding to kept nodes
                    knn_indices = knn_indices[keep_indices]
                    
                    # Map values in knn_indices
                    knn_indices = old_to_new[knn_indices]
                    
                    # Replace -1 (dropped neighbors) with the node itself (first column is usually self, but let's be safe)
                    # Actually, just replacing with the node's new index is a safe fallback for self-loop
                    # The new index of the i-th kept node is i.
                    self_indices = torch.arange(num_keep).view(-1, 1).expand_as(knn_indices)
                    mask = knn_indices == -1
                    knn_indices[mask] = self_indices[mask]

            return features, clusters, knn_indices, torch.tensor(label).long(), pid
            
        except Exception as e:
            print(f"Error loading {pid}: {e}")
            # Return dummy data to avoid crashing
            return torch.zeros(1, 1024), torch.zeros(1), torch.zeros(1, 10), torch.tensor(label).long(), pid

def collate_fn(batch):
    """
    Custom collate function to handle variable number of patches per slide.
    Since we cannot stack tensors of different sizes, we keep them as lists 
    or process them one by one in the training loop (batch_size=1).
    For simplicity here, we assume batch_size=1 or we return a list.
    """
    # Unzip the batch
    features, clusters, knn_indices, labels, pids = zip(*batch)
    
    # Stack labels (they are scalars)
    labels = torch.stack(labels)
    
    # Keep others as lists because N varies
    return list(features), list(clusters), list(knn_indices), labels, list(pids)
