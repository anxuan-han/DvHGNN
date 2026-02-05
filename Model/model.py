import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing torch_scatter, else define a simple fallback
try:
    from torch_scatter import scatter_mean
except ImportError:
    print("torch_scatter not found, using fallback implementation.")
    def scatter_mean(src, index, dim=0, dim_size=None):
        # Simple fallback for 2D src and 1D index
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        count = torch.zeros((dim_size, 1), device=src.device)
        
        out.index_add_(0, index, src)
        count.index_add_(0, index, torch.ones((src.size(0), 1), device=src.device))
        
        return out / (count.clamp(min=1e-6))

class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(HypergraphConv, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H_edge_index):
        """
        x: Node features (N, in_channels)
        H_edge_index: (2, E_total) Sparse representation of incidence matrix H.
                      row 0: node indices
                      row 1: hyperedge indices
        """
        node_indices, edge_indices = H_edge_index
        
        # 1. Node to Edge aggregation (Aggregate node features to hyperedges)
        # shape: (Num_Hyperedges, in_channels)
        edge_features = scatter_mean(x[node_indices], edge_indices, dim=0)
        
        # 2. Edge to Node aggregation (Aggregate hyperedge features back to nodes)
        # shape: (Num_Nodes, in_channels)
        x_new = scatter_mean(edge_features[edge_indices], node_indices, dim=0, dim_size=x.size(0))
        
        # 3. Transform
        x_new = self.W(x_new)
        x_new = self.layer_norm(x_new)
        x_new = F.relu(x_new)
        x_new = self.dropout(x_new)
        
        return x_new

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=256, D=128, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

class DualViewHGNN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2, dropout=0.5):
        super(DualViewHGNN, self).__init__()
        
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Semantic View - 2 Layers
        self.sem_hgnn1 = HypergraphConv(hidden_dim, hidden_dim, dropout=dropout)
        self.sem_hgnn2 = HypergraphConv(hidden_dim, hidden_dim, dropout=dropout)
        
        # Spatial View - 2 Layers
        self.spa_hgnn1 = HypergraphConv(hidden_dim, hidden_dim, dropout=dropout)
        self.spa_hgnn2 = HypergraphConv(hidden_dim, hidden_dim, dropout=dropout)
        
        # View Attention Fusion
        self.view_attn = nn.Linear(hidden_dim, 1)
        
        # Pooling (Gated Attention)
        self.pool_attn = Attn_Net_Gated(L=hidden_dim, D=128, dropout=True, n_classes=1)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def build_semantic_hypergraph(self, clusters, num_nodes):
        # clusters: (N,)
        # Hyperedges are the clusters themselves.
        node_indices = torch.arange(num_nodes, device=clusters.device)
        edge_indices = clusters
        
        return torch.stack([node_indices, edge_indices], dim=0)

    def build_spatial_hypergraph(self, knn_indices, num_nodes):
        # knn_indices: (N, k)
        # Each node defines a hyperedge containing its neighbors.
        N, k = knn_indices.shape
        
        # hyperedge_indices: 0,0,..,0, 1,1,..,1, ... (repeat k times each)
        hyperedge_indices = torch.arange(N, device=knn_indices.device).view(-1, 1).repeat(1, k).flatten()
        
        # node_indices: knn_indices flattened
        node_indices = knn_indices.flatten()
        
        # Filter out -1 (padding) if any
        mask = node_indices >= 0
        node_indices = node_indices[mask]
        hyperedge_indices = hyperedge_indices[mask]
        
        return torch.stack([node_indices, hyperedge_indices], dim=0)

    def forward(self, x, clusters, knn_indices):
        # x: (N, 1024)
        # clusters: (N,)
        # knn_indices: (N, k)
        
        N = x.size(0)
        h = self.fc_in(x)
        
        # --- Semantic View ---
        sem_edge_index = self.build_semantic_hypergraph(clusters, N)
        h_sem = self.sem_hgnn1(h, sem_edge_index)
        h_sem = h_sem + h # Residual
        h_sem = self.sem_hgnn2(h_sem, sem_edge_index) # Layer 2
        
        # --- Spatial View ---
        spa_edge_index = self.build_spatial_hypergraph(knn_indices, N)
        h_spa = self.spa_hgnn1(h, spa_edge_index)
        h_spa = h_spa + h # Residual
        h_spa = self.spa_hgnn2(h_spa, spa_edge_index) # Layer 2
        
        # --- Fusion ---
        # Stack: (N, 2, hidden_dim)
        h_stack = torch.stack([h_sem, h_spa], dim=1)
        # Attn scores: (N, 2, 1)
        attn_scores = self.view_attn(h_stack) 
        attn_weights = F.softmax(attn_scores, dim=1)
        
        h_fused = torch.sum(h_stack * attn_weights, dim=1) # (N, hidden_dim)
        
        # --- Pooling ---
        A = self.pool_attn(h_fused) # (N, 1)
        A = torch.softmax(A, dim=0)
        
        slide_rep = torch.mm(A.t(), h_fused) # (1, hidden_dim)
        
        # --- Classification ---
        logits = self.classifier(slide_rep) # (1, num_classes)
        
        return logits, A, h_fused
