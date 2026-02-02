import os
import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import DualViewHGNN
from dataset import HGNN_Dataset
import openslide
from PIL import Image
import cv2

# Config
H5_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/integrated/features"
CSV_PATH = "/scratchdata1/users/a1978372/Anxuan/framework/folder/rec_clean_2_final.csv"
MODEL_PATH = "/scratchdata1/users/a1978372/Anxuan/framework/try_gated/results/best_model.pth"
OUTPUT_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/try_gated/vis_results_overlay"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WSI Directories (Add your WSI paths here)
WSI_DIRS = [
    "/scratchdata1/users/a1978372/Anxuan/framework/TCGA",
    "/scratchdata1/users/a1978372/Anxuan/framework/folder/SAH",
    "/scratchdata1/users/a1978372/Anxuan/framework/folder/rec_related_slides"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_wsi_path(pid):
    # Try common extensions
    extensions = ['.svs', '.ndpi', '.tiff', '.tif']
    for d in WSI_DIRS:
        if not os.path.exists(d): continue
        for ext in extensions:
            # Try exact match
            path = os.path.join(d, pid + ext)
            if os.path.exists(path): return path
            # Try partial match (sometimes pid in filename is slightly different)
            for f in os.listdir(d):
                if pid in f and f.endswith(ext):
                    return os.path.join(d, f)
    return None

def visualize_patient(model, pid, h5_path, label, output_dir):
    print(f"Processing {pid} (Label: {label})...")
    
    # 1. Load Features & Coords
    try:
        with h5py.File(h5_path, 'r') as f:
            features = torch.from_numpy(f['features'][:]).float().to(DEVICE)
            coords = f['coordinates'][:] # (N, 2) at Level 0
            
            if 'clusters' in f:
                clusters = torch.from_numpy(f['clusters'][:]).long().to(DEVICE)
            else:
                clusters = torch.zeros(features.shape[0], dtype=torch.long).to(DEVICE)

            if 'knn_indices' in f:
                knn_indices = torch.from_numpy(f['knn_indices'][:]).long().to(DEVICE)
            else:
                N = features.shape[0]
                k = 10
                knn_indices = torch.arange(N).view(-1, 1).repeat(1, k).to(DEVICE)

        # 2. Model Inference
        model.eval()
        with torch.no_grad():
            logits, A, _ = model(features, clusters, knn_indices)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            score = probs[0, 1].item()
            
        attn_weights = A.cpu().numpy().flatten()
        # Normalize attention
        attn_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)

        # 3. Load WSI Thumbnail
        wsi_path = find_wsi_path(pid)
        if wsi_path is None:
            print(f"  -> Warning: WSI file not found for {pid}. Skipping overlay.")
            return

        slide = openslide.OpenSlide(wsi_path)
        w, h = slide.dimensions
        
        # Downsample factor for visualization (e.g., aim for ~2000px width)
        target_w = 3000
        downsample = w / target_w
        target_h = int(h / downsample)
        
        # Read region (whole slide)
        img = slide.get_thumbnail((target_w, target_h))
        img = np.array(img)[:, :, :3] # Drop alpha if exists
        
        # 4. Create Heatmap
        # Scale coordinates to thumbnail size
        coords_scaled = (coords / downsample).astype(int)
        
        # Create an empty mask
        heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        # Patch size on thumbnail (approximate)
        patch_size_scaled = int(256 / downsample)
        # 放大叠加点的半径（如原来基础上*1.5，或最小10）
        radius = max(10, int(patch_size_scaled * 0.75)) # 使点更大更明显
        
        # Draw attention on heatmap
        for i in range(len(coords_scaled)):
            x, y = coords_scaled[i]
            val = attn_norm[i]
            # Draw a filled circle
            # Note: cv2 uses (x, y)
            if x < img.shape[1] and y < img.shape[0]:
                cv2.circle(heatmap, (x + radius, y + radius), radius, float(val), -1)
            
        # Apply colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # 5. Overlay
        alpha = 0.9 # Lower alpha for more transparent heatmap
        mask = heatmap > 0.0
        mask = np.stack([mask, mask, mask], axis=2)
        
        overlay = img.copy()
        overlay[mask] = (alpha * heatmap_color[mask] + (1 - alpha) * img[mask]).astype(np.uint8)
        
        # 6. Plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1, 0.05]})


        # Subplot 1: Original（加透明度）
        axes[0].imshow(img, alpha=0.8)  # 原图也加透明度
        axes[0].set_title(f"Original WSI\n{pid}", fontsize=12)
        axes[0].axis('off')

        # Subplot 2: Overlay（加透明度）
        im = axes[1].imshow(overlay, alpha=1)  # overlay整体再加一层透明度
        status = "CORRECT" if pred == label else "WRONG"
        axes[1].set_title(f"Attention Overlay\nTrue: {label}, Pred: {pred} ({score:.4f}) - {status}", fontsize=12)
        axes[1].axis('off')

        # Subplot 3: Colorbar (legend)
        # 只显示 attention 热度的颜色映射
        from matplotlib.colors import Normalize
        import matplotlib as mpl
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('jet')
        cb1 = mpl.colorbar.ColorbarBase(axes[2], cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Attention (Normalized)', fontsize=12)
        axes[2].yaxis.set_ticks_position('right')
        # 设置colorbar半透明
        cb1.solids.set_alpha(0.5)
        fig.subplots_adjust(wspace=0.2)

        save_name = f"{pid}_L{label}_P{pred}_overlay.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved overlay to {save_path}")
        
    except Exception as e:
        print(f"  -> Error visualizing {pid}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Please train the model first.")
        return

    # Load Model
    model = DualViewHGNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load Dataset to get PIDs
    print("Loading dataset index...")
    dataset = HGNN_Dataset(CSV_PATH, H5_DIR)
    
    print(f"Total patients: {len(dataset)}")
    print(f"Saving results to: {OUTPUT_DIR}")
    
    # Strategy: Visualize a mix of Positive (Recurrence) and Negative cases
    pos_count = 0
    neg_count = 0
    target_pos = 80
    target_neg = 0
    
    for i in range(len(dataset)):
        item = dataset.data_list[i]
        pid = item['patient_id']
        label = item['label']
        h5_path = item['h5_path']
        
        if label == 1 and pos_count < target_pos:
             visualize_patient(model, pid, h5_path, label, OUTPUT_DIR)
             pos_count += 1
        elif label == 0 and neg_count < target_neg:
             visualize_patient(model, pid, h5_path, label, OUTPUT_DIR)
             neg_count += 1
             
        if pos_count >= target_pos and neg_count >= target_neg:
            break
            
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
