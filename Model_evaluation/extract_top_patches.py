import os
import h5py
import torch
import numpy as np
from model import DualViewHGNN
from dataset import HGNN_Dataset
import openslide
from PIL import Image

# 配置
H5_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/integrated/features"
CSV_PATH = "/scratchdata1/users/a1978372/Anxuan/framework/folder/rec_clean_2_final.csv"
MODEL_PATH = "/scratchdata1/users/a1978372/Anxuan/framework/try_gated/results/best_model.pth"
OUTPUT_DIR = "/scratchdata1/users/a1978372/Anxuan/framework/try_gated/patch_50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 896  # patch 尺寸，和特征提取时一致（与read_size一致）
TOPK = 50

WSI_DIRS = [
    "/scratchdata1/users/a1978372/Anxuan/framework/TCGA",
    "/scratchdata1/users/a1978372/Anxuan/framework/folder/SAH",
    "/scratchdata1/users/a1978372/Anxuan/framework/folder/rec_related_slides"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_wsi_path(pid):
    extensions = ['.svs', '.ndpi', '.tiff', '.tif']
    for d in WSI_DIRS:
        if not os.path.exists(d): continue
        for ext in extensions:
            path = os.path.join(d, pid + ext)
            if os.path.exists(path): return path
            for f in os.listdir(d):
                if pid in f and f.endswith(ext):
                    return os.path.join(d, f)
    return None

def save_patch(slide, coord, save_path):
    x, y = coord
    patch = slide.read_region((int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
    # 计算空白比例（假设空白为高亮/接近白色区域）
    patch_np = np.array(patch)
    # 设定阈值，像素RGB均大于220视为空白
    blank_mask = np.all(patch_np > 220, axis=2)
    blank_ratio = np.sum(blank_mask) / (PATCH_SIZE * PATCH_SIZE)
    if blank_ratio > 0.12:
        return False  # 空白比例大于15%，跳过
    patch.save(save_path)
    return True

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = DualViewHGNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading dataset index...")
    dataset = HGNN_Dataset(CSV_PATH, H5_DIR)
    print(f"Total patients: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset.data_list[i]
        pid = item['patient_id']
        label = item['label']
        h5_path = item['h5_path']

        with h5py.File(h5_path, 'r') as f:
            features = torch.from_numpy(f['features'][:]).float().to(DEVICE)
            coords = f['coordinates'][:]
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

        with torch.no_grad():
            logits, A, _ = model(features, clusters, knn_indices)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        if pred != label:
            continue  # 只保留预测正确的slide

        attn_weights = A.cpu().numpy().flatten()
        topk_idx = np.argsort(-attn_weights)[:TOPK]  # attention最高的topK

        wsi_path = find_wsi_path(pid)
        if wsi_path is None:
            print(f"WSI not found for {pid}, skip.")
            continue
        slide = openslide.OpenSlide(wsi_path)

        # 构建分层保存路径
        label_dir = os.path.join(OUTPUT_DIR, f"label{label}")
        patient_dir = os.path.join(label_dir, f"{pid}_P{pred}")
        os.makedirs(patient_dir, exist_ok=True)
        saved_count = 0
        used_idx = set()
        attn_order = np.argsort(-attn_weights)  # attention从高到低排序的所有索引
        for idx in attn_order:
            if saved_count >= TOPK:
                break
            if idx in used_idx:
                continue
            coord = coords[idx]
            attn = attn_weights[idx]
            save_name = f"patch{saved_count+1}_attn{attn:.4f}.png"
            save_path = os.path.join(patient_dir, save_name)
            ok = save_patch(slide, coord, save_path)
            if ok:
                saved_count += 1
                used_idx.add(idx)
        print(f"Saved {saved_count} patches for {pid} (label={label}, pred={pred})")

if __name__ == "__main__":
    main()
