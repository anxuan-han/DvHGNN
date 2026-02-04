import os
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import argparse

# --- 配置 ---
H5_KEY_COORDS = 'coordinates'
H5_KEY_KNN_INDICES = 'knn_indices'

def compute_knn(
    feature_dir: str, 
    n_neighbors: int
):
    
    h5_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
    
    # 查找所有 .h5 文件并开始处理
    for filename in tqdm(h5_files, desc=f"Computing KNN (k={n_neighbors})"):
        filepath = os.path.join(feature_dir, filename)
        
        try:
            # 1. 读取单个切片的坐标
            with h5py.File(filepath, 'r') as f:
                if H5_KEY_COORDS not in f:
                    print(f"\nSkipping {filename}: No coordinates found.")
                    continue
                coords = f[H5_KEY_COORDS][()]
            
            # 检查是否有足够的点进行 KNN
            n_samples = coords.shape[0]
            if n_samples < n_neighbors:
                # 如果点数少于 k，则使用所有点（包括自己）
                current_k = n_samples
            else:
                current_k = n_neighbors

            # 2. 执行 KNN 搜索
            # algorithm='auto' 会自动选择最快的算法 (BallTree, KDTree, or Brute-force)
            nbrs = NearestNeighbors(n_neighbors=current_k, algorithm='auto').fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            # indices 的形状是 (n_samples, current_k)
            # 第一个邻居总是自己（距离为0），后面才是其他邻居
            
            # 如果样本数少于要求的 k，用 -1 填充到固定长度
            if current_k < n_neighbors:
                padded_indices = np.full((n_samples, n_neighbors), -1, dtype=np.int32)
                padded_indices[:, :current_k] = indices
                indices = padded_indices

            # 3. 将 KNN 索引写入回 H5 文件
            with h5py.File(filepath, 'a') as f:
                # 写入前先删除旧的数据
                if H5_KEY_KNN_INDICES in f:
                    del f[H5_KEY_KNN_INDICES]
                
                # 写入索引
                f.create_dataset(H5_KEY_KNN_INDICES, data=indices, compression="gzip")
                
        except Exception as e:
            print(f"\nError processing {filename}: {e}")


# --- 主函数和 Argparser ---
def main():
    parser = argparse.ArgumentParser(
        description="Pre-calculate spatial KNN indices for WSI patches and save them back to H5 feature files."
    )
    
    # 必需参数：特征文件夹路径
    parser.add_argument(
        '--feature_dir',
        type=str,
        required=True,
        help="Path to the directory containing all WSI feature H5 files."
    )
    
    # 可选参数：邻居数量 k
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=20,
        help="Number of neighbors for KNN graph construction (default: 20)."
    )
    
    args = parser.parse_args()
    
    print("--- Starting Spatial KNN Index Generation ---")
    print(f"Feature Directory: {args.feature_dir}")
    print(f"Target Neighbors (k): {args.n_neighbors}\n")

    compute_knn(
        feature_dir=args.feature_dir,
        n_neighbors=args.n_neighbors
    )

    print("\n--- KNN Index Generation Finished Successfully ---")

if __name__ == '__main__':
    main()
