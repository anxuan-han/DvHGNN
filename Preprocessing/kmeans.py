import os
import h5py
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse # 导入 argparse 库

# --- 配置 ---
H5_KEY_FEATURES = 'features'        
H5_KEY_KMEANS_LABELS = 'clusters'  # 改为复数形式，匹配已有的数据集名称

def feature_kmeans(
    feature_dir: str, 
    n_clusters: int, 
    random_state: int = 42
):
    
    h5_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
    
    # 查找所有 .h5 文件并开始处理
    for filename in tqdm(h5_files, desc=f"Clustering Slides (M={n_clusters})"):
        filepath = os.path.join(feature_dir, filename)
        
        # 1. 读取单个切片的特征
        with h5py.File(filepath, 'r') as f:
            features = f[H5_KEY_FEATURES][()]
            
        # 2. 执行 K-Means 聚类
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state, 
            n_init=10, 
            verbose=0
        )
        slide_labels = kmeans.fit_predict(features) # [N_slide]

        # 3. 将聚类标签写入回 H5 文件
        with h5py.File(filepath, 'a') as f:
            # 写入前先删除旧的标签
            if H5_KEY_KMEANS_LABELS in f:
                del f[H5_KEY_KMEANS_LABELS]
                
            # 写入标签
            f.create_dataset(H5_KEY_KMEANS_LABELS, data=slide_labels, compression="gzip")


# --- 主函数和 Argparser ---
def main():
    parser = argparse.ArgumentParser(
        description="Pre-calculate local K-Means labels for WSI patches and save them back to H5 feature files."
    )
    
    # 必需参数：特征文件夹路径
    parser.add_argument(
        '--feature_dir',
        type=str,
        required=True,
        help="Path to the directory containing all WSI feature H5 files."
    )
    
    # 必需参数：聚类簇数 (超边数量 M)
    parser.add_argument(
        '--n_clusters',
        type=int,
        required=True,
        help="Number of K-Means clusters (hyperedges M) for each slide."
    )
    
    # 可选参数：随机种子
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help="Random state for K-Means initialization."
    )
    
    args = parser.parse_args()
    
    print("--- Starting Hypergraph K-Means Label Generation ---")
    print(f"Feature Directory: {args.feature_dir}")
    print(f"Target Hyperedges (M): {args.n_clusters}")
    print(f"Random State: {args.random_state}\n")



    feature_kmeans(
        feature_dir=args.feature_dir,
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )

    print("\n--- K-Means Label Generation Finished Successfully ---")

if __name__ == '__main__':
    main()