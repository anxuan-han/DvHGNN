# DvHGNN

![](https://github.com/anxuan-han/DvHGNN/blob/main/workflow.png)

## Requirements
* python>=3.8
* torch==1.13.1
* torchvision==0.14.1
* torch-geometric==2.6.1
* torch_scatter==2.1.2
* torch_cluster==1.6.3
* numpy==1.24.4
* scikit-learn==1.5.2
* scikit-image==0.24.0
* opencv-python==4.9.0.80
* openslide-python==1.4.1
* staintools==2.1.2
* pandas==2.2.3
* lifelines==0.30.0
* scipy==1.14.1
* matplotlib==3.9.3
* tqdm==4.67.1


## Dataset
The model was trained and internally validated on the TCGA-PRAD cohort and evaluated on an independent SAH cohort.

## Preprocessing

The preprocessing pipeline converts whole-slide images into structured patch- and graph-level representations for downstream modeling. All preprocessing steps are performed offline prior to model training.

### 1. Tissue mask generation

This step generates tissue masks from WSIs, automatically removing background regions and common pen-mark artifacts. The resulting masks define valid tissue regions for subsequent patch extraction.

~~~bash
python process.py \
  -pic_path "/path/to/WSI_folder" \
  -file_type svs \
  -min_area 1000000 \
  -min_hole 100000 \
  -out_dir "/path/to/out_dir"
~~~

**Output**

~~~
/path/to/out_dir/mask_pic/*.png
~~~

### 2. Patch extraction, stain normalization, and UNI feature extraction

This step performs patch extraction within masked tissue regions, applies stain normalization, and encodes each patch using a pretrained UNI model. Patch-level features and spatial coordinates are saved in `.h5` format for downstream graph construction.

~~~bash
python cut_norm_feature.py \
  -input_folder "TCGA" \
  -mask_folder "mask_pic" \
  -output_folder features \
  -patch_size 256 \
  -read_size 1024 \
  -file_type svs \
  -r "patch_0021.png" \
  -m macenko \
  --hf_token xxxxxxx \
  --batch_size 128 \
  --model_dir uni_model \
  --num_workers 8 \
  --grid_output_dir grid \
  -blank_threshold 0.6 \
  --save_raw_patches raw_patches \
  --save_normalized_patches normalized_patches
~~~

**Output**

~~~
features/*.h5
~~~

Each file contains patch-level embeddings (`features`) and spatial coordinates (`coordinates`).

### 3. Semantic hypergraph construction

This step constructs a semantic hypergraph by clustering patch-level features based on feature similarity. Patches assigned to the same cluster form hyperedges that capture global semantic relationships across the slide.

~~~bash
python kmeans.py \
  --feature_dir "TCGA" \
  --n_clusters 10 \
  --random_state 42
~~~

### 4. Spatial hypergraph construction

This step constructs a spatial hypergraph by connecting neighboring patches based on spatial proximity. Local spatial neighborhoods are modeled using k-nearest neighbors, capturing tissue architecture and local context.

~~~bash
python knn.py \
  --feature_dir "TCGA" \
  --n_neighbors 8
~~~

## Data Preparation

HGNN_Dataset requires a metadata CSV and a directory of precomputed `.h5` files.

### CSV
- `patient_id`: must match the `.h5` filename
- `recurrence`: label, cast to `int` (e.g. 0/1)

### HDF5 files
- Stored under `h5_dir`
- Named as `{patient_id}.h5`

### Datasets in each `.h5`
- `features`: shape `(N, D)`
- `clusters`: shape `(N,)`, defaults to all zeros if missing
- `knn_indices`: shape `(N, k)`, defaults to self-loop with `k=10` if missing
Here, `N` denotes the number of patches in a slide, `D` the feature dimension, and `k` the number of spatial neighbors per patch.

## Training

The model is trained using precomputed patch-level features and fixed semantic and spatial hypergraphs. Training is performed at the slide level, with one graph per sample.

```bash
python train.py \
  --h5_dir /path/to/features \
  --csv_path /path/to/labels.csv \
  --output_dir /path/to/results \
  --epochs 50 \
  --lr 0.0001 \
  --weight_decay 0.001 \
  --dropout 0.5 \
  --train_dropout 0.3
```


## Evaluation

Model performance is evaluated using standard survival and classification metrics. All evaluation scripts operate on model predictions generated during training.

### ROC analysis

Receiver operating characteristic (ROC) curves and area under the curve (AUC) are computed to assess discrimination performance.

```bash
python ROC_curve.py \
  --pred_path /path/to/predictions \
  --output_dir /path/to/results
```
### Kaplan–Meier survival analysis

Kaplan–Meier curves are used to compare survival outcomes between predicted risk groups.

```bash
python km_curve.py \
  --pred_path /path/to/predictions \
  --clinical_csv /path/to/clinical.csv \
  --output_dir /path/to/results
```

### ROC analysis

Receiver operating characteristic (ROC) curves and area under the curve (AUC) are computed to assess discrimination performance.

```bash
python ROC_curve.py \
  --pred_path /path/to/predictions \
  --output_dir /path/to/results
```
