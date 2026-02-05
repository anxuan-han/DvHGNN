# DvHGNN
<div align="justify">

Precise risk stratification for biochemical recurrence (BCR) in prostate cancer is currently constrained by the inability of clinical markers to resolve tumor heterogeneity and the loss of spatial context inherent in bulk molecular assays. Here, we present DvHGNN, an interpretable dual-view hypergraph neural network that predicts patient-level BCR risk directly from routine histopathology images by reconstructing the spatial syntax of the tumor ecosystem. Unlike conventional graph-based models that rely on simple pairwise connections, DvHGNN explicitly models high-order tissue architecture through two complementary lenses: a spatial view capturing emergent microenvironmental neighborhoods and a semantic view aggregating global phenotypic patterns. Across multi-institutional cohorts, DvHGNN consistently outperformed state-of-the-art multiple instance learning baselines. Beyond predictive accuracy, the model functioned as a digital discovery engine, revealing that high-risk tissue regions are characterized by distinct nuclear morphometrics and coordinated stromal remodeling. Notably, these histologically derived risk scores were intrinsically coupled with the upregulation of extracellular matrix genes, including COL1A1 and COMP, without the need for additional molecular profiling. These findings establish DvHGNN as a scalable framework that decodes the multi-scale tissue mechanisms driving disease recurrence, offering a biologically grounded approach to precision oncology.

</div>

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

## Quick Start

After preparing WSIs and metadata, the full pipeline can be executed as follows:
1. Generate tissue masks.
2. Extract patch features and construct hypergraphs.
3. Train the DvHGNN model.
4. Evaluate model performance and visualize attention maps.

## Preprocessing

The preprocessing pipeline converts whole-slide images into structured patch- and graph-level representations for downstream modeling. All preprocessing steps are performed offline prior to model training.

### 1. Tissue mask generation

This step generates tissue masks from WSIs, automatically removing background regions and common pen-mark artifacts. The resulting masks define valid tissue regions for subsequent patch extraction.

- If no pen-mark artifacts are present, this step effectively performs tissue–background separation to identify valid tissue regions for downstream processing.


```bash
python Preprocessing/process.py -pic_path "HE_slides_raw" -out_dir "Process_pics" -file_type svs -min_area 1000000 -min_hole 100000
```

- If pen-mark artifacts are present, the following script can be used to suppress pen-mark regions. Users are advised to adjust the relevant parameters based on the extent of the artifacts. Users are advised to adjust the pen-mark–related morphological parameters (e.g. dilation strength and kernel size) based on the extent of the artifacts.

```
python Preprocessing/process_mark.py \
  -pic_path "HE_slides_raw" \
  -out_dir "Process_pics" \
  -file_type svs \
  -min_area 1000000 \
  -min_hole 100000 \
  -red_dilate_iterations 1 \
  -blue_dilate_iterations 3 \
  -green_dilate_iterations 8 \
  -black_kernel_size 8 \
  -black_close_iterations 12 \
  -black_dilate_iterations 10 \
  -overwrite
```

After processing, the following three subdirectories will be created under `Process_pics`:

- `origin_pic`: original whole-slide images converted to a common image format for visualization and reference.
- `circle_pic`: intermediate visualization images illustrating detected tissue contours.
- `mask_pic`: binary tissue masks defining valid tissue regions for downstream patch extraction.


### 2. Patch extraction, stain normalization, and UNI feature extraction

This step performs patch extraction within masked tissue regions, applies stain normalization, and encodes each patch using a pretrained UNI model.  Patch-level features are extracted using a pretrained [UNI model](https://github.com/mahmoodlab/UNI), with pretrained weights specified via the `--model_dir` argument.

Patch-level features and spatial coordinates are saved in `.h5` format for downstream graph construction.

```bash
python Preprocessing/cut_norm_feature.py \
  -input_folder "HE_slides_raw" \
  -mask_folder "Process_pics/mask_pic" \
  -output_folder "Features" \
  -patch_size 256 \
  -read_size 1024 \
  -file_type svs \
  -r "ref_image.png" \
  -m macenko \
  --hf_token xxxxxx \
  --batch_size 32 \
  --model_dir uni_model_path \
  --num_workers 4 \
  -blank_threshold 0.6
```

**Output**
The following files will be generated under `Features`:
~~~
Features/*.h5
~~~
Each `.h5` file contains the following datasets:
- **`patch_indices`**: `(N,)`  
  Integer indices identifying each patch, preserving the original extraction order.
- **`features`**: `(N, 1024)`  
  Patch-level feature embeddings extracted by the pretrained UNI model.
- **`coordinates`**: `(N, 2)`  
  Spatial coordinates `(x, y)` of each extracted patch in the original WSI.





### 3. Semantic hypergraph construction

This step constructs a semantic hypergraph by clustering patch-level features based on feature similarity. Patches assigned to the same cluster form hyperedges that capture global semantic relationships across the slide.

```
python Preprocessing/kmeans.py --feature_dir "Features" --n_clusters 10 --random_state 42
```

After execution, each `.h5` file is updated with the following additional dataset:

- **`clusters`**: `(N,)`  
  Cluster assignments for each patch, indicating semantic group membership used to form hyperedges.


### 4. Spatial hypergraph construction

This step constructs a spatial hypergraph by connecting neighboring patches based on spatial proximity. Local spatial neighborhoods are modeled using k-nearest neighbors, capturing tissue architecture and local context.

```
python Preprocessing/knn.py --feature_dir /scratchdata1/users/a1978372/Anxuan/test_framwork/Features --n_neighbors 8
```

After execution, each `.h5` file is updated with the following additional dataset:

- **`knn_indices`**: `(N, k)`  
  Indices of the `k` nearest neighboring patches for each patch, based on spatial proximity.


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
python Model_training/train.py --h5_dir Features/ --csv_path data/recurrence_information.csv --output_dir Model_training/results --epochs 50 --lr 1e-4 --weight_decay 1e-3 --dropout 0.6  --train_ratio 0.8  --device cuda --seed 42
```

Alternatively, pretrained model weights trained for prostate cancer biochemical recurrence prediction can be downloaded from [Hugging Face](https://huggingface.co/ananananxuan/DvHGNN). 




## Evaluation

Model performance is evaluated using standard survival and classification metrics. All evaluation scripts operate on model predictions generated during training.

### ROC analysis 

Receiver operating characteristic (ROC) curves and area under the curve (AUC) are computed to assess discrimination performance.

```bash
python Model_evaluation/ROC_curve.py --dataset_type validation --h5_dir Features --csv_path data/recurrence_information.csv --hgnn_model_path Model_training/results/best_model.pth --output_dir Model_evaluation/results --device cuda --dropout 0.6 --seed 42 --train_ratio 0.6
```
The ROC curves and AUC results are saved under `Model_evaluation/results/validation/`.

## Visualization 

To facilitate qualitative inspection of model predictions, we provide a script to visualize attention distributions on whole-slide images. The generated attention maps highlight spatial regions that contribute most strongly to the model’s slide-level predictions.

```
python Visualization/attention_map.py --h5_dir Features --csv_path data/recurrence_information.csv --model_path Model_training/results/best_model.pth --output_dir Model_evaluation/attention_overlays --device cuda --wsi_dirs HE_slides_raw --max_patients 10 --target_pos 5 --target_neg 5
```
The generated attention map visualizations are saved under `Model_evaluation/attention_overlays`.

