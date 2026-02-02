import argparse

import os
import numpy as np
import torch
from PIL import Image
import timm
from huggingface_hub import hf_hub_download
import cv2
import h5py
import openslide
from torchvision import transforms
from normalization import reinhard_normalize, macenko_normalize
def get_uni_model(local_dir):
    hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model.eval()
    return model

def is_blank_patch(patch, threshold=0.8):
    """
    Check if a patch is blank based on multiple criteria.
    Uses a more robust approach to filter out patches with excessive white space.
    Args:
        patch: PIL Image
        threshold: Max allowed white/background pixel ratio (0.0-1.0)
    Returns:
        True if patch should be filtered out (is blank/low quality)
    """
    gray_patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    # 1. High brightness pixel ratio (background is typically 230+)
    bright_pixel_ratio = np.sum(gray_patch > 230) / gray_patch.size
    # 2. Very high brightness pixel ratio (almost white, 240+)
    very_bright_ratio = np.sum(gray_patch > 240) / gray_patch.size
    # 3. Mean brightness of the patch
    mean_brightness = np.mean(gray_patch)
    # 4. Standard deviation (low std = uniform/blank area)
    std_brightness = np.std(gray_patch)
    # 5. Edge detection
    edges = cv2.Canny(gray_patch, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    # 6. Tissue pixel ratio (darker pixels that likely represent tissue)
    tissue_pixel_ratio = np.sum(gray_patch < 200) / gray_patch.size
    # Multi-criteria filtering logic
    # Patch is blank if ANY of these conditions are met:
    # Condition 1: Too much bright background
    if bright_pixel_ratio > threshold:
        return True
    # Condition 2: Very uniform and bright (low variation + high brightness)
    if std_brightness < 15 and mean_brightness > 210:
        return True
    # Condition 3: Almost no tissue content
    if tissue_pixel_ratio < (1 - threshold):
        return True
    # Condition 4: No structure/edges
    if edge_density < 0.01:
        return True
    # Condition 5: Extremely bright overall
    if mean_brightness > 225:
        return True
    return False


def process_one_slide(slide_path, mask_path, output_h5, patch_size, read_size, blank_threshold, file_type, ref_img_path, norm_method, parallel):
    local_dir = "../assets/ckpts/uni2-h/"
    os.makedirs(local_dir, exist_ok=True)
    model = get_uni_model(local_dir)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    slide = openslide.OpenSlide(slide_path)
    mask = cv2.imread(mask_path, 0)
    w, h = slide.level_dimensions[0]
    # 自动检测magnification
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
        if avg_mpp < 0.3:
            magnification = "40x"
        elif avg_mpp < 0.6:
            magnification = "20x"
        else:
            magnification = "Unknown"
    else:
        magnification = "Unknown"
    if read_size is None:
        read_size_val = 512 if magnification == "40x" else patch_size
    else:
        read_size_val = read_size
    if ref_img_path:
        ref_img = Image.open(ref_img_path).convert('RGB')
    else:
        ref_img = None
    features, coordinates, patch_names = [], [], []
    def process_patch(x, y):
        patch = slide.read_region((x, y), 0, (read_size_val, read_size_val)).convert('RGB')
        mask_patch = mask[y:y+read_size_val, x:x+read_size_val]
        if mask_patch.shape != (read_size_val, read_size_val):
            return None
        if np.mean(mask_patch) < 10:
            return None
        if is_blank_patch(patch, blank_threshold):
            return None
        if norm_method == 'reinhard':
            patch = reinhard_normalize(patch, ref_img)
        elif norm_method == 'macenko':
            patch = macenko_normalize(patch, ref_img)
        tensor = transform(patch).unsqueeze(0)
        with torch.inference_mode():
            feature = model(tensor)
        feature = feature.squeeze().cpu().numpy()
        patch_name = f"{os.path.basename(slide_path)}_{x}_{y}"
        return feature, [x, y], patch_name
    coords = [(x, y) for y in range(0, h - read_size_val + 1, read_size_val) for x in range(0, w - read_size_val + 1, read_size_val)]
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda xy: process_patch(*xy), coords))
        for r in results:
            if r is not None:
                ftr, coord, pname = r
                features.append(ftr)
                coordinates.append(coord)
                patch_names.append(pname)
    else:
        for x, y in coords:
            r = process_patch(x, y)
            if r is not None:
                ftr, coord, pname = r
                features.append(ftr)
                coordinates.append(coord)
                patch_names.append(pname)
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('features', data=np.array(features), compression='gzip')
        f.create_dataset('coordinates', data=np.array(coordinates), compression='gzip')
        f.create_dataset('patch_name', data=np.array(patch_names, dtype='S'), compression='gzip')
        f.create_dataset('patch_size', data=np.array([patch_size]*len(features)), compression='gzip')
        f.create_dataset('read_size', data=np.array([read_size_val]*len(features)), compression='gzip')
        f.create_dataset('file_type', data=np.array([file_type.encode()]*len(features)), compression='gzip')
    print(f"Saved features to {output_h5}, total patches: {len(features)}")


def map_contours_to_level0(slide_path, contour_img_path, mask_img_path, output_folder):
    """Map contours and masks from detection level (assumed Level 3) to Level 0."""
    slide = openslide.OpenSlide(slide_path)
    level_0_dimensions = slide.level_dimensions[0]
    detection_level = min(3, slide.level_count - 1)
    contour_img = cv2.imread(contour_img_path)
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    contour_img_level0 = cv2.resize(
        contour_img, (level_0_dimensions[0], level_0_dimensions[1]), 
        interpolation=cv2.INTER_LINEAR
    )
    mask_img_level0 = cv2.resize(
        mask_img, (level_0_dimensions[0], level_0_dimensions[1]), 
        interpolation=cv2.INTER_NEAREST
    )
    os.makedirs(output_folder, exist_ok=True)
    contour_output_path = os.path.join(output_folder, "contour_level0.png")
    mask_output_path = os.path.join(output_folder, "mask_level0.png")
    cv2.imwrite(contour_output_path, contour_img_level0)
    cv2.imwrite(mask_output_path, mask_img_level0)
    print(f"Saved Level 0 contour image to: {contour_output_path}")
    print(f"Saved Level 0 mask image to: {mask_output_path}")
    slide.close()

def inspect_level0(slide_path):
    """Inspect and save the Level 0 image for checking."""
    slide = openslide.OpenSlide(slide_path)
    level_0_dimensions = slide.level_dimensions[0]
    print(f"Level 0 dimensions: {level_0_dimensions}")
    level_0_img = slide.read_region((0, 0), 0, level_0_dimensions)
    level_0_img = level_0_img.convert('RGB')
    output_path = "level0_inspection.png"
    level_0_img.save(output_path)
    print(f"Saved Level 0 image to: {output_path}")
    slide.close()


def batch_process(input_folder, mask_folder, output_folder, patch_size, read_size, blank_threshold, file_type, ref_img, norm_method, parallel):
    wsi_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(file_type)]
    for wsi_path in wsi_files:
        base = os.path.splitext(os.path.basename(wsi_path))[0]
        mask_path = os.path.join(mask_folder, base + '.png')
        output_h5 = os.path.join(output_folder, base + '.h5')
        process_one_slide(wsi_path, mask_path, output_h5, patch_size, read_size, blank_threshold, file_type, ref_img, norm_method, parallel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch cut + normalization + UNI feature extract, output h5")
    parser.add_argument('-input_folder', required=True, help='WSI folder')
    parser.add_argument('-mask_folder', required=True, help='Mask folder')
    parser.add_argument('-output_folder', required=True, help='Output folder for h5')
    parser.add_argument('-patch_size', type=int, required=True, help='Patch size')
    parser.add_argument('-read_size', type=int, default=None, help='Patch read size (auto by magnification if None)')
    parser.add_argument('-blank_threshold', type=float, required=True, help='Blank patch threshold')
    parser.add_argument('-file_type', type=str, required=True, help='WSI file type (svs/ndpi)')
    parser.add_argument('-r', dest='ref_img', type=str, required=True, help='Reference image for normalization')
    parser.add_argument('-m', dest='norm_method', type=str, required=True, help='Normalization method (macenko/reinhard)')
    parser.add_argument('-p', dest='parallel', action='store_true', help='Enable parallel processing')
    args = parser.parse_args()
    batch_process(args.input_folder, args.mask_folder, args.output_folder, args.patch_size, args.read_size, args.blank_threshold, args.file_type, args.ref_img, args.norm_method, args.parallel)
