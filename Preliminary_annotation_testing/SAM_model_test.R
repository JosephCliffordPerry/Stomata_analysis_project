# ===============================
# R + reticulate Setup for CPU-only PyTorch
# ===============================

# 1️⃣ Force CPU-only (ignore any GPU)
Sys.setenv(CUDA_VISIBLE_DEVICES = "")  # Prevent PyTorch from loading GPU/CUDA DLLs
Sys.setenv(RETICULATE_PYTHON = "managed")  # Use a managed Python environment

# 2️⃣ Create / reuse ephemeral Python environment with required packages
reticulate::py_require(
  packages = c(
    "numpy",
    "opencv-python",
    "matplotlib",
    "scikit-image",
    "torch",          # CPU-only PyTorch will be automatically installed
    "torchvision",
    "segment-anything"
  ),
  python_version = "3.12.4"  # Use your current Python 3.12
)

# 3️⃣ Verify PyTorch installation in R
reticulate::py_run_string("
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
device = 'cpu'  # Force CPU
")

# 4️⃣ Run your segmentation function safely
reticulate::py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure, morphology
from skimage.filters import threshold_local
from segment_anything import sam_model_registry, SamPredictor
import torch

device = 'cpu'  # Force CPU

def segment_cells_with_sam(filename='plant_cells_20x.png',
                           min_cell_area=100,
                           max_cell_area=5000,
                           invert=False,
                           block_size_adaptive=25,
                           edge_weight=1.0,
                           erosion_radius=2,
                           sam_checkpoint='sam_vit_h_4b8939.pth',
                           model_type='vit_h'):

    # 1️⃣ Load image
    img_bgr = cv2.imread(filename)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)

    # 2️⃣ CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)

    # 3️⃣ Gaussian blur
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    # 4️⃣ Sobel edges
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5️⃣ Combine blurred + edges
    combined = cv2.addWeighted(blur, 0.8, sobel_mag, edge_weight, 0)

    # 6️⃣ Adaptive threshold
    adaptive_thresh = threshold_local(combined, int(block_size_adaptive), offset=-0.02)
    init_mask = combined > adaptive_thresh

    # 7️⃣ Morphological erosion
    selem = morphology.disk(int(erosion_radius))
    eroded_mask = morphology.erosion(init_mask, selem)

    # 8️⃣ Label and filter
    labeled = measure.label(eroded_mask)
    props = measure.regionprops_table(labeled, properties=['label','area','centroid'])
    for i, area in enumerate(props['area']):
        if area < min_cell_area or area > max_cell_area:
            labeled[labeled == props['label'][i]] = 0

    labeled_filtered = measure.label(labeled > 0)
    centroids = np.array([p.centroid for p in measure.regionprops(labeled_filtered)])

    # 9️⃣ Load SAM
    print(f'Loading SAM model ({model_type}) on {device}...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 🔟 Run SAM on each centroid
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    sam_masks = []
    for c in centroids:
        input_point = np.array([[c[1], c[0]]])  # (x,y)
        input_label = np.array([1])
        mask, score, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        sam_masks.append(mask)

    # 1️⃣1️⃣ Combine SAM masks
    if len(sam_masks) > 0:
        combined_sam = np.sum(np.stack(sam_masks), axis=0) > 0
    else:
        combined_sam = np.zeros_like(gray, dtype=bool)

    # 1️⃣2️⃣ Overlay for visualization
    labeled_after_rgb = color.label2rgb(labeled_filtered, bg_label=0, bg_color=(0,0,0))
    overlay_initial = (0.7 * (gray[..., None]/255.0) + 0.6 * labeled_after_rgb).clip(0, 1)
    overlay_sam = (0.7 * (gray[..., None]/255.0) + 0.6 * np.stack([combined_sam]*3, axis=-1)).clip(0, 1)

    # 1️⃣3️⃣ Display results
    fig, axs = plt.subplots(2,3, figsize=(16,9))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original')
    axs[0,1].imshow(init_mask, cmap='gray'); axs[0,1].set_title('Initial Mask')
    axs[0,2].imshow(labeled_after_rgb); axs[0,2].set_title('Labeled Filtered')
    axs[1,0].imshow(overlay_initial); axs[1,0].set_title('Filtered Overlay (Pre-SAM)')
    axs[1,1].imshow(combined_sam, cmap='gray'); axs[1,1].set_title('SAM Combined Mask')
    axs[1,2].imshow(overlay_sam); axs[1,2].set_title('SAM Overlay on Original')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()

    print(f'Objects after filtering: {len(centroids)}')
    print(f'SAM masks generated: {len(sam_masks)}')

    return combined_sam
")


segment_cells_with_sam <- function(filename,
                                   min_cell_area=100,
                                   max_cell_area=5000,
                                   invert=FALSE,
                                   block_size_adaptive=25,
                                   edge_weight=1.0,
                                   erosion_radius=2,
                                   sam_checkpoint="D:/models/sam_vit_h_4b8939.pth",
                                   model_type="vit_h") {
  py$segment_cells_with_sam(
    filename = r_to_py(as.character(filename)),
    min_cell_area = r_to_py(as.integer(min_cell_area)),
    max_cell_area = r_to_py(as.integer(max_cell_area)),
    invert = r_to_py(as.logical(invert)),
    block_size_adaptive = r_to_py(as.integer(block_size_adaptive)),
    edge_weight = r_to_py(as.numeric(edge_weight)),
    erosion_radius = r_to_py(as.integer(erosion_radius)),
    sam_checkpoint = r_to_py(as.character(sam_checkpoint)),
    model_type = r_to_py(as.character(model_type))
  )
}

seg_sam <- segment_cells_with_sam(
  filename = "D:/stomata/training_tifs_8bit/A_T2R3_Ab_15X_frame_0001.tif",
  min_cell_area = 50,
  max_cell_area = 5000,
  invert = TRUE,
  block_size_adaptive = 25,
  edge_weight = 1.0,
  erosion_radius = 2,
  sam_checkpoint = "D:/models/sam_vit_h_4b8939.pth",
  model_type = "vit_h"
)
