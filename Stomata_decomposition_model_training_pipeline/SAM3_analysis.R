library(reticulate)

Sys.setenv(RETICULATE_PYTHON = "managed")

py_require(
  packages = c(
    "numpy",
    "opencv-python",
    "matplotlib",
    "scikit-image",
    "ultralytics",
    "torch" ,
    "torchvision" ,
    "torchaudio" ,
    "segment-anything@git+https://github.com/facebookresearch/segment-anything.git"
  ),
  python_version = "3.12.4"
)
py_run_string("
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import os

# --- SAM configuration ---
MODEL_TYPE = 'vit_h'
CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'

MAX_RETRIES = 30
RANDOM_OFFSET_PERCENTAGE = 0.05
BORDER_EXCLUSION_MARGIN = 1

# --- Download checkpoint if missing ---
if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        f'https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_PATH}',
        CHECKPOINT_PATH
    )

# --- Load model ---
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)

# --- Folders ---
img_dir = Path('E:/Stomata_maize/all_images/all_images/crops')
overlay_dir = img_dir / 'overlays'
overlay_dir.mkdir(exist_ok=True)

image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff']]

def avoids_forbidden_regions(mask, w, h):
    if not np.any(mask):
        return False
    coords = np.argwhere(mask)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    if xmin < BORDER_EXCLUSION_MARGIN or xmax > (w-1-BORDER_EXCLUSION_MARGIN) or ymin < BORDER_EXCLUSION_MARGIN or ymax > (h-1-BORDER_EXCLUSION_MARGIN):
        return False
    return True

for image_path in image_paths:
    print(f'Processing {image_path.name} ...')
    image = cv2.imread(str(image_path))
    if image is None:
        print('Failed to load image, skipping.')
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- Set the image for the predictor (required) ---
    predictor.set_image(image_rgb)

    h, w, _ = image_rgb.shape
    total_area = h*w
    min_mask_ratio = 0.5
    max_mask_ratio = 0.8

    best_mask = None
    best_score = -1
    best_input_points = None
    best_input_labels = None

    for attempt in range(MAX_RETRIES):
        # Positive central point with random offset
        if attempt == 0:
            pos_point = np.array([[w//2, h//2]])
        else:
            offset_x = int((np.random.rand()-0.5)*2*RANDOM_OFFSET_PERCENTAGE*w)
            offset_y = int((np.random.rand()-0.5)*2*RANDOM_OFFSET_PERCENTAGE*h)
            cx = np.clip(w//2 + offset_x, 0, w-1)
            cy = np.clip(h//2 + offset_y, 0, h-1)
            pos_point = np.array([[cx, cy]])

        # Define surrounding positive points
        positive_offset_ratio = 0.2
        x_c, y_c = pos_point[0]
        p1 = [x_c, int(np.clip(y_c - positive_offset_ratio*h, 0, h-1))]
        p2 = [x_c, int(np.clip(y_c + positive_offset_ratio*h, 0, h-1))]
        p3 = [int(np.clip(x_c - positive_offset_ratio*w, 0, w-1)), y_c]
        p4 = [int(np.clip(x_c + positive_offset_ratio*w, 0, w-1)), y_c]
        points = np.array([pos_point[0], p1, p2, p3, p4])
        labels = np.ones(len(points), dtype=int)

        # Add negative points based on attempt number
        neg_points = []
        offset_neg = int(0.1*min(w,h))
        if attempt >= int(MAX_RETRIES*0.4):
            # Four border negatives
            neg_points.extend([[w//2, BORDER_EXCLUSION_MARGIN], [w//2, h-1-BORDER_EXCLUSION_MARGIN],
                               [BORDER_EXCLUSION_MARGIN, h//2], [w-1-BORDER_EXCLUSION_MARGIN, h//2]])
        if neg_points:
            points = np.concatenate([points, np.array(neg_points)])
            labels = np.concatenate([labels, np.zeros(len(neg_points),dtype=int)])

        # --- Predict masks ---
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        # Find best valid mask
        for mask, score in zip(masks, scores):
            area_ratio = mask.sum()/total_area
            if min_mask_ratio <= area_ratio <= max_mask_ratio and avoids_forbidden_regions(mask, w, h):
                if score > best_score:
                    best_mask = mask
                    best_score = score
                    best_input_points = points
                    best_input_labels = labels

    # --- Save overlay ---
    if best_mask is not None:
        overlay = image_rgb.copy()
        color = np.array([30,144,255])
        alpha = 0.6
        overlay[best_mask] = (overlay[best_mask]*(1-alpha) + color*alpha).astype(np.uint8)
        out_path = overlay_dir / f'{image_path.stem}_overlay.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save mask as numpy array
        mask_path = overlay_dir / f'{image_path.stem}_mask.npy'
        np.save(mask_path, best_mask)

        print(f'Saved overlay and mask for {image_path.name}')
    else:
        print(f'No valid mask found for {image_path.name}')
")
