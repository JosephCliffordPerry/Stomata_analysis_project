library(reticulate)

find_stomatal_complex <- function(
    img_dir,
    overlay_dir = file.path(img_dir,"overlays"),
    min_ratio = 0.5,
    max_ratio = 0.8
){
  
  dir.create(overlay_dir, showWarnings = FALSE, recursive = TRUE)
  
  py_run_string("
import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

MODEL_TYPE = 'vit_h'
CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
MAX_RETRIES = 10
RANDOM_OFFSET_PERCENTAGE = 0.05

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        f'https://dl.fbaipublicfiles.com/segment_anything/' + CHECKPOINT_PATH,
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)


def run_segment_collect(image_path, min_ratio, max_ratio):

    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    total_area = h * w

    predictor.set_image(image_rgb)

    valid_masks = []

    for attempt in range(MAX_RETRIES):

        if attempt == 0:
            cx, cy = w//2, h//2
        else:
            cx = int(np.clip(w//2 + (np.random.rand()-0.5)*2*RANDOM_OFFSET_PERCENTAGE*w, 0, w-1))
            cy = int(np.clip(h//2 + (np.random.rand()-0.5)*2*RANDOM_OFFSET_PERCENTAGE*h, 0, h-1))

        points = np.array([[cx, cy]])
        labels = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        for mask in masks:
            ratio = mask.sum() / total_area
            if min_ratio <= ratio <= max_ratio:
                valid_masks.append(mask.astype(np.uint8))

    if len(valid_masks) == 0:
        return None

    stack = np.stack(valid_masks)
    vote = np.sum(stack, axis=0)
    density = vote / len(valid_masks)

    thresholds = np.unique(density)
    thresholds = np.sort(thresholds)[::-1]

    best_mask = None

    for t in thresholds:

        candidate = (density >= t).astype(np.uint8)

        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8
        )

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / total_area

            if min_ratio <= ratio <= max_ratio:
                best_mask = (labels_cc == i).astype(np.uint8)
                break

        if best_mask is not None:
            break

    if best_mask is None:
        best_mask = (vote >= (len(valid_masks)/2)).astype(np.uint8)

    return best_mask


def run_batch(image_paths, overlay_dir, min_ratio, max_ratio):

    os.makedirs(overlay_dir, exist_ok=True)

    for image_path in image_paths:

        base = os.path.splitext(os.path.basename(image_path))[0]
        print(f'Processing: {base}')

        mask = run_segment_collect(image_path, min_ratio, max_ratio)

        if mask is None:
            continue

        out_path = os.path.join(overlay_dir, base + '_complex_mask.npy')
        np.save(out_path, mask)

    print('Processing complete')
  ")
  
  img_files <- list.files(
    img_dir,
    pattern="\\.(png|jpg|jpeg|tif|tiff)$",
    full.names=TRUE
  )
  
  py$run_batch(img_files, normalizePath(overlay_dir, winslash = "/"), min_ratio, max_ratio)
}