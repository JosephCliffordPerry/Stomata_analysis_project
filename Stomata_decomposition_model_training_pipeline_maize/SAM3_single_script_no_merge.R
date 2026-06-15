#!/usr/bin/env Rscript

# -------------------------------
# ENV SETUP
# -------------------------------

Sys.unsetenv("PYTHONHOME")
Sys.unsetenv("PYTHONPATH")

library(reticulate)
library(dplyr)

use_python("/home/jp19193/uv-stomata-env/bin/python", required = TRUE)

# -------------------------------
# PYTHON BACKEND (LOAD ONCE)
# -------------------------------

py_run_string("
import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

MODEL_TYPE = 'vit_b'
CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_RETRIES_COMPLEX = 10
MAX_RETRIES_COMP = 30
JITTER = 0.05

# -------------------------
# LOAD MODEL
# -------------------------

if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        'https://dl.fbaipublicfiles.com/segment_anything/' + CHECKPOINT_PATH,
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# -------------------------
# DEBUG HOOKS
# -------------------------

print('SAM loaded OK')

# -------------------------
# HELPERS
# -------------------------

def jitter_point(x, y, w, h):
    dx = (np.random.rand() - 0.5) * 2 * JITTER * w
    dy = (np.random.rand() - 0.5) * 2 * JITTER * h
    return (
        int(np.clip(x + dx, 0, w - 1)),
        int(np.clip(y + dy, 0, h - 1))
    )

# -------------------------
# COMPLEX SEGMENTATION
# -------------------------

def segment_complex(image, min_ratio, max_ratio):

    h, w = image.shape[:2]
    total_area = h * w

    predictor.set_image(image)

    cx, cy = w // 2, h // 2

    valid = []

    for i in range(MAX_RETRIES_COMPLEX):

        if i == 0:
            px, py = cx, cy
        else:
            px, py = jitter_point(cx, cy, w, h)

        masks, scores, logits = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        print('complex retry', i, 'masks:', len(masks))

        for m, s in zip(masks, scores):

            r = m.sum() / total_area

            if min_ratio <= r <= max_ratio:

                valid.append({
                    'mask': m.astype(np.uint8),
                    'score': float(s),
                    'area_ratio': float(r),
                    'retry': int(i)
                })

    print('complex valid masks:', len(valid))

    return valid

# -------------------------
# COMPANION SEGMENTATION
# -------------------------

def segment_companion(image, pts, labs, min_ratio, max_ratio):

    h, w = image.shape[:2]
    total_area = h * w

    predictor.set_image(image)

    valid = []

    for i in range(MAX_RETRIES_COMP):

        jittered = pts.copy()

        if i > 0:
            jitter = np.random.uniform(-JITTER, JITTER, size=jittered.shape)
            jittered[:, 0] += jitter[:, 0] * w
            jittered[:, 1] += jitter[:, 1] * h

        masks, scores, logits = predictor.predict(
            point_coords=jittered,
            point_labels=labs,
            multimask_output=True
        )

        for m, s in zip(masks, scores):

            r = m.sum() / total_area

            if min_ratio <= r <= max_ratio:

                valid.append({
                    'mask': m.astype(np.uint8),
                    'score': float(s),
                    'area_ratio': float(r),
                    'retry': int(i)
                })

    return valid

# -------------------------
# PIPELINE
# -------------------------

def run_pipeline(image_paths, min_ratio=0.4, max_ratio=0.8):

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    out = []

    for path in image_paths:

        print('processing:', path)

        img = cv2.imread(path)

        if img is None:
            print('FAILED READ:', path)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base = os.path.splitext(os.path.basename(path))[0]

        complex_masks = segment_complex(img, min_ratio, max_ratio)

        if len(complex_masks) == 0:
            print('NO COMPLEX MASKS:', base)
            continue

        image_results = []

        for complex_idx, complex_result in enumerate(complex_masks):

            cmask = complex_result['mask']

            pts = np.column_stack(np.where(cmask > 0))

            if len(pts) < 2:
                continue

            p1 = pts[0]
            p2 = pts[-1]

            mid = (p1 + p2) / 2

            pos = np.array([
                mid + [5, 0],
                mid + [-5, 0],
                mid + [0, 5]
            ])

            neg = np.array([
                mid + [10, 10],
                mid + [-10, -10]
            ])

            pts1 = np.vstack([pos, neg]).astype(np.float32)

            lab1 = np.array([1]*len(pos) + [0]*len(neg)).astype(np.float32)

            companion_masks = segment_companion(
                img,
                pts1,
                lab1,
                0.05,
                0.2
            )

            image_results.append({
                'complex_index': int(complex_idx),
                'complex_mask': complex_result['mask'],
                'complex_score': complex_result['score'],
                'complex_area_ratio': complex_result['area_ratio'],
                'complex_retry': complex_result['retry'],
                'companion1': companion_masks
            })

        out.append({
            'image': base,
            'detections': image_results
        })

    return out
")

# -------------------------------
# TASK SPLITTING
# -------------------------------

img_dir <- "/home/jp19193/R_stomata_pipeline/crops"

files <- list.files(
  img_dir,
  pattern="\\.(png|jpg|jpeg|tif|tiff)$",
  full.names=TRUE
)

stopifnot(length(files) > 0)

task_id <- as.integer(Sys.getenv("SGE_TASK_ID", "1"))
n_tasks <- as.integer(Sys.getenv("SGE_TASK_LAST", "1"))

chunks <- split(files, (seq_along(files)-1) %% n_tasks)
batch <- chunks[[task_id]]

cat("Task", task_id, "processing", length(batch), "images\n")

# -------------------------------
# OUTPUT STRUCTURE
# -------------------------------

output_root <- file.path(img_dir, "inference_outputs")
dir.create(output_root, showWarnings = FALSE, recursive = TRUE)

task_dir <- file.path(output_root, paste0("task_", task_id))
dir.create(task_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------------
# SAVE CHUNKS
# -------------------------------

chunk_size <- 3

save_chunk_array <- function(i_start, i_end, batch, task_dir) {
  
  sub_batch <- batch[i_start:i_end]
  
  cat("Processing chunk:", i_start, "to", i_end, "\n")
  
  res <- py$run_pipeline(sub_batch)
  
  if (length(res) == 0) {
    cat("WARNING: empty batch\n")
    return(NULL)
  }
  
  out_list <- list()
  
  for (img_i in seq_along(res)) {
    
    img <- res[[img_i]]
    img_name <- img$image
    
    out_list[[img_name]] <- list()
    
    for (det in img$detections) {
      
      idx <- as.character(det$complex_index)
      
      out_list[[img_name]][[idx]] <- list(
        complex_mask = det$complex_mask,
        complex_score = det$complex_score,
        complex_area_ratio = det$complex_area_ratio,
        complex_retry = det$complex_retry,
        companion1 = det$companion1
      )
    }
  }
  
  save_file <- file.path(
    task_dir,
    paste0("inference_", sprintf("%06d", i_start), "_", sprintf("%06d", i_end), ".RDA")
  )
  
  save(out_list, file = save_file)
  
  cat("Saved:", save_file, "\n")
}

# -------------------------------
# RUN
# -------------------------------

total <- length(batch)

for (i in seq(1, total, by = chunk_size)) {
  
  end <- min(i + chunk_size - 1, total)
  
  save_chunk_array(i, end, batch, task_dir)
}

cat("DONE\n")