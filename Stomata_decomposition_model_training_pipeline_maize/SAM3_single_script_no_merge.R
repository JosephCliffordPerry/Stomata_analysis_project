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
")
py_run_string("
MODEL_TYPE = 'vit_b'
CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_RETRIES_COMPLEX = 10
MAX_RETRIES_COMP = 30
JITTER = 0.05




if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        'https://dl.fbaipublicfiles.com/segment_anything/' + CHECKPOINT_PATH,
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
")

# -------------------------
# helpers
# -------------------------

def smooth(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def jitter_point(x, y, w, h):
    dx = (np.random.rand() - 0.5) * 2 * JITTER * w
    dy = (np.random.rand() - 0.5) * 2 * JITTER * h
    return (
        int(np.clip(x + dx, 0, w - 1)),
        int(np.clip(y + dy, 0, h - 1))
    )


# -------------------------
# COMPLEX SEGMENTATION (RESTORED)
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

        for m in masks:
            r = m.sum() / total_area
            if min_ratio <= r <= max_ratio:
                valid.append(m.astype(np.uint8))

    if len(valid) == 0:
        return None

    stack = np.stack(valid)
    vote = stack.mean(axis=0)

    best = (vote >= 0.5).astype(np.uint8)
    return smooth(best)


# -------------------------
# COMPANION SEGMENTATION (RESTORED)
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

        for m in masks:
            r = m.sum() / total_area
            if min_ratio <= r <= max_ratio:
                valid.append(m.astype(np.uint8))

    if len(valid) == 0:
        return None

    stack = np.stack(valid)
    vote = stack.mean(axis=0)

    return smooth((vote >= 0.5).astype(np.uint8))


# -------------------------
# PIPELINE (ARRAY OUTPUT)
# -------------------------

def run_pipeline(image_paths, min_ratio=0.4, max_ratio=0.8):

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    out = []

    for path in image_paths:

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base = os.path.splitext(os.path.basename(path))[0]

        retry_outputs = []

        cmask = segment_complex(img, min_ratio, max_ratio)
        if cmask is None:
            continue

        h, w = cmask.shape

        pts = np.column_stack(np.where(cmask > 0))
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

        # ---- multiple companion retries (explicit structure preserved)
        for r in range(MAX_RETRIES_COMP):

            c1 = segment_companion(img, pts1, lab1, 0.05, 0.2)

            retry_outputs.append({
                'complex': cmask.astype(np.uint8),
                'companion1': None if c1 is None else c1.astype(np.uint8)
            })

        out.append({
            'image': base,
            'retries': retry_outputs
        })

    return out
")

# -------------------------------
# TASK SPLITTING
# -------------------------------

img_dir <- "/home/jp19193/R_stomata_pipeline/crops"

files <- normalizePath(
  list.files(
    img_dir,
    pattern="\\.(png|jpg|jpeg|tif|tiff)$",
    full.names=TRUE
  ),
  winslash="/",
  mustWork=TRUE
)

stopifnot(all(file.exists(files)))

task_id <- as.integer(Sys.getenv("SGE_TASK_ID", "1"))
n_tasks <- as.integer(Sys.getenv("SGE_TASK_LAST", "1"))

chunks <- split(files, (seq_along(files)-1) %% n_tasks)
batch <- chunks[[task_id]]

cat("Task", task_id, "processing", length(batch), "images\n")

# -------------------------------
# OUTPUT STRUCTURE
# -------------------------------

img_dir <- "/home/jp19193/R_stomata_pipeline/crops"

output_root <- file.path(img_dir, "inference_outputs")
dir.create(output_root, showWarnings = FALSE, recursive = TRUE)

task_dir <- file.path(output_root, paste0("task_", task_id))
dir.create(task_dir, showWarnings = FALSE, recursive = TRUE)

files <- normalizePath(
  list.files(
    img_dir,
    pattern="\\.(png|jpg|jpeg|tif|tiff)$",
    full.names=TRUE
  ),
  winslash="/",
  mustWork=TRUE
)

stopifnot(all(file.exists(files)))

task_id <- as.integer(Sys.getenv("SGE_TASK_ID", "1"))
n_tasks <- as.integer(Sys.getenv("SGE_TASK_LAST", "1"))

chunks <- split(files, (seq_along(files)-1) %% n_tasks)
batch <- chunks[[task_id]]

cat("Task", task_id, "processing", length(batch), "images\n")

# -------------------------------
# SAVE INFERENCE CHUNKS (NO MERGING)
# -------------------------------

chunk_size <- 10

save_chunk_array <- function(i_start, i_end, batch, task_dir) {
  
  sub_batch <- batch[i_start:i_end]
  
  cat("Processing chunk:", i_start, "to", i_end, "\n")
  
  res <- py$run_pipeline(as.list(sub_batch))
  
  if (length(res) == 0) return(NULL)
  
  out_list <- list()
  
  for (img_i in seq_along(res)) {
    
    img <- res[[img_i]]
    img_name <- img$image
    retries <- img$retries
    
    out_list[[img_name]] <- list()
    
    for (r in seq_along(retries)) {
      
      retry <- retries[[r]]
      
      out_list[[img_name]][[as.character(r)]] <- list(
        complex = retry$complex,
        companion1 = retry$companion1
      )
    }
  }
  
  save_file <- file.path(
    task_dir,
    paste0(
      "inference_",
      sprintf("%06d", i_start),
      "_",
      sprintf("%06d", i_end),
      ".RDA"
    )
  )
  
  save(out_list, file = save_file)
  
  cat("Saved:", save_file, "\n")
}