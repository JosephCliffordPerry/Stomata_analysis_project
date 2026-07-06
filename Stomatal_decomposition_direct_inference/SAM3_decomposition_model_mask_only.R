library(reticulate)
library(dplyr)

find_stomatal_complex_masks_only <- function(
    img_dir,
    overlay_dir = file.path(img_dir, "masks"),
    resume = TRUE
){
  
  dir.create(overlay_dir, showWarnings = FALSE, recursive = TRUE)
  
  # -----------------------------
  # PYTHON: SAM ONLY
  # -----------------------------
  py_run_string("
import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

MODEL_TYPE = 'vit_b'
CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
MAX_RETRIES = 24
RANDOM_OFFSET_PERCENTAGE = 0.1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        'https://dl.fbaipublicfiles.com/segment_anything/' + CHECKPOINT_PATH,
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def run_sam(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    aspect_ratio = h / w
    if aspect_ratio < 1.0 or aspect_ratio > 2.8:
        return None

    predictor.set_image(image_rgb)

    masks_out = []
    scores_out = []

    for attempt in range(MAX_RETRIES):

        if attempt == 0:
            cx, cy = w // 2, h // 2
        else:
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)

        input_point = np.array([[cx, cy]])
        input_label = np.array([1], dtype=int)

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        for m, s in zip(masks, scores):
            masks_out.append(m.astype(np.uint8))
            scores_out.append(float(s))

    return {
        'masks': masks_out,
        'scores': scores_out,
        'shape': [h, w]
    }
")
  
  # -----------------------------
  # FILES
  # -----------------------------
  img_files <- list.files(
    img_dir,
    pattern="\\.(png|jpg|jpeg|tif|tiff)$",
    full.names=TRUE
  )
  
  # -----------------------------
  # RESUME SUPPORT
  # -----------------------------
  if (resume) {
    completed <- tools::file_path_sans_ext(
      list.files(overlay_dir, pattern="\\.RDS$")
    )
    img_files <- img_files[!tools::file_path_sans_ext(basename(img_files)) %in% completed]
  }
  
  results <- list()
  
  # -----------------------------
  # MAIN LOOP
  # -----------------------------
  for (img_path in img_files) {
    
    base <- tools::file_path_sans_ext(basename(img_path))
    cat("Processing:", base, "\n")
    
    res <- py$run_sam(img_path)
    if (is.null(res)) next
    
    masks <- lapply(res$masks, function(x) x == 1)
    scores <- unlist(res$scores)
    
    results[[base]] <- list(
      image = base,
      image_path = img_path,
      masks = masks,
      scores = scores,
      shape = res$shape
    )
    
    saveRDS(
      results[[base]],
      file = file.path(overlay_dir, paste0(base, ".RDS"))
    )
  }
  
  return(results)
}