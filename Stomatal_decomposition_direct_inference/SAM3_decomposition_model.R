#you may need to delete and redownload the sam version as it is easily corrupted 
library(reticulate)
library(sf)
library(dplyr)
library(terra)
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

find_stomatal_complex <- function(
    img_dir,
    overlay_dir = file.path(img_dir, "overlays"),
    min_ratio = 0.40,
    max_ratio = 0.8,
    min_sam_score = 0.85,
    edge_tol_px = 2,
    debug = TRUE,
    contour_refine = FALSE,
    resume = TRUE,
    retry_failed = FALSE
){
  
  dir.create(overlay_dir, showWarnings = FALSE, recursive = TRUE)
  failed_file <- file.path(
    overlay_dir,
    "failed_inferences.txt"
  )
  
  failed_images <- character(0)
  
  if (file.exists(failed_file)) {
    failed_images <- readLines(
      failed_file,
      warn = FALSE
    )
  }
  # ---------------------------------------------------------
  # PYTHON: ONLY SAM INFERENCE (NO FILTERING, NO CONTOURS)
  # ---------------------------------------------------------
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
    # -----------------------------
    # ASPECT RATIO FILTER
    # -----------------------------
    aspect_ratio = h / w
    if aspect_ratio < 1.0 or aspect_ratio > 2.8:
        return None
    predictor.set_image(image_rgb)

    masks_out = []
    scores_out = []

    no_neg_points_end = int(MAX_RETRIES * 0.40)
    ul_lr_neg_points_end = no_neg_points_end + int(MAX_RETRIES * 0.30)

    offset_ratio_neg_points = 0.10

    for attempt in range(MAX_RETRIES):

        # ----------------------------------------------------
        # Positive point
        # ----------------------------------------------------

        if attempt == 0:

            current_offset_x = 0
            current_offset_y = 0

            current_input_point = np.array([[w // 2, h // 2]])

        else:

            offset_limit_x = w * RANDOM_OFFSET_PERCENTAGE
            offset_limit_y = h * RANDOM_OFFSET_PERCENTAGE

            current_offset_x = np.random.uniform(
                -offset_limit_x,
                offset_limit_x
            )

            current_offset_y = np.random.uniform(
                -offset_limit_y,
                offset_limit_y
            )

            cx = int(np.clip(
                w // 2 + current_offset_x,
                0,
                w - 1
            ))

            cy = int(np.clip(
                h // 2 + current_offset_y,
                0,
                h - 1
            ))

            current_input_point = np.array([[cx, cy]])

        # ----------------------------------------------------
        # Negative points
        # ----------------------------------------------------

        input_points = current_input_point
        input_labels = np.array([1], dtype=int)

        neg_points = []

        xoff = int(w * offset_ratio_neg_points)
        yoff = int(h * offset_ratio_neg_points)

        if attempt < no_neg_points_end:

            # 40%: no negatives
            pass

        elif attempt < ul_lr_neg_points_end:

            # 30%: UL + LR
            neg_points = [
                [xoff, yoff],
                [w - xoff, h - yoff]
            ]

        else:

            # 30%: UR + LL
            neg_points = [
                [w - xoff, yoff],
                [xoff, h - yoff]
            ]

        if len(neg_points) > 0:

            neg_points = np.asarray(
                neg_points,
                dtype=float
            )

            # Apply the same jitter as the positive point
            neg_points[:, 0] += current_offset_x
            neg_points[:, 1] += current_offset_y

            neg_points[:, 0] = np.clip(
                neg_points[:, 0],
                0,
                w - 1
            )

            neg_points[:, 1] = np.clip(
                neg_points[:, 1],
                0,
                h - 1
            )

            neg_points = neg_points.astype(int)

            input_points = np.vstack([
                current_input_point,
                neg_points
            ])

            input_labels = np.concatenate([
                np.array([1], dtype=int),
                np.zeros(len(neg_points), dtype=int)
            ])

        # ----------------------------------------------------
        # SAM prediction
        # ----------------------------------------------------

        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
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
  if (resume) {
    
    completed <- tools::file_path_sans_ext(
      list.files(
        overlay_dir,
        pattern = "\\.RDS$"
      )
    )
    
    if (length(completed) > 0)
      cat("Found", length(completed), "completed images.\n")
  }
  img_files <- list.files(
    img_dir,
    pattern="\\.(png|jpg|jpeg|tif|tiff)$",
    full.names=TRUE
  )
  if (resume) {
    
    bases <- tools::file_path_sans_ext(basename(img_files))
    if (retry_failed) {
      
      keep <- !(bases %in% completed)
      
      cat(
        "Retrying failed detections.\n"
      )
      
    } else {
      
      keep <- !(
        bases %in% completed |
          bases %in% failed_images
      )
      
    }
    
    img_files <- img_files[keep]
    
    cat(
      length(completed),
      "completed,",
      length(failed_images),
      "failed,",
      sum(keep),
      "remaining.\n"
    )
  }
  all_polygons <- list()
  
  # ---------------------------------------------------------
  # R FILTERING + CONSENSUS
  # ---------------------------------------------------------
  mask_touches_edge <- function(mask, tol_px){
    
    h <- nrow(mask)
    w <- ncol(mask)
    
    tol_px <- min(tol_px, h, w)
    
    if (tol_px <= 0) {
      return(
        any(mask[1, ]) ||
          any(mask[h, ]) ||
          any(mask[, 1]) ||
          any(mask[, w])
      )
    }
    
    top    <- mask[1:tol_px, , drop = FALSE]
    bottom <- mask[(h - tol_px + 1):h, , drop = FALSE]
    left   <- mask[, 1:tol_px, drop = FALSE]
    right  <- mask[, (w - tol_px + 1):w, drop = FALSE]
    
    any(top) || any(bottom) || any(left) || any(right)
  }
  
  for(i in seq_along(img_files)){
    
    img_path <- img_files[i]
    base <- tools::file_path_sans_ext(basename(img_path))
    
    cat("Processing:", base, "\n")
    
    tryCatch({
      
      res <- py$run_sam(img_path)
      
      if(is.null(res))
        stop("SAM returned NULL")
    
    masks <- lapply(res$masks, function(x) x == 1)
    scores <- unlist(res$scores)
    
    h <- res$shape[[1]]
    w <- res$shape[[2]]
    total_area <- h * w
    
    # -----------------------------
    # FILTER IN R
    # -----------------------------
    valid_masks <- list()
    
    for(j in seq_along(masks)){
      
      m <- masks[[j]]
      
      if(scores[j] < min_sam_score) next
      if(mask_touches_edge(m, edge_tol_px)) next
      
      ratio <- (sum(m) / total_area)
      if(ratio < min_ratio || ratio > max_ratio) next
      
      valid_masks[[length(valid_masks) + 1]] <- m
    }
    
    if(length(valid_masks) == 0) next
    
    # -----------------------------
    # CONSENSUS
    # -----------------------------
    stack <- simplify2array(valid_masks)
    vote <- apply(stack, c(1,2), sum)
    density <- vote / length(valid_masks)
    
    best_mask <- (density >= 0.5)
    
    sf_list <- list()
    k <- 1
    
    
    mask_to_poly <- function(mask){
      
      r <- rast(matrix(as.integer(mask),
                       nrow = nrow(mask),
                       ncol = ncol(mask)))
      
      p <- as.polygons(
        r,
        dissolve = TRUE,
        values = TRUE
      )
      
      p <- p[p$lyr.1 == 1, ]
      
      if(nrow(p) == 0)
        return(NULL)
      
      st_as_sf(p)
    }
    # ----------------------------------
    # consensus polygon
    # ----------------------------------
    consensus_sf <- mask_to_poly(best_mask)
    
    if(!is.null(consensus_sf)){
      
      consensus_sf$image <- base
      consensus_sf$object <- "Consensus"
      consensus_sf$mask_id <- 0
      
      sf_list[[k]] <- consensus_sf
      k <- k + 1
    }
    
    
    if(contour_refine){
    refined_consensus<-refine_with_active_contour(consensus_sf = consensus_sf,density = density,img_path = img_path)
    }
    if(debug){
      
      for(mask_id in seq_along(valid_masks)){
        
        poly_sf <- mask_to_poly(valid_masks[[mask_id]])
        
        if(is.null(poly_sf))
          next
        
        poly_sf$image <- base
        poly_sf$object <- "ValidMask"
        poly_sf$mask_id <- mask_id
        
        sf_list[[k]] <- poly_sf
        k <- k + 1
      }
    }
    
    if(length(sf_list) == 0)
      next
    
    image_sf <- bind_rows(sf_list)
    
    saveRDS(
      image_sf,
      file.path(
        overlay_dir,
        paste0(base, ".RDS")
      )
    )
    cat(
      "SUCCESS:",
      base,
      "\n"
    )
    if(base %in% failed_images){
      
      failed_images <- failed_images[
        failed_images != base
      ]
      
      writeLines(
        failed_images,
        failed_file
      )
      
    }
    all_polygons[[base]] <- image_sf
  }, error = function(e){
    
    cat(
      "FAILED:",
      base,
      "\n",
      conditionMessage(e),
      "\n"
    )
    
    if(!(base %in% failed_images)){
      
      write(
        base,
        failed_file,
        append = TRUE
      )
      
      failed_images <<- c(
        failed_images,
        base
      )
      
    }
    
  })
  }
}

find_stomatal_complex(img_dir ="E:/Stomata_maize/all_images/all_images/crops",overlay_dir = "E:/Stomata_maize/all_images/consensus_and_inference_rda3")
