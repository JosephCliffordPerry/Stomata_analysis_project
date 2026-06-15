library(reticulate)
library(dplyr)
library(ggplot2)
library(patchwork)
library(viridis)

find_stomatal_complex <- function(
    img_dir,
    overlay_dir = file.path(img_dir,"overlays"),
    min_ratio = 0.5,
    max_ratio = 0.8
){
  
  dir.create(overlay_dir, showWarnings = FALSE, recursive = TRUE)
  
  # ---------------- PYTHON BACKEND ----------------
  
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

if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        f'https://dl.fbaipublicfiles.com/segment_anything/'+CHECKPOINT_PATH,
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)

def run_segment_collect(image_path, min_ratio, max_ratio):

    image = cv2.imread(image_path)
    if image is None:
        return None, [], None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    total_area = h*w

    predictor.set_image(image_rgb)

    valid_masks = []

    for attempt in range(MAX_RETRIES):

        if attempt == 0:
            cx, cy = w//2, h//2
        else:
            cx = int(np.clip(w//2 + (np.random.rand()-0.5)*2*RANDOM_OFFSET_PERCENTAGE*w,0,w-1))
            cy = int(np.clip(h//2 + (np.random.rand()-0.5)*2*RANDOM_OFFSET_PERCENTAGE*h,0,h-1))

        points = np.array([[cx,cy]])
        labels = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords = points,
            point_labels = labels,
            multimask_output = True
        )

        for mask in masks:
            ratio = mask.sum()/total_area
            if min_ratio <= ratio <= max_ratio:
                valid_masks.append(mask.astype(np.uint8))

    if len(valid_masks) == 0:
        return image_rgb, [], None

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

        for i in range(1,num_labels):

            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / total_area

            if min_ratio <= ratio <= max_ratio:
                best_mask = (labels_cc == i).astype(np.uint8)
                break

        if best_mask is not None:
            break

    if best_mask is None:
        best_mask = (vote >= (len(valid_masks)/2)).astype(np.uint8)

    return image_rgb, valid_masks, best_mask
  ")
  
  np <- import("numpy")
  cv2 <- import("cv2")
  
  img_files <- list.files(
    img_dir,
    pattern="\\.(png|jpg|jpeg|tif|tiff)$",
    full.names=TRUE
  )
  
  # ---------------- PROCESS LOOP ----------------
  
  for(img_path in img_files){
    
    base <- tools::file_path_sans_ext(basename(img_path))
    
    cat("Processing:", base, "\n")
    
    res <- py$run_segment_collect(img_path, min_ratio, max_ratio)
    
    img_rgb <- py_to_r(res[[1]])
    # masks <- lapply(res[[2]], py_to_r)
    consensus <- py_to_r(res[[3]])
    
    if(length(masks)==0) next
    
    # ---------------- IMAGE PREP ----------------
    
    # img_array <- img_rgb/255
    # img_array <- img_array[dim(img_array)[1]:1,,]
    
    # ---------------- MASK DF ----------------
    
    # mask_list <- list()
    # 
    # for(i in seq_along(masks)){
    #   
    #   coords <- which(masks[[i]]>0, arr.ind=TRUE)
    #   
    #   mask_list[[i]] <- data.frame(
    #     mask=i,
    #     row=coords[,1],
    #     col=coords[,2]
    #   )
    # }
    # 
    # mask_df <- bind_rows(mask_list)
    
    # ---------------- CONSENSUS DF ----------------
    
    cons_df <- data.frame()
    
    if(!is.null(consensus)){
      coords <- which(consensus>0, arr.ind=TRUE)
      cons_df <- data.frame(row=coords[,1], col=coords[,2])
      
      np$save(
        file.path(overlay_dir,paste0(base,"_complex_mask.npy")),
        consensus
      )
    }
    
    # # ---------------- HEATMAP PANEL ----------------
    # 
    # p_heat <- ggplot() +
    #   annotation_raster(
    #     img_array,
    #     xmin=0,
    #     xmax=dim(img_array)[2],
    #     ymin=0,
    #     ymax=dim(img_array)[1]
    #   ) +
    #   stat_density_2d(
    #     data=mask_df,
    #     aes(col,row,fill=after_stat(density)),
    #     geom="raster",
    #     contour=FALSE,
    #     alpha=0.8
    #   ) +
    #   scale_fill_viridis_c() +
    #   coord_equal() +
    #   theme_void() +
    #   theme(legend.position="none") +
    #   ggtitle("SAM Vote Density")
    # 
    # # ---------------- CONSENSUS PANEL ----------------
    # 
    # p_cons <- ggplot() +
    #   annotation_raster(
    #     img_array,
    #     xmin=0,
    #     xmax=dim(img_array)[2],
    #     ymin=0,
    #     ymax=dim(img_array)[1]
    #   ) +
    #   geom_point(
    #     data=cons_df,
    #     aes(col,row),
    #     color="red",
    #     size=0.2
    #   ) +
    #   coord_equal() +
    #   theme_void() +
    #   theme(legend.position="none") +
    #   ggtitle("Consensus Stomatal Complex")
    # 
    # p <- p_heat + p_cons
    # 
    # ggsave(
    #   file.path(overlay_dir,paste0(base,"_complex_overlay.png")),
    #   p,
    #   width=8,
    #   height=4
    # )
    
  }
  
  cat("Processing complete\n")
}