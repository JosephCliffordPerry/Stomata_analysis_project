library(reticulate)
library(dplyr)
library(ggplot2)

np <- import("numpy")

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
        f'https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_PATH}',
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)

def run_segment_with_labels(image_path, points, labels, min_ratio, max_ratio):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    total_area = h*w
    predictor.set_image(image)
    best_mask = None
    best_score = -1
    for attempt in range(MAX_RETRIES):
        pts = points.copy()
        labs = labels.copy()
        if attempt > 0:
            jitter = np.random.uniform(
                -RANDOM_OFFSET_PERCENTAGE,
                 RANDOM_OFFSET_PERCENTAGE,
                 size=pts.shape
            )
            pts[:,0] += jitter[:,0]*w
            pts[:,1] += jitter[:,1]*h
        masks, scores, logits = predictor.predict(
            point_coords=pts,
            point_labels=labs,
            multimask_output=True
        )
        for mask, score in zip(masks, scores):
            ratio = mask.sum() / total_area
            if min_ratio <= ratio <= max_ratio:
                if score > best_score:
                    best_mask = mask
                    best_score = score
    return best_mask
")

# ---------------- GEOMETRY FUNCTIONS ----------------
longest_line <- function(mask){
  pts <- which(mask>0, arr.ind=TRUE)
  hull <- chull(pts[,2], pts[,1])
  hull_pts <- pts[hull,]
  d <- as.matrix(dist(hull_pts))
  pair <- which(d==max(d), arr.ind=TRUE)[1,]
  list(p1=hull_pts[pair[1],], p2=hull_pts[pair[2],])
}

generate_points <- function(p1, p2){
  v <- p2 - p1
  v <- v / sqrt(sum(v^2))
  perp <- c(-v[2], v[1])
  L <- sqrt(sum((p2-p1)^2))
  mid <- (p1 + p2)/2
  pos_t <- c(0.3,0.5,0.7)
  offset <- L*0.18
  side1 <- t(sapply(pos_t,function(t){ mid + (t-0.5)*L*v + perp*offset }))
  side2 <- t(sapply(pos_t,function(t){ mid + (t-0.5)*L*v - perp*offset }))
  neg_diam <- t(sapply(pos_t,function(t){ mid + (t-0.5)*L*v }))
  # Negative points pushed further out (factor 0.5)
  neg_out1 <- side1 + perp*offset*0.5
  neg_out2 <- side2 - perp*offset*0.5
  list(
    positives1 = side1,
    positives2 = side2,
    negatives = rbind(neg_diam, neg_out1, neg_out2)
  )
}

# ---------------- PATHS ----------------
mask_dir <- "E:/Stomata_maize/all_images/all_images/crops/overlays"
img_dir  <- "E:/Stomata_maize/all_images/all_images/crops"
out_dir  <- file.path(img_dir,"dual_masks_overlay")
dir.create(out_dir, showWarnings = FALSE)
mask_files <- list.files(mask_dir,"_mask.npy$",full.names = TRUE)
results <- list()

# ---------------- HELPER: Safe seg dataframe ----------------
make_seg_df <- function(mask, base, name){
  if(is.null(mask) || sum(mask)==0){
    return(data.frame(image=character(0), object=character(0), row=integer(0), col=integer(0)))
  } else {
    coords <- which(mask>0, arr.ind=TRUE)
    df <- data.frame(image=base, object=name, row=coords[,1], col=coords[,2])
    return(df)
  }
}

# ---------------- MAIN LOOP ----------------
for(mask_file in mask_files){
  base <- sub("_mask.npy","",basename(mask_file))
  img_path <- file.path(img_dir,paste0(base,".png"))
  if(!file.exists(img_path)) next
  
  print(paste0("Processing image: ", base))
  
  mask <- py_to_r(np$load(mask_file))
  line <- longest_line(mask)
  pts <- generate_points(line$p1,line$p2)
  
  # --- SAM points & labels ---
  pos1 <- pts$positives1[,c(2,1)]
  pos2 <- pts$positives2[,c(2,1)]
  neg  <- pts$negatives[,c(2,1)]
  points1 <- rbind(pos1,neg)
  labels1 <- c(rep(1,nrow(pos1)), rep(0,nrow(neg)))
  points2 <- rbind(pos2,neg)
  labels2 <- c(rep(1,nrow(pos2)), rep(0,nrow(neg)))
  
  # --- run SAM inference ---
  comp1 <- py$run_segment_with_labels(img_path, np$array(points1,dtype="float32"), np$array(labels1,dtype="float32"), 0.05,0.20)
  comp2 <- py$run_segment_with_labels(img_path, np$array(points2,dtype="float32"), np$array(labels2,dtype="float32"), 0.05,0.20)
  comp1 <- py_to_r(comp1)
  comp2 <- py_to_r(comp2)
  
  # --- dataframes ---
  seg_df <- bind_rows(
    make_seg_df(mask, base, "Complex"),
    make_seg_df(comp1, base, "Companion_1"),
    make_seg_df(comp2, base, "Companion_2")
  )
  
  points_df <- data.frame(
    image = base,
    object = c(
      "Diameter_A","Diameter_B",
      paste0("Companion1_prompt_",1:nrow(pos1)),
      paste0("Companion2_prompt_",1:nrow(pos2))
    ),
    row = c(pos1[2,2], pos2[2,2], pos1[,2], pos2[,2]),
    col = c(pos1[2,1], pos2[2,1], pos1[,1], pos2[,1])
  )
  
  neg_points <- data.frame(
    image = base,
    object = paste0("Negative_", seq_len(nrow(neg))),
    row = neg[,2],
    col = neg[,1]
  )
  
  points_df <- bind_rows(points_df, neg_points)
  df_all <- bind_rows(seg_df, points_df)
  results[[base]] <- df_all
  
  # --- read original image and flip Y only ---
  py_run_string(sprintf("import cv2; img = cv2.cvtColor(cv2.imread(r'%s'), cv2.COLOR_BGR2RGB)", img_path))
  img_array <- py_to_r(py$img)/255
  img_array_flipped <- img_array[dim(img_array)[1]:1, , ]  # flip Y-axis only
  
  # --- plot overlay ---
  plot_seg <- seg_df %>% sample_n(min(nrow(seg_df),20000))
  
  p <- ggplot() +
    annotation_raster(
      img_array_flipped,
      xmin = 0, xmax = dim(img_array_flipped)[2],
      ymin = 0, ymax = dim(img_array_flipped)[1]
    ) +
    geom_point(data=plot_seg, aes(col,row,color=object), size=0.2) +
    geom_point(data=points_df %>% filter(!grepl("Negative",object)),
               aes(col,row), shape=4, size=3, stroke=1.2, color="yellow") +
    geom_point(data=points_df %>% filter(grepl("Negative",object)),
               aes(col,row), shape=4, size=2, stroke=1.0, color="black") +
    geom_segment(aes(x=line$p1[2], y=line$p1[1], xend=line$p2[2], yend=line$p2[1]),
                 color="white", linewidth=0.6) +
    scale_color_manual(values=c(Complex="blue", Companion_1="red", Companion_2="green")) +
    coord_equal() +
    theme_void()
  
  ggsave(file.path(out_dir,paste0(base,"_dual_overlay.png")), p, width=4, height=4)
}

# Save combined dataframe
final_df <- bind_rows(results)
saveRDS(final_df, file.path(out_dir,"cell_components_dataframe.rds"))