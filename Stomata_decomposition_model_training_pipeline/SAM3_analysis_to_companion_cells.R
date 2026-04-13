library(reticulate)
library(dplyr)
library(ggplot2)
library(patchwork)

find_companion_cells <- function(
    mask_dir,
    img_dir,
    out_dir = file.path(img_dir,"dual_masks_overlay"),
    min_ratio = 0.05,
    max_ratio = 0.20
){

    # ---------------- SAM BACKEND ----------------
  np <- import("numpy")
  cv2 <- import("cv2")
  py_run_string("
import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

MODEL_TYPE = 'vit_h'
CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'

MAX_RETRIES = 20
RANDOM_OFFSET_PERCENTAGE = 0.05

if not os.path.exists(CHECKPOINT_PATH):
    torch.hub.download_url_to_file(
        f'https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_PATH}',
        CHECKPOINT_PATH
    )

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)


def smooth_single_component(mask):

    mask = mask.astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        cnt = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(cnt)

        smooth_mask = np.zeros_like(mask)

        cv2.drawContours(smooth_mask, [hull], -1, 1, -1)

        mask = smooth_mask

    return mask



def run_segment_collect(image_path, points, labels, min_ratio, max_ratio):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    total_area = h*w

    predictor.set_image(image)

    valid_masks = []

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
            point_coords = pts,
            point_labels = labs,
            multimask_output = True
        )

        for mask in masks:

            ratio = mask.sum()/total_area

            if min_ratio <= ratio <= max_ratio:
                valid_masks.append(mask.astype(np.uint8))

    if len(valid_masks) == 0:
        return [], None


    stack = np.stack(valid_masks)

    vote = np.sum(stack, axis=0)
    density = vote / len(valid_masks)


    pos_points = points[labels == 1]

    prompt_centroid = np.array([
        pos_points[:,1].mean(),
        pos_points[:,0].mean()
    ])

    max_centroid_dist = 0.25 * max(h, w)


    thresholds = np.linspace(1.0, 0.0, 200)

    best_mask = None
    best_dist = None


    for t in thresholds:

        candidate = (density >= t).astype(np.uint8)

        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8
        )

        for i in range(1, num_labels):

            component = (labels_cc == i)

            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / total_area

            if ratio > max_ratio:
                continue


            intersects_prompt = False

            for p in pos_points:

                x = int(p[0])
                y = int(p[1])

                if 0 <= y < h and 0 <= x < w:
                    if component[y, x]:
                        intersects_prompt = True
                        break

            if not intersects_prompt:
                continue


            comp_centroid = centroids[i]
            comp_centroid = np.array([comp_centroid[1], comp_centroid[0]])

            dist = np.linalg.norm(comp_centroid - prompt_centroid)

            if dist > max_centroid_dist:
                continue


            if ratio >= min_ratio:

                component = smooth_single_component(component)

                return valid_masks, component.astype(np.uint8)


            if best_mask is None or dist < best_dist:

                best_mask = component.astype(np.uint8)
                best_dist = dist



    if best_mask is None:

        candidate = (density > 0).astype(np.uint8)

        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8
        )

        if num_labels > 1:

            dists = []

            for i in range(1, num_labels):

                comp_centroid = centroids[i]
                comp_centroid = np.array([comp_centroid[1], comp_centroid[0]])

                d = np.linalg.norm(comp_centroid - prompt_centroid)

                dists.append(d)

            best = 1 + np.argmin(dists)

            best_mask = (labels_cc == best).astype(np.uint8)


    if best_mask is not None:

        best_mask = smooth_single_component(best_mask)


    return valid_masks, best_mask
")
    
    # ---------------- FUNCTIONS ----------------
    
    longest_line <- function(mask){
      
      pts <- which(mask>0, arr.ind=TRUE)
      hull <- chull(pts[,2], pts[,1])
      hull_pts <- pts[hull,]
      
      d <- as.matrix(dist(hull_pts))
      pair <- which(d==max(d), arr.ind=TRUE)[1,]
      
      list(
        p1 = hull_pts[pair[1],],
        p2 = hull_pts[pair[2],]
      )
    }
    
    generate_points <- function(p1,p2,mask){
      
      v <- p2-p1
      v <- v/sqrt(sum(v^2))
      
      perp <- c(-v[2],v[1])
      
      L <- sqrt(sum((p2-p1)^2))
      mid <- (p1+p2)/2
      
      pos_t <- c(0.3,0.5,0.7)
      offset <- L*0.18
      
      side1 <- t(sapply(pos_t,function(t){
        mid+(t-0.5)*L*v+perp*offset
      }))
      
      side2 <- t(sapply(pos_t,function(t){
        mid+(t-0.5)*L*v-perp*offset
      }))
      
      neg_diam <- t(sapply(pos_t,function(t){
        mid+(t-0.5)*L*v
      }))
      
      neg_out1 <- side1+perp*offset*0.6
      neg_out2 <- side2-perp*offset*0.6
      
      list(
        positives1=side1,
        positives2=side2,
        negatives=rbind(neg_diam,neg_out1,neg_out2)
      )
    }
    
    make_seg_df <- function(mask,base,name){
      
      if(is.null(mask) || sum(mask)==0){
        return(data.frame())
      }
      
      coords <- which(mask>0, arr.ind=TRUE)
      
      data.frame(
        image = base,
        object = name,
        row = coords[,1],
        col = coords[,2]
      )
    }
    
    # ---------------- PATHS ----------------
    dir.create(out_dir,showWarnings=FALSE)
    
    mask_files <- list.files(mask_dir,"_mask.npy$",full.names=TRUE)
    
    results <- list()
    processed_count <- 0
    
    # ---------------- MAIN LOOP ----------------
    
    for(mask_file in mask_files){
      
      base <- sub("_complex_mask.npy","",basename(mask_file))
      img_path <- file.path(img_dir,paste0(base,".png"))
      
      if(!file.exists(img_path)) next
      
      cat("Processing:",base,"\n")
      
      mask <- py_to_r(np$load(mask_file))
      
      line <- longest_line(mask)
      
      pts <- generate_points(line$p1,line$p2,mask)
      
      pos1 <- pts$positives1[,c(2,1)]
      pos2 <- pts$positives2[,c(2,1)]
      neg  <- pts$negatives[,c(2,1)]
      
      points1 <- rbind(pos1,neg)
      labels1 <- c(rep(1,nrow(pos1)),rep(0,nrow(neg)))
      
      points2 <- rbind(pos2,neg)
      labels2 <- c(rep(1,nrow(pos2)),rep(0,nrow(neg)))
      
      res1 <- py$run_segment_collect(
        img_path,
        np$array(points1,dtype="float32"),
        np$array(labels1,dtype="float32"),
        min_ratio,max_ratio
      )
      
      res2 <- py$run_segment_collect(
        img_path,
        np$array(points2,dtype="float32"),
        np$array(labels2,dtype="float32"),
        min_ratio,max_ratio
      )
      
      masks1 <- lapply(res1[[1]],py_to_r)
      masks2 <- lapply(res2[[1]],py_to_r)
      
      consensus1 <- py_to_r(res1[[2]])
      consensus2 <- py_to_r(res2[[2]])
      
      seg_list <- list()
      
      seg_list[[1]] <- make_seg_df(mask,base,"Complex")
      
      for(i in seq_along(masks1)){
        seg_list[[length(seg_list)+1]] <-
          make_seg_df(masks1[[i]],base,paste0("Companion1_mask_",i))
      }
      
      for(i in seq_along(masks2)){
        seg_list[[length(seg_list)+1]] <-
          make_seg_df(masks2[[i]],base,paste0("Companion2_mask_",i))
      }
      
      seg_list[[length(seg_list)+1]] <-
        make_seg_df(consensus1,base,"Companion1_consensus")
      
      seg_list[[length(seg_list)+1]] <-
        make_seg_df(consensus2,base,"Companion2_consensus")
      
      seg_df <- bind_rows(seg_list)
      
      results[[base]] <- seg_df
      
      py_run_string(sprintf(
        "import cv2; img = cv2.cvtColor(cv2.imread(r'%s'), cv2.COLOR_BGR2RGB)",
        img_path
      ))
      
      img_array <- py_to_r(py$img)/255
      img_array <- img_array[dim(img_array)[1]:1,,]
      
      # ---------------- heatmap panel ----------------
      p_all <- ggplot() +
        annotation_raster(
          img_array,
          xmin=0,
          xmax=dim(img_array)[2],
          ymin=0,
          ymax=dim(img_array)[1]
        ) +
        stat_density_2d(
          data=seg_df,
          aes(col,row,fill=after_stat(density)),
          geom="raster",
          contour=FALSE,
          alpha=0.8
        ) +
        scale_fill_viridis_c() +
        coord_equal() +
        theme_void() +
        theme(legend.position="none") +
        ggtitle("All SAM Masks Density")
      
      # ---------------- consensus panel ----------------
      
      complex_df <- seg_df %>% filter(object=="Complex")
      
      consensus_df <- seg_df %>%
        filter(object %in% c("Companion1_consensus","Companion2_consensus"))
      
      p_consensus <- ggplot() +
        annotation_raster(
          img_array,
          xmin=0,
          xmax=dim(img_array)[2],
          ymin=0,
          ymax=dim(img_array)[1]
        ) +
        geom_point(
          data=consensus_df,
          aes(col,row),
          color="red",
          size=0.2
        ) +
        geom_point(
          data=complex_df,
          aes(col,row),
          color="blue",
          size=0.2,
          alpha=0.7
        ) +
        coord_equal() +
        theme_void() +
        theme(legend.position="none") +
        ggtitle("Consensus Companion Cells")
      
      p <- p_all + p_consensus
      
      ggsave(
        file.path(out_dir,paste0(base,"_multipanel_overlay.png")),
        p,
        width=8,
        height=4
      )
      
      processed_count <- processed_count+1
      
      if(processed_count %% 10 == 0){
        
        combined <- bind_rows(results)
        
        saveRDS(
          combined,
          file.path(out_dir,"cell_components_dataframe_checkpoint.rds")
        )
        
        cat("Checkpoint saved after",processed_count,"images\n")
      }
      
    }
    
    final_df <- bind_rows(results)
    
    saveRDS(
      final_df,
      file.path(out_dir,"cell_components_dataframe_final.rds")
    )
    
    cat("Processing complete\n")}