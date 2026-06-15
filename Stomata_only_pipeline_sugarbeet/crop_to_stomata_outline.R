yolo_seg_inference <- function(image_dir, model_path, save_masks = FALSE) {
  
  Sys.setenv(RETICATE_PYTHON = "managed")
  reticulate::py_require(
    packages = c("ultralytics", "opencv-python", "numpy"),
    python_version = "3.12.4"
  )
  
  np  <- import("numpy", convert = FALSE)
  cv2 <- import("cv2", convert = FALSE)
  ultralytics <- import("ultralytics")
  
  model <- ultralytics$YOLO(model_path)
  image_paths <- list.files(image_dir, pattern = "\\.(tif|png|jpg)$", full.names = TRUE)
  
  # folder for masks
  if (save_masks) {
    dir.create(file.path(image_dir, "saved_masks"), showWarnings = FALSE)
  }
  
  all_segmentations <- list()
  
  for (img_path in image_paths) {
    
    img_name <- tools::file_path_sans_ext(basename(img_path))
    message("Processing: ", img_name)
    
    img <- cv2$imread(img_path, cv2$IMREAD_COLOR)
    if (is.null(img)) { 
      message("⚠️ Failed to read image - skipping") 
      next 
    }
    
    # original resolution
    orig_h <- dim(img)[1]
    orig_w <- dim(img)[2]
    
    # ensure 3 channels
    if (length(dim(img)) == 2L) {
      img <- cv2$cvtColor(img, cv2$COLOR_GRAY2BGR)
    } else if (length(dim(img)) == 3L && dim(img)[3] == 1L) {
      img <- cv2$cvtColor(img, cv2$COLOR_GRAY2BGR)
    } else if (length(dim(img)) == 3L && dim(img)[3] == 4L) {
      img <- cv2$cvtColor(img, cv2$COLOR_BGRA2BGR)
    }
    
    # YOLO prediction
    results <- tryCatch({
      model$predict(source=img, task="segment", save=FALSE, verbose=FALSE)[[1]]
    }, error = function(e) {
      message("⚠️ Inference failed: ", e$message)
      return(NULL)
    })
    
    if (is.null(results)) next
    
    seg_masks <- results$masks
    if (is.null(seg_masks) || length(seg_masks$xy) == 0) {
      message("⚠️ No objects detected — skipping.")
      all_segmentations[[basename(img_path)]] <- list()
      next
    }
    
    n_objects <- length(seg_masks$xy)
    image_obj_list <- list()
    
    # YOLO model mask resolution
    mask_h <- seg_masks$shape[[1]]
    mask_w <- seg_masks$shape[[2]]
    area_scale <- (orig_w / mask_w) * (orig_h / mask_h)
    
    for (i in 1:n_objects) {
      
      py_i <- i - 1
      
      # raw YOLO mask (640×640)
      mask_np <- seg_masks$data[py_i]$cpu()$numpy()
      mask_r  <- py_to_r(mask_np)
      
      # -------------------------------
      # Resize mask back to original size
      # -------------------------------
      mask_resized_np <- cv2$resize(
        mask_np,
        dsize = reticulate::tuple(orig_w, orig_h),
        interpolation = cv2$INTER_NEAREST
      )
      mask_resized <- py_to_r(mask_resized_np)
      mask_resized <- matrix(mask_resized, nrow = orig_h, byrow = TRUE)
      
      # polygon
      seg_poly_r <- py_to_r(seg_masks$xy[[i]])
      
      # bounding box
      bbox_np <- results$boxes$xyxy[py_i]$cpu()$numpy()
      bbox_r <- as.numeric(py_to_r(bbox_np))
      
      # class
      cls_r <- as.numeric(py_to_r(results$boxes$cls[py_i]$cpu()$numpy()))
      
      # confidence
      conf_r <- as.numeric(py_to_r(results$boxes$conf[py_i]$cpu()$numpy()))
      
      # -------------------------------
      # Corrected area using scaling
      # -------------------------------
      true_area <- sum(mask_resized > 0)
      
      # -------------------------------
      # Save mask if option is on
      # -------------------------------
      if (save_masks) {
        save_path <- file.path(image_dir, "saved_masks",
                               paste0(img_name, "_mask_", i, ".png"))
        
        # convert to 0–255 uint8
        mask_uint8 <- cv2$normalize(mask_resized_np, NULL, 0L, 255L, cv2$NORM_MINMAX)
        cv2$imwrite(save_path, mask_uint8)
      }
      
      obj <- list(
        class_id = cls_r,
        confidence = round(conf_r, 4),
        bbox_xyxy = bbox_r,
        segmentation_polygon = seg_poly_r,
        mask_binary = mask_resized,
        pixel_area = true_area
      )
      
      image_obj_list[[i]] <- obj
    }
    
    all_segmentations[[basename(img_path)]] <- image_obj_list
  }
  
  return(all_segmentations)
}

#image_dir <- "D:/stomata/November_images/november_toy_data/crops"
# model_path <-"single_stomata_internal.pt"
#stomata_list<-yolo_seg_inference(image_dir = crop_dir,model_path = "single_stomata_internal.pt", save_masks = TRUE)  


