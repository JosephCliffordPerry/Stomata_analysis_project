library(reticulate)
library(tools)
library(imager)
library(dplyr)

yolo_seg_inference <- function(image_dir, model_path) {
  
  Sys.setenv(RETICATE_PYTHON = "managed")
  reticulate::py_require(
    packages = c("ultralytics", "opencv-python", "numpy"),
    python_version = "3.12.4"
  )
  
  np <- import("numpy", convert = FALSE)
  cv2 <- import("cv2", convert = FALSE)
  ultralytics <- import("ultralytics")
  
  model <- ultralytics$YOLO(model_path)
  image_paths <- list.files(image_dir, pattern = "\\.(tif|png|jpg)$", full.names = TRUE)
  
  all_segmentations <- list()
  
  for (img_path in image_paths) {
    
    message("Processing: ", basename(img_path))
    
    img <- cv2$imread(img_path, cv2$IMREAD_COLOR)
    if (is.null(img)) { 
      message("⚠️ Failed to read image - skipping") 
      next 
    }
    
    # --- ensure 3 channels ---
    if (length(dim(img)) == 2L) {
      img <- cv2$cvtColor(img, cv2$COLOR_GRAY2BGR)
    } else if (length(dim(img)) == 3L && dim(img)[3] == 1L) {
      img <- cv2$cvtColor(img, cv2$COLOR_GRAY2BGR)
    } else if (length(dim(img)) == 3L && dim(img)[3] == 4L) {
      img <- cv2$cvtColor(img, cv2$COLOR_BGRA2BGR)
    }
    
    # --- segmentation inference ---
    results <- tryCatch({
      model$predict(source = img, task = "segment", save = FALSE, verbose = FALSE)[[1]]
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
    
    for (i in 1:n_objects) {
      
      py_i <- i - 1   # python index
      
      # --------------------------
      # 1️⃣ Mask (tensor → numpy → R matrix)
      # --------------------------
      mask_np <- seg_masks$data[py_i]$cpu()$numpy()
      mask_r <- py_to_r(mask_np)
      mask_r <- matrix(mask_r, nrow = nrow(mask_r), byrow = TRUE)
      
      # --------------------------
      # 2️⃣ Polygon (list of xy points → R vector)
      # --------------------------
      seg_poly_r <- py_to_r(seg_masks$xy[[i]])
      poly_flat <- as.numeric(unlist(seg_poly_r))
      
      # --------------------------
      # 3️⃣ Bounding box (tensor → numeric)
      # --------------------------
      bbox_np <- results$boxes$xyxy[py_i]$cpu()$numpy()
      bbox_r <- as.numeric(py_to_r(bbox_np))
      
      # --------------------------
      # 4️⃣ Class ID
      # --------------------------
      cls_np <- results$boxes$cls[py_i]$cpu()$numpy()
      cls_r <- as.numeric(py_to_r(cls_np))
      
      # --------------------------
      # 5️⃣ Confidence
      # --------------------------
      conf_np <- results$boxes$conf[py_i]$cpu()$numpy()
      conf_r <- as.numeric(py_to_r(conf_np))
      
      # --------------------------
      # 6️⃣ Store clean R values
      # --------------------------
      obj <- list(
        class_id      = cls_r,
        confidence    = round(conf_r, 4),
        bbox_xyxy     = bbox_r,
        segmentation_polygon = poly_flat,
        mask_binary   = mask_r,
        pixel_area    = sum(mask_r > 0)
      )
      
      image_obj_list[[i]] <- obj
    }
    
    all_segmentations[[basename(img_path)]] <- image_obj_list
  }
  
  return(all_segmentations)
}
