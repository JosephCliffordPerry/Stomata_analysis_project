library(reticulate)
library(tools)
library(imager)
library(dplyr)

# ------------------------------
# 1️⃣ YOLO OBB inference (3-channel)
# ------------------------------
yolo_obb_inference <- function(image_dir, model_path) {
  
  Sys.setenv(RETICATE_PYTHON = "managed")
  reticulate::py_require(
    packages = c("ultralytics", "opencv-python", "numpy"),
    python_version = "3.12.4"
  )
  
  np <- import("numpy", convert = FALSE)
  cv2 <- import("cv2", convert = FALSE)
  ultralytics <- import("ultralytics")
  
  model <- ultralytics$YOLO(model_path)
  image_paths <- list.files(image_dir, pattern = "\\.tif$", full.names = TRUE)
  
  all_results <- list()
  
  for (img_path in image_paths) {
    message("Processing: ", basename(img_path))
    
    img <- cv2$imread(img_path, cv2$IMREAD_UNCHANGED)
    if (is.null(img)) { message("⚠️ Failed to read image"); next }
    
    # Ensure 3 channels
    if (length(dim(img)) == 2L || (length(dim(img)) == 3L && dim(img)[3] == 1L)) {
      img <- cv2$cvtColor(img, cv2$COLOR_GRAY2BGR)
    } else if (length(dim(img)) == 3L && dim(img)[3] == 4L) {
      img <- cv2$cvtColor(img, cv2$COLOR_BGRA2BGR)
    }
    
    results <- tryCatch({
      model$predict(source = img, task = "obb", save = FALSE, verbose = FALSE)[[1]]
    }, error = function(e) { message("⚠️ Inference failed: ", e$message); return(NULL) })
    
    if (is.null(results)) next
    obb_list <- results$obb$xyxyxyxy
    if (obb_list$size(0L) == 0L) next
    
    image_preds <- list()
    for (i in 0:(obb_list$size(0L)-1)) {
      obb <- obb_list[i]$cpu()$numpy()
      obb_r <- py_to_r(obb)
      flat_vector <- as.numeric(t(obb_r))
      image_preds[[i+1]] <- flat_vector
    }
    
    all_results[[basename(img_path)]] <- image_preds
  }
  
  return(all_results)
}


# ------------------------------
# 2️⃣ Crop crops & save
# ------------------------------
crop_obbs_and_save <- function(detections, image_dir, output_dir) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  crops_all <- list()
  
  for (img_name in names(detections)) {
    img_path <- file.path(image_dir, img_name)
    img_cimg <- load.image(img_path)
    
    obbs <- detections[[img_name]]
    img_crops <- list()
    
    for (i in seq_along(obbs)) {
      pts <- obbs[[i]]              # x1y1x2y2x3y3x4y4
      pts_matrix <- matrix(pts, ncol=2, byrow=TRUE)
      
      # 1️⃣ Center
      center <- colMeans(pts_matrix)
      
      # 2️⃣ Compute width and height
      edge1 <- pts_matrix[2, ] - pts_matrix[1, ]  # top edge
      edge2 <- pts_matrix[3, ] - pts_matrix[2, ]  # right edge
      width <- sqrt(sum(edge1^2))
      height <- sqrt(sum(edge2^2))
      
      # ➕ Increase crop area by +10%
      width  <- width  * 1.10
      height <- height * 1.10
      
      # 3️⃣ Rotation angle
      angle <- -atan2(edge1[2], edge1[1]) * 180 / pi  
      
      # 4️⃣ Rotate around center
      rotated_img <- imrotate(
        img_cimg,
        angle = angle,
        cx = center[1],
        cy = center[2],
        interpolation = 1
      )
      
      # 5️⃣ Compute expanded crop rectangle
      x_start <- max(1, round(center[1] - width/2))
      y_start <- max(1, round(center[2] - height/2))
      x_end <- min(dim(rotated_img)[2], round(center[1] + width/2))
      y_end <- min(dim(rotated_img)[1], round(center[2] + height/2))
      
      cropped_img <- imsub(rotated_img, x %in% x_start:x_end, y %in% y_start:y_end)
      
      # 6️⃣ Save crop file
      crop_filename <- file.path(output_dir, paste0(file_path_sans_ext(img_name), "_obb_", i, ".png"))
      save.image(cropped_img, file = crop_filename)
      
      # 7️⃣ Record metadata
      img_crops[[i]] <- data.frame(
        original_image = img_name,
        crop_file = crop_filename,
        x_center = center[1],
        y_center = center[2],
        width_expanded = width,
        height_expanded = height,
        angle = angle
      )
    }
    
    crops_all[[img_name]] <- do.call(rbind, img_crops)
  }
  
  crops_df <- do.call(rbind, crops_all)
  write.csv(crops_df, file.path(output_dir, "cropped_obbs.csv"), row.names = FALSE)
  return(crops_df)
}


# ------------------------------
# 3️⃣ Main pipeline
# ------------------------------
individual_stomata_crop <- function(image_dir, model_path) {
  message("=== Starting YOLO OBB inference + Cropping ===")
  detections <- yolo_obb_inference(image_dir, model_path)
  if (length(detections) == 0) { message("No detections found"); return(NULL) }
  
  output_dir <- file.path(image_dir, "crops")
  crops_df <- crop_obbs_and_save(detections, image_dir, output_dir)
  
  message("=== Done ===")
  return(crops_df)
}


# ------------------------------
# Example run
# ------------------------------
# image_dir <- "D:/stomata/November_images/November_image_data_8bit_tifs"
# model_path <- "Stomata_obbox.pt"
# 
# crops_df <- individual_stomata_crop(image_dir, model_path)
