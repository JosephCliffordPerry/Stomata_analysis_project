# ==============================================================
# YOLO Inference + FFmpeg Check + Cropping via FFmpeg
# ==============================================================
library(reticulate)
library(tools)

# --------------------------------------------------------------
# 1. Ensure FFmpeg availability
# --------------------------------------------------------------
get_ffmpeg_path <- function() {
  sys_ffmpeg <- Sys.which("ffmpeg")
  if (nzchar(sys_ffmpeg)) {
    message("Using system FFmpeg: ", sys_ffmpeg)
    return(normalizePath(sys_ffmpeg))
  }
  
  message("FFmpeg not found. Using managed Python environment...")
  Sys.setenv(RETICULATE_PYTHON = "managed")
  reticulate::py_require(
    packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","imageio-ffmpeg", "ultralytics", "numpy"), 
    python_version = "3.12.4"
  )

  
  ffmpeg <- reticulate::import("imageio_ffmpeg")
  exe <- ffmpeg$get_ffmpeg_exe()
  message("Using managed FFmpeg: ", exe)
  return(normalizePath(exe))
}

ffmpeg_path <- get_ffmpeg_path()

# --------------------------------------------------------------
# 2. YOLO inference (single model)
# --------------------------------------------------------------
run_inference <- function(image_dir, model_path) {
  Sys.setenv(RETICULATE_PYTHON = "managed")
  reticulate::py_require(packages = c("ultralytics", "numpy"), python_version = "3.12.4")
  
  ultralytics <- import("ultralytics")
  model <- ultralytics$YOLO(model_path)
  
  image_paths <- list.files(image_dir, pattern = "\\.jpg$", full.names = TRUE)
  results_list <- list()
  
  for (image_path in image_paths) {
    message("Processing: ", basename(image_path))
    results <- tryCatch({
      model$predict(source = image_path, task = "obb", save = FALSE, verbose = FALSE)[[1]]
    }, error = function(e) NULL)
    
    if (is.null(results)) next
    
    obb_list <- results$obb$xyxyxyxy
    if (obb_list$size(0L) == 0L) next
    
    for (i in 0:(obb_list$size(0L) - 1)) {
      obb <- obb_list[i]$cpu()$numpy()
      points <- matrix(unlist(py_to_r(obb)), ncol = 2, byrow = TRUE)
      coords <- as.vector(t(points))
      results_list <- append(
        results_list,
        list(data.frame(
          image = basename(image_path),
          shape_id = i + 1,
          x1 = coords[1], y1 = coords[5],
          x2 = coords[2], y2 = coords[6],
          x3 = coords[3], y3 = coords[7],
          x4 = coords[4], y4 = coords[8],
          stringsAsFactors = FALSE
        ))
      )
    }
  }
  
  if (length(results_list) == 0) return(data.frame())
  do.call(rbind, results_list)
}

# --------------------------------------------------------------
# 3. Crop using FFmpeg (fast + efficient)
# --------------------------------------------------------------
crop_detections_ffmpeg <- function(detections, image_dir, output_base_dir, ffmpeg_path) {
  if (nrow(detections) == 0) {
    message("No detections found.")
    return(invisible(NULL))
  }
  
  # Create the shared output folder
  dir.create(output_base_dir, recursive = TRUE, showWarnings = FALSE)
  
  for (i in seq_len(nrow(detections))) {
    row <- detections[i, ]
    img_path <- file.path(image_dir, row$image)
    
    if (!file.exists(img_path)) {
      message("Image not found, skipping: ", row$image)
      next
    }
    
    # Compute bounding box coordinates
    x_coords <- c(row$x1, row$x2, row$x3, row$x4)
    y_coords <- c(row$y1, row$y2, row$y3, row$y4)
    
    x_min <- as.integer(round(max(0, min(x_coords))))
    y_min <- as.integer(round(max(0, min(y_coords))))
    width  <- as.integer(round(max(1, max(x_coords) - x_min)))
    height <- as.integer(round(max(1, max(y_coords) - y_min)))
    
    # Skip invalid or zero-size boxes
    if (width <= 1 || height <= 1) {
      message(sprintf("Skipping invalid crop for %s (shape_id=%d)", row$image, row$shape_id))
      next
    }
    
    # Output file name: combine image name + shape ID
    base_name <- file_path_sans_ext(row$image)
    output_path <- file.path(output_base_dir, sprintf("%s_shape_%d.jpg", base_name, row$shape_id))
    
    # FFmpeg crop command
    cmd <- sprintf(
      '"%s" -y -i "%s" -vf "crop=%d:%d:%d:%d" -frames:v 1 "%s"',
      ffmpeg_path, img_path, width, height, x_min, y_min, output_path
    )
    
    system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
  }
  
  message("✅ Cropping complete. All jpgF crops saved to: ", normalizePath(output_base_dir))
}



# --------------------------------------------------------------
# 4. Main runner
# --------------------------------------------------------------
individual_stomata_crop <- function(image_dir,model_path){
  
  message("=== Starting YOLO Inference and FFmpeg Cropping ===")
  detections <- run_inference(image_dir, model_path)
  message("Detected ", nrow(detections), " objects total.")
  
  # Save all crops into one folder named "crops"
  output_dir <- file.path(image_dir, "crops")
  crop_detections_ffmpeg(detections, image_dir, output_dir, ffmpeg_path)
  
  message("=== Done ===")
}
# --------------------------------------------------------------
# 5. Run
# --------------------------------------------------------------
individual_stomata_crop(image_dir = "D:/stomata/Just testing with Leica-ATC2000 - 20X/",
                        model_path = "stomata_test1.pt")
image_dir <- "D:/stomata/Just testing with Leica-ATC2000 - 20X/"
model_path <- "stomata_test1.pt"

