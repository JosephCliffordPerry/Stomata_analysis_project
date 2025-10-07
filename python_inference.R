library(reticulate)

run_inference <- function(image_dir, model_path) {
  # --- Load Python environment ---
  Sys.setenv(RETICULATE_PYTHON = "managed")
  py_require(packages = c("ultralytics", "numpy"), python_version = "3.12.4")
  np <- import("numpy")
  ultralytics <- import("ultralytics")
  
  # --- Load single YOLO model ---
  model <- ultralytics$YOLO(model_path)
  
  # --- Get image files ---
  image_paths <- list.files(image_dir, pattern = "\\.tif$", full.names = TRUE)
  n_images <- length(image_paths)
  
  # --- Storage for all detections ---
  results_list <- list()
    for (idx in seq_along(image_paths)) {
      image_path <- image_paths[idx]
      preds <- list()
      
      # --- Run model prediction ---
      results <- tryCatch({
        model$predict(
          source = image_path,
          task = "obb",
          save = FALSE,
          verbose = FALSE
        )[[1]]
      }, error = function(e) NULL)
      
      if (is.null(results)) next
      
      obb_list <- results$obb$xyxyxyxy
      if (obb_list$size(0L) == 0L) next
      
      # --- Extract oriented bounding boxes ---
      for (i in 0:(obb_list$size(0L) - 1)) {
        obb <- obb_list[i]$cpu()$numpy()
        points <- matrix(unlist(py_to_r(obb)), ncol = 2, byrow = TRUE)
        coords <- as.vector(t(points))
        results_list <- append(
          results_list,
          list(data.frame(
            image = basename(image_path),
            shape_id = i + 1,
            x1 = coords[1], y1 = coords[2],
            x2 = coords[3], y2 = coords[4],
            x3 = coords[5], y3 = coords[6],
            x4 = coords[7], y4 = coords[8],
            stringsAsFactors = FALSE
          ))
        )
      }
      
    }
  
  
  # --- Combine into one data frame ---
  if (length(results_list) == 0) {
    return(data.frame())
  } else {
    return(do.call(rbind, results_list))
  }
}
stomata_locations<-run_inference(image_dir = "D:/stomata/frames",model_path = "stomata_test1.pt")
