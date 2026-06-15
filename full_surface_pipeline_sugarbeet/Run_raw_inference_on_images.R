# ===============================
# CONFIG
# ===============================
input_dir  <- "E:/Stomata/Sugarbeet_stomata_imaging/sugarbeet_all_mips"
output_dir <- file.path(input_dir, "rda_outputs")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ===============================
# LOAD LIBRARIES
# ===============================
suppressPackageStartupMessages({
  library(tools)   # for file_path_sans_ext
})

source("full_surface_pipeline/Tile_based_yolo_inference.R")

# ===============================
# FILE LIST
# ===============================
image_files <- list.files(
  input_dir,
  pattern = "\\.(png|jpg|jpeg|tif|tiff)$",
  full.names = TRUE,
  ignore.case = TRUE
)

# ===============================
# MAIN LOOP
# ===============================
for (img_path in image_files) {
  
  cat("Processing:", img_path, "\n")
  
  # Run processing
  result <- tryCatch(
    run_yolo_pipeline(params <- list(
      image_path = img_path,
      model_path = "E:/Stomata/Sugarbeet_stomata_imaging/beetmip_model2/beetmip_model2.pt",
      tile_size  = 128,
      overlap    = 64,
      iou_thresh = 0.5,
      min_area   = 1000,
      max_area   = 3000,
      min_circ   = 0.5,
      max_circ   = 1.0,
      alpha      = 0.4
    )),
    error = function(e) {
      warning(sprintf("Failed on %s: %s", img_path, e$message))
      return(NULL)
    }
  )
  
  if (is.null(result)) next
  
  # Output filename
  base_name <- file_path_sans_ext(basename(img_path))
  out_file  <- file.path(output_dir, paste0(base_name, ".RDA"))
  
  # Save
  save(result, file = out_file, compress = "xz")
}

cat("Done.\n")

