library(reticulate)
library(dplyr)
library(stringr)

Sys.setenv(RETICULATE_PYTHON = "managed")

# ✅ Load Python environment and segmentation function
reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image", "scipy", "imageio-ffmpeg", "ultralytics"),
  python_version = "3.12.4"
)

# Assume `segment_cells_active_contour_edge_refined` is already defined in Python
# (use the corrected "thicker boundaries" version we created earlier)

segment_cells_folder <- function(input_folder, output_folder,
                                 min_cell_area = 100,
                                 max_cell_area = 5000,
                                 iterations = 300,
                                 invert = FALSE,
                                 block_size_adaptive = 25,
                                 edge_weight = 1.0,
                                 min_distance_peaks = 5,
                                 outline_thickness = 2) {
  
  # Create output folder if it doesn't exist
  if(!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
  
  # List all TIFF/PNG/JPG images in folder
  files <- list.files(input_folder, pattern = "\\.(tif|tiff|png|jpg)$", full.names = TRUE)
  
  # Initialize results list
  results <- data.frame(
    filename = character(),
    cell_count = integer(),
    stringsAsFactors = FALSE
  )
  
  for(file in files) {
    cat("Processing:", basename(file), "\n")
    
    # Run segmentation in Python
    out <- py$segment_cells_active_contour_edge_refined(
      filename = r_to_py(file),
      min_cell_area = r_to_py(as.integer(min_cell_area)),
      max_cell_area = r_to_py(as.integer(max_cell_area)),
      iterations = r_to_py(as.integer(iterations)),
      invert = r_to_py(as.logical(invert)),
      block_size_adaptive = r_to_py(as.integer(block_size_adaptive)),
      edge_weight = r_to_py(as.numeric(edge_weight)),
      min_distance_peaks = r_to_py(as.integer(min_distance_peaks)),
      outline_thickness = r_to_py(as.integer(outline_thickness))
    )
    
    labeled_rgb <- out[[1]]
    overlay <- out[[2]]
    
    # Count cells
    labeled_np <- py$np$asarray(labeled_rgb)
    n_cells <- length(unique(as.vector(labeled_np))) - 1  # subtract background
    
    # Save overlay as PNG
    out_file <- file.path(output_folder, paste0(tools::file_path_sans_ext(basename(file)), "_boundaries.png"))
    plt <- import("matplotlib.pyplot")
    plt$imsave(out_file, overlay)
    
    # Append to results
    results <- rbind(results, data.frame(filename = basename(file), cell_count = n_cells))
  }
  
  return(results)
}

# 🔹 Example usage:
input_folder <- "D:/stomata/training_tifs_8bit"
output_folder <- "D:/stomata/output_boundaries"

cell_counts <- segment_cells_folder(
  input_folder = input_folder,
  output_folder = output_folder,
  min_cell_area = 1,
  max_cell_area = 5000,
  iterations = 300,
  invert = TRUE,
  block_size_adaptive = 25,
  edge_weight = 1.0,
  min_distance_peaks = 3,
  outline_thickness = 2
)

# View the dataframe of counts
print(cell_counts)
