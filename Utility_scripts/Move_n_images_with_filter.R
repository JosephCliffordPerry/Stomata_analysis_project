library(magick)
library(fs)

# -------------------------------
# PARAMETERS
# -------------------------------
input_dir  <- "E:/Stomata_maize/all_images/all_images/crops"
output_dir <- "E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct"
N <- 1000  # number of valid images required

dir_create(output_dir)

# -------------------------------
# GET IMAGE FILES
# -------------------------------
img_files <- dir_ls(input_dir, regexp = "\\.(jpg|jpeg|png|tif|tiff)$", recurse = FALSE)

# randomise order once
set.seed(123)
img_files <- sample(img_files)

# -------------------------------
# VALIDATION FUNCTION
# -------------------------------
is_valid_image <- function(f) {
  info <- tryCatch(image_info(image_read(f)), error = function(e) NULL)
  if (is.null(info)) return(FALSE)
  
  width  <- info$width
  height <- info$height
  
  if (height == 0) return(FALSE)
  
  aspect_ratio <- width / height
  area <- width * height
  
  return(
    aspect_ratio >= 1 &&
      aspect_ratio <= 2.8 &&
      width >= 30 &&
      area >= 3500 &&
      width<= 200
  )
}

# -------------------------------
# ITERATIVE SAMPLING
# -------------------------------
selected_files <- character(0)

for (f in img_files) {
  if (length(selected_files) >= N) break
  
  if (is_valid_image(f)) {
    selected_files <- c(selected_files, f)
  }
}

# -------------------------------
# CHECK IF ENOUGH FOUND
# -------------------------------
if (length(selected_files) < N) {
  warning(sprintf("Only %d valid images found (requested %d).", 
                  length(selected_files), N))
}

# -------------------------------
# MOVE FILES
# -------------------------------
file_move(selected_files, file.path(output_dir, path_file(selected_files)))

# -------------------------------
# SUMMARY
# -------------------------------
cat("Images moved:", length(selected_files), "\n")

library(fs)

# -------------------------------
# =Just random images
# -------------------------------


# -------------------------------
# GET IMAGE FILES
# -------------------------------
img_files <- dir_ls(input_dir, regexp = "\\.(jpg|jpeg|png|tif|tiff)$", recurse = FALSE)

# -------------------------------
# RANDOM SELECTION
# -------------------------------
set.seed(123)  # reproducibility

selected_files <- sample(img_files, size = min(N, length(img_files)), replace = FALSE)

# -------------------------------
# MOVE FILES
# -------------------------------
file_move(selected_files, file.path(output_dir, path_file(selected_files)))

# -------------------------------
# SUMMARY
# -------------------------------
cat("Images moved:", length(selected_files), "\n")
