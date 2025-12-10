# ============================================
# Cellpose 4 segmentation from R using reticulate
# ============================================

# Load required R packages
if (!require("reticulate")) install.packages("reticulate")
if (!require("tiff")) install.packages("tiff")

library(reticulate)
library(tiff)

# --- 1. Environment setup ---
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("cellpose==4.*", "numpy", "tifffile")
)

# --- 2. Import Python modules ---
np <- import("numpy")
models <- import("cellpose.models")
io <- import("cellpose.io")
tifffile <- import("tifffile")

# --- 3. Load an 8-bit TIFF image ---
input_tif <- "D:/stomata/training_tifs_8bit/D_T2R2_Ab_15X_frame_0002.tif"
output_mask <- "segmented_mask.tif"

img_r <- readTIFF(input_tif)
img_np <- np$array(img_r)

# --- 4. Create Cellpose 4 model and run segmentation ---
model <- models$CellposeModel(model_type = "cyto")

result <- model$eval(
  x = list(img_np),
  channels = c(0, 0),       # grayscale
  diameter = NULL,          # auto-detect
  batch_size = 1,
  flow_threshold = 0.4,
  cellprob_threshold = 0.0
)

# result is a list: [masks, flows, styles, diams]


# Display the original image
image(results$original, col = gray((0:255)/255), axes = FALSE)
image(results$binary, col = gray((0:255)/255), axes = FALSE)
image(results$combined, col = gray((0:255)/255), axes = FALSE)
image(results$enhanced, col = gray((0:255)/255), axes = FALSE)
image(results$closed, col = gray((0:255)/255), axes = FALSE)

library(imager)

img_cimg <- as.cimg(results$overlay, dims = c(1002, 1004, 3))
plot(img_cimg)

