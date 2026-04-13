source("Stomata_decomposition_model_training_pipeline/SAM3_analysis_to_companion_cells.R")
source("Stomata_decomposition_model_training_pipeline/SAM3_analysis_optimised.R")
library(reticulate)

Sys.setenv(RETICULATE_PYTHON = "managed")

py_require(
  packages = c(
    "numpy",
    "opencv-python",
    "matplotlib",
    "scikit-image",
    "ultralytics",
    "torch" ,
    "torchvision" ,
    "torchaudio" ,
    "segment-anything@git+https://github.com/facebookresearch/segment-anything.git"
  ),
  python_version = "3.12.4"
)

find_stomatal_complex(
  img_dir = "E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct",
  overlay_dir = "E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct/overlays",
  min_ratio = 0.4,
  max_ratio = 0.8
)
df <- find_companion_cells(
  mask_dir = "E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct/overlays",
  img_dir  = "E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct"
)
