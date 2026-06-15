library(reticulate)
library(imager)
library(geometry)

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
# allow numpy loading
np <- import("numpy")

mask_dir  <- "E:/Stomata_maize/all_images/test_images/test_filtered_crops/overlays"
img_dir   <- "E:/Stomata_maize/all_images/test_images/test_filtered_crops"
out_dir   <- file.path(img_dir, "aligned")
dir.create(out_dir, showWarnings = FALSE)

mask_files <- list.files(mask_dir, pattern = "_mask.npy$", full.names = TRUE)

# function to find longest line on mask boundary
longest_edge_line <- function(mask){
  
  pts <- which(mask == TRUE, arr.ind = TRUE)
  
  # convex hull of mask pixels
  hull_idx <- chull(pts[,2], pts[,1])
  hull_pts <- pts[hull_idx,]
  
  # compute pairwise distances
  d <- as.matrix(dist(hull_pts))
  
  pair <- which(d == max(d), arr.ind = TRUE)[1,]
  
  p1 <- hull_pts[pair[1],]
  p2 <- hull_pts[pair[2],]
  
  list(p1 = p1, p2 = p2)
}

# compute rotation angle
line_angle <- function(p1, p2){
  dx <- p2[2] - p1[2]
  dy <- p2[1] - p1[1]
  atan2(dy, dx) * 180/pi
}

for(mask_file in mask_files){
  
  cat("Processing:", mask_file, "\n")
  
  mask <- np$load(mask_file)
  mask <- py_to_r(mask)
  
  # get image path
  base <- sub("_mask.npy","", basename(mask_file))
  img_path <- file.path(img_dir, paste0(base,".png"))
  
  if(!file.exists(img_path)){
    img_path <- file.path(img_dir, paste0(base,".tif"))
  }
  
  if(!file.exists(img_path)){
    next
  }
  
  img <- load.image(img_path)
  
  # find longest boundary line
  line <- longest_edge_line(mask)
  
  angle <- line_angle(line$p1, line$p2)
  
  # rotate image to align line horizontally
  aligned <- imrotate(img, -angle)
  
  out_path <- file.path(out_dir, paste0(base, "_aligned.png"))
  save.image(aligned, out_path)
  
}
