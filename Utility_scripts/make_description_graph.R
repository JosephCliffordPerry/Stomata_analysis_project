library(tidyverse)
library(ggplot2)
library(jpeg)
library(png)
library(tiff)

# --------------------------------------------------
# INPUT FOLDERSA
# --------------------------------------------------
image_folder  <- "D:/stomata/November_image_data_8bit_tifs"
label_folder  <- "D:/stomata/November_partial_annotations"
output_folder <- "D:/stomata/November_partial_annotations_graphs"

dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)


# --------------------------------------------------
# --------------------------------------------------
read_yolo_segmentation <- function(file_path) {
  lines <- readLines(file_path)
  split <- strsplit(lines, " ")
  
  polys <- lapply(seq_along(split), function(i) {
    nums <- as.numeric(split[[i]])
    cls <- nums[1]
    coords <- nums[-1]
    
    xs <- coords[seq(1, length(coords), 2)]
    ys <- coords[seq(2, length(coords), 2)]
    
    tibble(
      polygon_id = i,
      class_id = cls,
      x_scaled = xs,
      y_scaled = ys
    )
  })
  
  bind_rows(polys)
}

# --------------------------------------------------
# Function to plot polygons
# --------------------------------------------------
library(ggplot2)
library(grid)
library(dplyr)
library(tiff)

# --------------------------------------------------
# Function to plot polygons with auto-scaled & inverted coordinates
# --------------------------------------------------
plot_polys <- function(image_path, poly_df, out_path) {
  
  # read the image (TIFF)
  img <- tryCatch({
    suppressWarnings(tiff::readTIFF(image_path))
  }, error = function(e) NULL)
  
  if (is.null(img)) {
    message("⚠ Could not read image: ", image_path)
    return(NULL)
  }
  
  # image dimensions
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  # scale & invert polygon coordinates
  poly_df <- poly_df %>%
    mutate(
      x_scaled = x_scaled * w,           # scale X to image width
      y_scaled = h - (y_scaled * h)      # scale & invert Y to match image origin top-left
    )
  
  # convert image to raster grob
  g <- rasterGrob(
    img,
    width = unit(1, "npc"),
    height = unit(1, "npc"),
    interpolate = FALSE
  )
  
  # assign random colors to polygons
  poly_df$col <- sample(colors(), n_distinct(poly_df$polygon_id), replace = TRUE)[poly_df$polygon_id]
  
  # plot
  p <- ggplot(poly_df, aes(x = x_scaled, y = y_scaled, group = polygon_id, fill = col)) +
    annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
    geom_polygon(color = "black", linewidth = 0.2, alpha = 0.4) +
    scale_fill_identity() +
    coord_equal() +
    theme_void()
  
  # save output
  ggsave(out_path, plot = p, width = 8, height = 8, dpi = 300)
}

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
image_files <- list.files(image_folder, pattern = "\\.tif$", full.names = FALSE)

for (img_name in image_files) {
  
  message("Processing: ", img_name)
  
  # expected label name
  label_name <- sub("\\.tif$", "_all_cells.txt", img_name)
  label_path <- file.path(label_folder, label_name)
  
  # check
  if (!file.exists(label_path)) {
    message("  ❗ Label missing: ", label_name)
    next
  }
  
  # read polygons
  poly_df <- read_yolo_segmentation(label_path)
  
  # output plot file
  out_png <- file.path(output_folder, sub("\\.tif$", ".png", img_name))
  
  # full image path
  img_path <- file.path(image_folder, img_name)
  
  # plot
  plot_polys(img_path, poly_df, out_png)
  
  message("  ✔ Saved plot: ", out_png)
}

message("Done.")
