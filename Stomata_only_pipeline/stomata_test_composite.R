library(tidyverse)
library(ggplot2)
library(jpeg)
library(png)
library(grid)
library(patchwork)

# --------------------------------------------------
# INPUT FOLDERS
# --------------------------------------------------
image_folder  <- "D:/stomata/November_images/November_stomata_crops"
label_folder  <- "D:/stomata/November_images/November_sam_multi_guard_annotations"
output_folder <- "D:/stomata/November_images/November_sam_multi_guard_graphs"
dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------
# Read YOLO segmentation
# --------------------------------------------------
read_yolo_segmentation <- function(file_path) {
  lines <- readLines(file_path)
  split <- strsplit(lines, " ")
  
  polys <- lapply(seq_along(split), function(i) {
    nums <- suppressWarnings(as.numeric(split[[i]]))
    if(any(is.na(nums))) return(NULL)
    
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
# Plot single image + polygons (PNG version)
# --------------------------------------------------
plot_polys_single <- function(image_path, poly_df) {
  
  # try loading PNG — if fails, return NULL so it doesn’t crash composites
  img <- tryCatch({
    suppressWarnings(png::readPNG(image_path))
  }, error = function(e) NULL)
  
  if (is.null(img)) {
    return(ggplot() + ggtitle("IMAGE READ FAIL"))
  }
  
  # image dims
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  # if no polygons, just return the image
  if (nrow(poly_df) == 0) {
    g <- rasterGrob(
      img,
      width = unit(1, "npc"),
      height = unit(1, "npc"),
      interpolate = FALSE
    )
    return(
      ggplot() +
        annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
        coord_equal() +
        theme_void()
    )
  }
  
  # scale & invert
  poly_df <- poly_df %>%
    mutate(
      x_scaled = x_scaled * w,
      y_scaled = h - (y_scaled * h)
    )
  
  g <- rasterGrob(
    img,
    width = unit(1, "npc"),
    height = unit(1, "npc"),
    interpolate = FALSE
  )
  
  poly_df$col <- sample(colors(), n_distinct(poly_df$polygon_id), replace = TRUE)[poly_df$polygon_id]
  
  ggplot(poly_df, aes(x = x_scaled, y = y_scaled, group = polygon_id, fill = col)) +
    annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
    geom_polygon(color = "black", linewidth = 0.25, alpha = 0.4) +
    scale_fill_identity() +
    coord_equal() +
    theme_void()
}


# --------------------------------------------------
# MAIN — Build 3 composites of 8 images each
# --------------------------------------------------
image_files <- list.files(image_folder, pattern = "\\.png$", full.names = FALSE)

# only keep those that have labels
valid_files <- image_files[
  file.exists(file.path(label_folder, sub("\\.png$", "\\.txt", image_files)))
]

# random selection
set.seed(7)
composites <- list()

for (comp_i in 1:3) {
  
  selected_imgs <- sample(valid_files, 8)
  
  message("Building composite ", comp_i)
  
  plot_list <- lapply(selected_imgs, function(img_name) {
    label_name <- sub("\\.png$", ".txt", img_name)
    label_path <- file.path(label_folder, label_name)
    img_path <- file.path(image_folder, img_name)
    
    poly_df <- suppressWarnings(read_yolo_segmentation(label_path))
    
    p <- plot_polys_single(img_path, poly_df) +
      ggtitle(img_name) +
      theme(plot.title = element_text(size = 8))
    
    return(p)
  })
  
  composite_plot <- wrap_plots(plot_list, ncol = 4)
  
  # store
  composites[[comp_i]] <- composite_plot
  
  # also save PNG
  ggsave(
    filename = file.path(output_folder, paste0("composite_", comp_i, ".png")),
    plot = composite_plot,
    width = 12,
    height = 6,
    dpi = 300
  )
}

# final returned object
composites
