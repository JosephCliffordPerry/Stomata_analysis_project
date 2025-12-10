#reads and filters annotations 

# ---- Helper function: compute polygon area ----
polygon_area <- function(x, y) {
  # Shoelace formula
  0.5 * abs(sum(x * c(tail(y, -1), y[1])) - sum(y * c(tail(x, -1), x[1])))
}

# ---- Helper function: compute polygon perimeter ----
polygon_perimeter <- function(x, y) {
  sum(sqrt(diff(c(x, x[1]))^2 + diff(c(y, y[1]))^2))
}

# ---- Function to read YOLOv8 format  with area + circularity thresholds ----
read_yolo_segmentation_file <- function(file_path, 
                                        image_width = NULL, 
                                        image_height = NULL,
                                        min_area = 0, 
                                        max_area = Inf,
                                        min_circularity = 0, 
                                        max_circularity = 1) {
  # Read all lines
  lines <- readLines(file_path)
  parsed <- strsplit(lines, "\\s+")
  
  polygons <- lapply(seq_along(parsed), function(i) {
    vals <- as.numeric(parsed[[i]])
    class_id <- vals[1]
    coords <- vals[-1]
    
    seg_df <- data.frame(
      x = coords[seq(1, length(coords), by = 2)],
      y = coords[seq(2, length(coords), by = 2)]
    )
    
    # Denormalize if dimensions provided
    if (!is.null(image_width) && !is.null(image_height)) {
      seg_df$x <- seg_df$x * image_width
      seg_df$y <- seg_df$y * image_height
    }
    
    # Compute polygon area and perimeter
    area <- polygon_area(seg_df$x, seg_df$y)
    perimeter <- polygon_perimeter(seg_df$x, seg_df$y)
    
    # Avoid division by zero
    circularity <- if (perimeter > 0) (4 * pi * area) / (perimeter^2) else 0
    
    list(
      id = i,
      class_id = class_id,
      segmentation = seg_df,
      area = area,
      perimeter = perimeter,
      circularity = circularity
    )
  })
  
  # Filter polygons based on thresholds
  filtered_polygons <- Filter(function(p) {
    p$area >= min_area && 
      p$area <= max_area &&
      p$circularity >= min_circularity &&
      p$circularity <= max_circularity
  }, polygons)
  
  return(filtered_polygons)
}

# ---- Example usage ----
# Reject polygons smaller than 100 px² or larger than 10,000 px²
# Keep only shapes with circularity between 0.3 and 1.0
# (Assuming image width=1002, height=1004)

yolo_polygons <- read_yolo_segmentation_file(
  "grid_of_4_cells_all.txt",
  image_width = 1002,
  image_height = 1004,
  min_area = 300,
  max_area = 10000,
  min_circularity = 0.1,
  max_circularity = 1.0
)

# ---- Summary ----
cat("Polygons kept:", length(yolo_polygons), "\n")
# cat("Areas:", sapply(yolo_polygons, `[[`, "area"), "\n")
# cat("Circularities:", round(sapply(yolo_polygons, `[[`, "circularity"), 3), "\n")

# ---- Plot results ----
library(ggplot2)
library(purrr)
library(dplyr)

library(grid)
library(png)
yolo_df <- map_df(yolo_polygons, function(poly) {
  seg <- poly$segmentation
  seg$polygon_id <- poly$id
  seg$class_id <- poly$class_id
  seg$circularity <- poly$circularity
  seg
})
# 
# ggplot(yolo_df, aes(x = x, y = y, group = polygon_id, fill = factor(class_id))) +
#   geom_polygon(alpha = 0.4, color = "black") +
#   coord_equal() +
#   theme_minimal() +
#   labs(title = "Filtered SAM Segmentations (Area + Circularity)")
# 

img <- readPNG("D:/stomata/sam_overlays/overlay_0715.png")
img_width <- dim(img)[2]
img_height <- dim(img)[1]
yolo_df <- yolo_df %>%
  mutate(
    x_scaled = x / 1002 * img_width,
    y_scaled = y / 1004 * img_height
  )
yolo_df <- yolo_df %>%
  mutate(
    x_scaled = x / 1002 * img_width,
    y_scaled = img_height - (y / 1004 * img_height)  # invert y
  )



g <- rasterGrob(img, width = unit(1,"npc"), height = unit(1,"npc"))

ggplot(yolo_df, aes(x = x_scaled, y = y_scaled, group = polygon_id, fill = factor(class_id))) +
  annotation_custom(g, xmin = 0, xmax = img_width, ymin = 0, ymax = img_height) +
  geom_polygon(alpha = 0.4, color = "black") +
  coord_equal() +
  theme_minimal() +
  labs(title = "Filtered SAM Segmentations (Area + Circularity)")













