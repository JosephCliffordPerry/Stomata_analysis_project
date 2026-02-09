# ============================================================
# Binary PNG masks -> YOLOv8 segmentation labels ONLY
# using sf + convex hull
# ============================================================

library(png)
library(sf)

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
mask_dir <- "E:/Stomatahub_datasets/Bean"
out_dir  <- "labels"
class_id <- 0

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# Convert mask to polygons using convex hull
# ------------------------------------------------------------
mask_to_polygons_sf <- function(mask) {
  
  # convert mask to integer 0/1
  mask_bin <- ifelse(mask < 1, 1L, 0L)
  
  # get foreground pixel coordinates
  fg <- which(mask_bin > 0, arr.ind = TRUE)
  if (nrow(fg) == 0) return(st_sf(geometry = st_sfc()))
  
  # convert to sf points
  pts <- st_as_sf(
    data.frame(x = fg[, "col"], y = fg[, "row"]),
    coords = c("x", "y"),
    crs = NA
  )
  
  # optional: cluster points to separate objects using 8-connectivity
  # convert point coords to adjacency
  # for simplicity, we buffer by 1 pixel and union touching points
  pts_buffer <- st_buffer(pts, 1)
  merged <- st_union(pts_buffer)
  
  # cast to polygons (may be multiple objects)
  polys <- st_cast(merged, "POLYGON")
  
  # compute convex hull for each polygon
  polys <- st_convex_hull(polys)
  
  st_sf(geometry = polys)
}

# ------------------------------------------------------------
# sf polygon -> YOLO segmentation
# ------------------------------------------------------------
sf_poly_to_yolo <- function(poly, w, h) {
  
  poly <- st_cast(poly, "POLYGON", warn = FALSE)
  coords <- st_coordinates(poly)[, c("X","Y")]
  
  if (nrow(coords) < 3) return(NULL)
  
  # close the polygon
  if (!all(coords[1,] == coords[nrow(coords),])) {
    coords <- rbind(coords, coords[1,])
  }
  
  as.vector(cbind(coords[,1]/w, coords[,2]/h))
}

write_yolo_labels <- function(polys, path, w, h) {
  
  if (nrow(polys) == 0) {
    file.create(path)
    return()
  }
  
  lines <- character(0)
  
  for (i in seq_len(nrow(polys))) {
    v <- sf_poly_to_yolo(polys[i,], w, h)
    if (is.null(v) || length(v) < 6) next
    lines <- c(lines, paste(class_id, paste(sprintf("%.6f", v), collapse = " ")))
  }
  
  writeLines(lines, path)
}

# ------------------------------------------------------------
# MAIN LOOP WITH PROGRESS BAR
# ------------------------------------------------------------
mask_paths <- list.files(mask_dir, "\\.png$", full.names = TRUE)

pb <- txtProgressBar(min = 0, max = length(mask_paths), style = 3)

for (i in seq_along(mask_paths)) {
  
  mask_path <- mask_paths[i]
  
  base <- tools::file_path_sans_ext(basename(mask_path))
  out_label <- file.path(out_dir, paste0(base, ".txt"))
  
  mask <- readPNG(mask_path)
  if (length(dim(mask)) == 3) mask <- mask[, , 1]
  
  h <- nrow(mask)
  w <- ncol(mask)
  
  polys <- mask_to_polygons_sf(mask)
  
  write_yolo_labels(polys, out_label, w, h)
  
  setTxtProgressBar(pb, i)
}

close(pb)
