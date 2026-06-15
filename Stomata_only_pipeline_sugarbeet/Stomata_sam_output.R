library(fs)
library(purrr)
library(sf)
library(magick)

all_areas <<- c()

# -----------------------
# Geometry functions
# -----------------------
get_image_dimensions <- function(image_path) {
  img <- image_read(image_path)
  info <- image_info(img)
  list(width = info$width, height = info$height)
}

polygon_area <- function(x, y) {
  0.5 * abs(sum(x * c(tail(y, -1), y[1])) - sum(y * c(tail(x, -1), x[1])))
}

polygon_perimeter <- function(x, y) {
  sum(sqrt(diff(c(x, x[1]))^2 + diff(c(y, y[1]))^2))
}

polygon_circularity <- function(area, perimeter) {
  if(perimeter > 0) 4*pi*area / perimeter^2 else 0
}

convex_hull <- function(df) {
  idx <- chull(df$x, df$y)
  df[idx, ]
}

recompute_geometry <- function(df) {
  area <- polygon_area(df$x, df$y)
  perimeter <- polygon_perimeter(df$x, df$y)
  circ <- polygon_circularity(area, perimeter)
  list(area = area, perimeter = perimeter, circularity = circ)
}

# -----------------------
# Class filtering
# -----------------------
class_params <- list(
  `0` = list(min_area = 0, max_area = 500, min_circ = 0.5, max_circ = 1),
  `1` = list(min_area = 0, max_area = 1000, min_circ = 0.5, max_circ = 1)
)

# -----------------------
# Read YOLO file
# -----------------------
read_yolo_file <- function(file_path, image_width, image_height, class_params) {
  lines <- readLines(file_path, warn = FALSE)
  parsed <- strsplit(lines, "\\s+")
  
  polys <- lapply(parsed, function(vals) {
    vals <- as.numeric(vals)
    if(length(vals) < 3) return(NULL)
    
    class_id <- as.character(vals[1])
    coords <- vals[-1]
    
    df <- data.frame(
      x = coords[seq(1, length(coords), by = 2)] * image_width,
      y = coords[seq(2, length(coords), by = 2)] * image_height
    )
    
    geom <- recompute_geometry(df)
    p <- class_params[[class_id]]
    
    if(is.null(p)) return(NULL)
    if(geom$area < p$min_area) return(NULL)
    if(geom$area > p$max_area) return(NULL)
    if(geom$circularity < p$min_circ) return(NULL)
    if(geom$circularity > p$max_circ) return(NULL)
    
    all_areas <<- c(all_areas, geom$area)
    
    list(
      class_id = class_id,
      segmentation = df,
      area = geom$area,
      perimeter = geom$perimeter,
      circularity = geom$circularity,
      img_w = image_width,
      img_h = image_height
    )
  })
  
  Filter(Negate(is.null), polys)
}

# -----------------------
# Merge polygons by class
# -----------------------
merge_polygons_by_class <- function(polys) {
  if(length(polys) == 0) return(list())
  
  merged_polys <- list()
  
  for(cls in unique(sapply(polys, `[[`, "class_id"))) {
    cls_polys <- polys[sapply(polys, `[[`, "class_id") == cls]
    
    # Convert to sf objects, ensure polygons are closed
    sf_list <- lapply(cls_polys, function(p) {
      coords <- as.matrix(p$segmentation)
      if(nrow(coords) < 3) return(NULL)
      if(!all(coords[1,] == coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
      st_sf(class_id = p$class_id, geometry = st_sfc(st_polygon(list(coords))))
    })
    
    sf_list <- Filter(Negate(is.null), sf_list)
    if(length(sf_list) == 0) next
    
    sf_combined <- do.call(rbind, sf_list)
    sf_combined <- st_make_valid(sf_combined)
    merged_geom <- st_union(sf_combined)
    
    # Convert merged polygons back to list format
    geoms <- if(inherits(merged_geom, "sfc_GEOMETRYCOLLECTION")) merged_geom else list(merged_geom)
    
    for(g in geoms) {
      hull <- st_convex_hull(g)
      coords <- st_coordinates(hull)[,1:2]
      if(!all(coords[1,] == coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
      df <- data.frame(x = coords[,1], y = coords[,2])
      geom <- recompute_geometry(df)
      merged_polys[[length(merged_polys)+1]] <- list(
        class_id = cls_polys[[1]]$class_id,
        segmentation = df,
        area = geom$area,
        perimeter = geom$perimeter,
        circularity = geom$circularity,
        img_w = cls_polys[[1]]$img_w,
        img_h = cls_polys[[1]]$img_h
      )
    }
  }
  
  merged_polys
}

# -----------------------
# Postprocess: convex hull + merge
# -----------------------
final_postprocess <- function(polys) {
  polys <- lapply(polys, function(p) {
    hull <- convex_hull(p$segmentation)
    geom <- recompute_geometry(hull)
    list(
      class_id = p$class_id,
      segmentation = hull,
      area = geom$area,
      perimeter = geom$perimeter,
      circularity = geom$circularity,
      img_w = p$img_w,
      img_h = p$img_h
    )
  })
  
  merge_polygons_by_class(polys)
}

# -----------------------
# Export YOLOv8 txt
# -----------------------
export_yolov8 <- function(polys, out_file) {
  if(length(polys) == 0){
    file.create(out_file)
    return()
  }
  
  lines <- map_chr(polys, function(p) {
    xs <- p$segmentation$x / p$img_w
    ys <- p$segmentation$y / p$img_h
    coords <- paste(c(rbind(xs, ys)), collapse = " ")
    paste0(p$class_id, " ", coords)
  })
  
  writeLines(lines, con = out_file)
}

# -----------------------
# Process single file
# -----------------------
process_file <- function(input_file, output_file, class_params) {
  base_name <- fs::path_ext_remove(input_file)
  img_path_candidates <- c(
    paste0(base_name, ".jpg"),
    paste0(base_name, ".png"),
    paste0(base_name, ".jpeg"),
    paste0(base_name, ".tif"),
    paste0(base_name, ".tiff")
  )
  
  img_path <- img_path_candidates[file.exists(img_path_candidates)][1]
  if(is.na(img_path)) stop(paste("ERROR: No matching image file found for", input_file))
  
  dims <- get_image_dimensions(img_path)
  polys <- read_yolo_file(input_file, dims$width, dims$height, class_params)
  if(length(polys) == 0){
    file.create(output_file)
    return()
  }
  
  final <- final_postprocess(polys)
  export_yolov8(final, output_file)
}

# -----------------------
# Process folder
# -----------------------
process_folder <- function(input_folder, output_folder, class_params) {
  dir_create(output_folder)
  files <- list.files(input_folder, pattern = "\\.txt$", full.names = TRUE)
  
  for(f in files) {
    cat("Processing:", basename(f), "\n")
    out_f <- file.path(output_folder, basename(f))
    process_file(f, out_f, class_params)
  }
}

# -----------------------
# RUN PIPELINE
# -----------------------
process_folder(
  input_folder = "D:/stomata/November_images/Crop_and_label",
  output_folder = "D:/stomata/November_images/November_sam_multi_guard_annotations",
  class_params = class_params
)

cat("\nArea summary:\n")
print(summary(all_areas))
