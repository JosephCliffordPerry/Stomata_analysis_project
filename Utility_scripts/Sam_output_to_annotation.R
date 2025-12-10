library(fs)
library(purrr)
library(sf)

# ======================================================
#  GLOBAL AREA STORAGE
# ======================================================

all_areas <<- c()

# ======================================================
#  GEOMETRY SUPPORT
# ======================================================

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
  area <- polygon_area(df$x, df$y)             # <-- already pixel units
  perimeter <- polygon_perimeter(df$x, df$y)
  circularity <- polygon_circularity(area, perimeter)
  list(area = area, perimeter = perimeter, circularity = circularity)
}

poly_to_sfg_safe <- function(p){
  coords <- p$segmentation
  if(nrow(coords) < 3) return(NULL)
  if(!all(coords[1,] == coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
  st_polygon(list(as.matrix(coords)))
}

bbox_intersect_fast <- function(a, b){
  return(!(a$xmax<a$xmin || b$xmax<b$xmin ||
             a$ymax<b$ymin || b$ymax<a$ymin))
}

get_bbox <- function(df){
  list(xmin = min(df$x), xmax = max(df$x),
       ymin = min(df$y), ymax = max(df$y))
}

# ======================================================
#  READ YOLO FILE
#  (AREA STORED AND FILTERED IN PIXELS)
# ======================================================

read_yolo_file <- function(file_path, image_width, image_height,
                           min_area, max_area, min_circ, max_circ) {
  
  lines <- readLines(file_path, warn = FALSE)
  parsed <- strsplit(lines, "\\s+")
  
  polys <- lapply(parsed, function(vals){
    vals <- as.numeric(vals)
    if(length(vals) < 3) return(NULL)
    
    class_id <- vals[1]
    coords <- vals[-1]
    
    df <- data.frame(
      x = coords[seq(1, length(coords), by = 2)] * image_width,
      y = coords[seq(2, length(coords), by = 2)] * image_height
    )
    
    geom <- recompute_geometry(df)
    pixel_area <- geom$area     # <-- always pixel²
    
   
    # filter using pixel units
    if(pixel_area < min_area) return(NULL)
    if(pixel_area > max_area) return(NULL)
    if(geom$circularity < min_circ) return(NULL)
    if(geom$circularity > max_circ) return(NULL)
    
    # store area globally
    all_areas <<- c(all_areas, pixel_area)
    
    list(
      class_id = class_id,
      segmentation = df,
      area = pixel_area,
      perimeter = geom$perimeter,
      circularity = geom$circularity
    )
  })
  
  Filter(Negate(is.null), polys)
}

# ======================================================
# FAST CENTROID MERGE
# ======================================================

centroid_merge <- function(polys, multiplier = 0.25) {
  if(length(polys) < 2) return(polys)
  
  compute_centroid <- function(df) c(mean(df$x), mean(df$y))
  compute_diameter <- function(area) 2 * sqrt(area / pi)
  
  remaining <- seq_along(polys)
  merged <- list()
  id_counter <- 1
  
  while(length(remaining) > 0) {
    i <- remaining[1]
    p <- polys[[i]]
    cen_p <- compute_centroid(p$segmentation)
    dia_p <- compute_diameter(p$area)
    thresh <- dia_p * multiplier
    
    to_merge <- c(i)
    for(j in remaining[-1]) {
      q <- polys[[j]]
      cen_q <- compute_centroid(q$segmentation)
      dist <- sqrt((cen_p[1] - cen_q[1])^2 + (cen_p[2] - cen_q[2])^2)
      if(dist < thresh) to_merge <- c(to_merge, j)
    }
    
    merged_seg <- do.call(rbind, lapply(to_merge, function(k) polys[[k]]$segmentation))
    
    merged[[id_counter]] <- list(
      class_id = polys[[to_merge[1]]]$class_id,
      segmentation = merged_seg
    )
    
    id_counter <- id_counter + 1
    remaining <- setdiff(remaining, to_merge)
  }
  
  merged
}

# ======================================================
# SMALL SHAPES TAKE PRIORITY
# ======================================================

trim_intersecting_large_shapes <- function(polys, trim_threshold = 3000) {
  
  safe_extract_polygon <- function(geom) {
    geom <- st_make_valid(geom)
    cls <- unique(st_geometry_type(geom))
    if (all(cls %in% c("POLYGON", "MULTIPOLYGON"))) {
      return(geom)
    }
    geom <- st_collection_extract(geom, "POLYGON")
    return(geom)
  }
  
  to_sfg <- function(p) {
    poly <- poly_to_sfg_safe(p)
    if (is.null(poly)) return(st_geometrycollection())
    poly
  }
  
  small <- polys[sapply(polys, function(p) p$area <= trim_threshold)]
  large <- polys[sapply(polys, function(p) p$area > trim_threshold)]
  
  small_sf <- st_sfc(lapply(small, to_sfg))
  large_sf <- st_sfc(lapply(large, to_sfg))
  
  small_sf <- st_make_valid(small_sf)
  large_sf <- st_make_valid(large_sf)
  
  trimmed_large <- list()
  
  for(i in seq_along(large)) {
    
    big <- large[[i]]
    big_geom <- large_sf[i]
    
    big_geom <- safe_extract_polygon(big_geom)
    if (all(st_is_empty(big_geom))) next
    
    for(j in seq_along(small)) {
      
      s_geom <- small_sf[j]
      if (st_is_empty(s_geom)) next
      
      hit <- try(st_intersects(s_geom, big_geom, sparse = FALSE)[1,1], silent = TRUE)
      if (inherits(hit, "try-error") || !isTRUE(hit)) next
      
      big_geom <- suppressWarnings(try(st_difference(big_geom, s_geom), silent = TRUE))
      if (inherits(big_geom, "try-error")) break
      
      big_geom <- st_make_valid(big_geom)
      big_geom <- safe_extract_polygon(big_geom)
      if (all(st_is_empty(big_geom))) break
    }
    
    if (all(st_is_empty(big_geom))) next
    
    parts <- st_cast(big_geom, "POLYGON", warn = FALSE)
    
    for(k in seq_along(parts)) {
      coords <- st_coordinates(parts[k])
      
      if ("L2" %in% colnames(coords)) {
        ring <- coords[coords[,"L2"] == coords[1,"L2"], ]
      } else {
        ring <- coords
      }
      
      if(nrow(ring) < 3) next
      
      seg_df <- data.frame(x = ring[,1], y = ring[,2])
      geom <- recompute_geometry(seg_df)
      
      trimmed_large <- append(trimmed_large, list(
        list(
          class_id = big$class_id,
          segmentation = seg_df,
          area = geom$area,
          perimeter = geom$perimeter,
          circularity = geom$circularity
        )
      ))
    }
  }
  
  c(small, trimmed_large)
}

# ======================================================
# FINAL POSTPROCESS (CONVEX HULL)
# ======================================================

final_postprocess <- function(polys) {
  lapply(polys, function(p) {
    hull <- convex_hull(p$segmentation)
    geom <- recompute_geometry(hull)
    list(
      class_id = p$class_id,
      segmentation = hull,
      area = geom$area,
      perimeter = geom$perimeter,
      circularity = geom$circularity
    )
  })
}

# ======================================================
# EXPORT (NORMALIZED OUTPUT)
# ======================================================

export_yolov8 <- function(polys, out_file, image_width, image_height) {
  
  if(length(polys) == 0){
    file.create(out_file)
    return()
  }
  
  lines <- map_chr(polys, function(p) {
    xs <- p$segmentation$x / image_width
    ys <- p$segmentation$y / image_height
    coords <- paste(c(rbind(xs, ys)), collapse = " ")
    paste0(p$class_id, " ", coords)
  })
  
  writeLines(lines, con = out_file)
}

# ======================================================
# PROCESS ONE FILE
# ======================================================

process_file <- function(input_file, output_file, image_width, image_height,
                         min_area, max_area, min_circ, max_circ, centroid_mult) {
  
  polys <- read_yolo_file(input_file, image_width, image_height,
                          min_area, max_area, min_circ, max_circ)
  
  if(length(polys) == 0){
    file.create(output_file)
    return()
  }
  
  merged <- centroid_merge(polys, centroid_mult)
  post <- final_postprocess(merged)
  final <- trim_intersecting_large_shapes(post, trim_threshold = 3000)
  export_yolov8(final, output_file, image_width, image_height)
}

# ======================================================
# PROCESS FOLDER
# ======================================================

process_folder <- function(input_folder, output_folder, image_width, image_height,
                           min_area = 500, max_area = 5000,
                           min_circ = 0.30, max_circ = 1,
                           centroid_mult = 0.1){
  
  dir_create(output_folder)
  
  files <- list.files(input_folder, pattern = "\\.txt$", full.names = TRUE)
  
  for(f in files){
    out_f <- file.path(output_folder, basename(f))
    cat("Processing: ", basename(f), "\n")
    process_file(f, out_f,
                 image_width, image_height,
                 min_area, max_area,
                 min_circ, max_circ,
                 centroid_mult)
  }
}

# ======================================================
# RUN
# ======================================================

process_folder(
  input_folder = "D:/stomata/November_batch_outputs",
  output_folder = "D:/stomata/November_partial_annotations",
  image_width = 1002,
  image_height = 1004
)

cat("\nArea summary (pixel units):\n")
print(summary(all_areas))

hist(all_areas, main="Object Areas", xlab="Pixels²", ylab="Count")
