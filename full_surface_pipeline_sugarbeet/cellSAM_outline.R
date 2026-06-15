# ============================================================
# Batch tiled YOLOv8 segmentation export from .npy masks
# ============================================================

source("full_surface_pipeline/full_surface_testing.R")

library(sf)
library(raster)
library(terra)
library(jpeg)
library(png)
library(tiff)
library(reticulate)

# ============================================================
# PARAMETERS
# ============================================================

image_dir <- "E:/Stomata/Sugarbeet_stomata_imaging/mips2"
mask_dir  <- "E:/Stomata/Sugarbeet_stomata_imaging/masks (6)/content/outputs"

tile_size    <- 128
min_coverage <- 0.7

out_root <- "yolo_dataset"
img_dir  <- file.path(out_root, "images")
lbl_dir  <- file.path(out_root, "labels")

dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(lbl_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Python / NumPy
# ============================================================

Sys.setenv(RETICULATE_PYTHON = "managed")
np <- import("numpy")

# ============================================================
# IMAGE HELPERS
# ============================================================

read_image_any <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("tif","tiff")) tiff::readTIFF(path)
  else if (ext %in% c("jpg","jpeg")) jpeg::readJPEG(path)
  else if (ext == "png") png::readPNG(path)
  else stop("Unsupported image type")
}

ensure_3d <- function(img) {
  if (length(dim(img)) == 2) array(img, dim = c(dim(img), 1)) else img
}

flip_vertical <- function(img) {
  img[dim(img)[1]:1, , , drop = FALSE]
}

# ============================================================
# MASK → POLYGONS
# ============================================================

mask_to_instance_polygons <- function(arr) {
  r <- terra::rast(arr)
  ids <- setdiff(unique(as.vector(arr)), 0)
  if (length(ids) == 0) return(st_sf(geometry = st_sfc()))
  
  out <- list()
  for (id in ids) {
    m <- r
    m[m != id] <- NA
    m[m == id] <- 1
    p <- terra::as.polygons(m, dissolve = TRUE, na.rm = TRUE)
    if (!is.null(p) && nrow(p) > 0)
      out[[length(out) + 1]] <- st_as_sf(p)
  }
  
  if (length(out) == 0) return(st_sf(geometry = st_sfc()))
  do.call(rbind, out)
}

# ============================================================
# TILE CROPPING + SAFE CLIPPING (FIX APPLIED HERE)
# ============================================================

crop_tiles_and_polygons <- function(image_path, tiles, polys_sf) {
  
  img <- flip_vertical(ensure_3d(read_image_any(image_path)))
  out <- vector("list", length(tiles))
  
  for (i in seq_along(tiles)) {
    
    t <- tiles[[i]]
    
    tile_img <- img[t$y_start:t$y_end, t$x_start:t$x_end, , drop = FALSE]
    
    tile_box <- st_polygon(list(matrix(
      c(
        t$x_start, t$y_start,
        t$x_end,   t$y_start,
        t$x_end,   t$y_end,
        t$x_start, t$y_end,
        t$x_start, t$y_start
      ),
      ncol = 2, byrow = TRUE
    )))
    
    clipped <- suppressWarnings(
      st_intersection(polys_sf, st_sfc(tile_box))
    )
    
    poly_out <- list()
    
    for (j in seq_len(nrow(clipped))) {
      
      geom <- st_geometry(clipped[j, ])
      if (st_is_empty(geom)) next
      
      gtype <- as.character(st_geometry_type(geom))
      
      geom <- switch(
        gtype,
        "GEOMETRYCOLLECTION" = st_collection_extract(geom, "POLYGON"),
        "MULTIPOLYGON"       = geom,
        "POLYGON"            = st_sfc(geom),
        NULL
      )
      
      if (is.null(geom) || length(geom) == 0) next
      
      for (k in seq_along(geom)) {
        c <- st_coordinates(geom[k])
        poly_out[[length(poly_out) + 1]] <-
          data.frame(
            x = c[, "X"] - t$x_start,
            y = c[, "Y"] - t$y_start
          )
      }
    }
    
    out[[i]] <- list(
      image    = tile_img,
      polygons = poly_out
    )
  }
  
  out
}

# ============================================================
# YOLO EXPORT
# ============================================================

sf_poly_to_yolo <- function(p, tile_size) {
  if (!all(p[1, ] == p[nrow(p), ])) p <- rbind(p, p[1, ])
  as.vector(rbind(p$x / tile_size, p$y / tile_size))
}

write_yolo_labels <- function(polys, path, tile_size) {
  if (length(polys) == 0) {
    file.create(path)
    return()
  }
  
  lines <- sapply(polys, function(p) {
    v <- sf_poly_to_yolo(p, tile_size)
    if (length(v) < 6) return(NULL)
    paste(0, paste(sprintf("%.6f", v), collapse = " "))
  })
  
  writeLines(lines[!is.na(lines)], path)
}

# ============================================================
# MAIN LOOP
# ============================================================

image_paths <- list.files(
  image_dir,
  pattern = "\\.(tif|tiff|jpg|jpeg|png)$",
  full.names = TRUE
)

tile_id <- 1L

for (img_path in image_paths) {
  
  base <- tools::file_path_sans_ext(basename(img_path))
  mask_path <- file.path(mask_dir, paste0(base, "_mask.npy"))
  
  if (!file.exists(mask_path)) next
  
  message("Processing ", base)
  
  img <- flip_vertical(ensure_3d(read_image_any(img_path)))
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  mask <- np$load(mask_path)
  polys_sf <- mask_to_instance_polygons(mask)
  
  raster_bin <- polygons_to_raster(
    lapply(seq_len(nrow(polys_sf)), function(i)
      list(segmentation = st_coordinates(polys_sf[i, ])[, 1:2])),
    w, h
  )
  
  tiles <- pack_boxes(raster_bin, tile_size, min_coverage)
  
  tiles_out <- crop_tiles_and_polygons(img_path, tiles, polys_sf)
  
  for (t in tiles_out) {
    name <- sprintf("%s_%06d", base, tile_id)
    jpeg::writeJPEG(
      t$image,
      file.path(img_dir, paste0(name, ".jpg")),
      quality = 0.95
    )
    write_yolo_labels(
      t$polygons,
      file.path(lbl_dir, paste0(name, ".txt")),
      tile_size
    )
    tile_id <- tile_id + 1L
  }
}

