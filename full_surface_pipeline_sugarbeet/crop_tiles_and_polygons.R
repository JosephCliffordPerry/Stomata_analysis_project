# ============================================================
# Robust crop_tiles_and_polygons with multi-format image support
# and safe handling of 2D / 3D images
# ============================================================

library(sf)
library(tiff)
library(jpeg)
library(png)

# ------------------------------------------------------------
# Read image (tif / jpg / png)
# ------------------------------------------------------------
read_image_any <- function(image_path) {
  
  ext <- tolower(tools::file_ext(image_path))
  
  if (ext %in% c("tif", "tiff")) {
    img <- tiff::readTIFF(image_path)
  } else if (ext %in% c("jpg", "jpeg")) {
    img <- jpeg::readJPEG(image_path)
  } else if (ext == "png") {
    img <- png::readPNG(image_path)
  } else {
    stop("Unsupported image format: ", ext)
  }
  
  img
}

# ------------------------------------------------------------
# Ensure image is always 3D (h, w, c)
# ------------------------------------------------------------
ensure_3d <- function(img) {
  if (length(dim(img)) == 2) {
    img <- array(img, dim = c(dim(img), 1))
  }
  img
}

# ------------------------------------------------------------
# Vertical flip (matrix ↔ image coordinate convention)
# ------------------------------------------------------------
flip_vertical <- function(img) {
  img[dim(img)[1]:1, , , drop = FALSE]
}

# ============================================================
# Main tiling + polygon cropping function
# ============================================================

crop_tiles_and_polygons <- function(
    image_path,
    tiles,
    poly_list,
    image_width,
    image_height
) {
  
  # ----------------------------------------------------------
  # Read image ONCE
  # ----------------------------------------------------------
  img <- read_image_any(image_path)
  img <- ensure_3d(img)
  img <- flip_vertical(img)
  
  # ----------------------------------------------------------
  # Convert polygons to sf once
  # ----------------------------------------------------------
  polys_sf <- st_sfc(lapply(poly_list, function(p) {
    coords <- p$segmentation
    if (!all(coords[1, ] == coords[nrow(coords), ])) {
      coords <- rbind(coords, coords[1, ])
    }
    st_polygon(list(as.matrix(coords)))
  }))
  
  polys_sf <- st_sf(
    id = seq_along(polys_sf),
    geometry = polys_sf
  )
  
  # ----------------------------------------------------------
  # Iterate over tiles
  # ----------------------------------------------------------
  out <- vector("list", length(tiles))
  
  for (i in seq_along(tiles)) {
    
    t <- tiles[[i]]
    
    # --------------------------------------------------------
    # Crop image (EXPLICIT 3D INDEXING)
    # --------------------------------------------------------
    tile_img <- img[
      t$y_start:t$y_end,
      t$x_start:t$x_end,
      ,
      drop = FALSE
    ]
    
    # --------------------------------------------------------
    # Tile polygon
    # --------------------------------------------------------
    tile_poly <- st_polygon(list(matrix(
      c(
        t$x_start, t$y_start,
        t$x_end,   t$y_start,
        t$x_end,   t$y_end,
        t$x_start, t$y_end,
        t$x_start, t$y_start
      ),
      ncol = 2,
      byrow = TRUE
    )))
    
    tile_sf <- st_sfc(tile_poly)
    
    # --------------------------------------------------------
    # Clip polygons to tile
    # --------------------------------------------------------
    clipped <- suppressWarnings(
      st_intersection(polys_sf, tile_sf)
    )
    
    poly_out <- lapply(seq_len(nrow(clipped)), function(j) {
      
      if (st_is_empty(clipped[j, ])) return(NULL)
      
      ring <- st_coordinates(
        st_cast(st_geometry(clipped[j, ]), "POLYGON")
      )[, c("X", "Y")]
      
      data.frame(
        x = ring[, 1] - t$x_start,
        y = ring[, 2] - t$y_start
      )
    })
    
    poly_out <- Filter(Negate(is.null), poly_out)
    
    out[[i]] <- list(
      tile_id  = i,
      bbox     = t,
      image    = tile_img,
      polygons = poly_out
    )
  }
  
  out
}
