# Image sectioning --------------------------------------------------------
source("full_surface_pipeline/full_surface_testing.R")
source("full_surface_pipeline/crop_tiles_and_polygons.R")
# USAGE EXAMPLE

# poly_list: filtered polygons (after convex hull)
image_width <- 1004
image_height <- 1002
# invert_polylist_y <- function(poly_list, image_height) {
#   
#   lapply(poly_list, function(p) {
#     df <- p$segmentation
#     
#     df$y <- image_height - df$y
#     
#     p$segmentation <- df
#     p
#   })
# }
# poly_list_inverted <- invert_polylist_y(poly_list, image_height) #to make the polygon match the image
raster_bin <- polygons_to_raster(poly_list, image_width, image_height)
tiles <- pack_boxes(raster_bin, tile_size = 128, min_coverage = 0.7)
plot<-plot_tiles_with_polygons(image_path, tiles, poly_list)

tiles_cropped <- crop_tiles_and_polygons(
  image_path  = image_path,
  tiles       = tiles,
  poly_list   = poly_list,
  image_width = image_width,
  image_height = image_height
)



library(raster)
library(sf)
library(imager)

polygons_to_mask_tile <- function(polygons, tile_size = 128) {

  r <- raster(
    nrow = tile_size,
    ncol = tile_size,
    xmn = 0, xmx = tile_size,
    ymn = 0, ymx = tile_size
  )
  values(r) <- 0

  if (length(polygons) == 0) return(r)

  sf_polys <- st_sfc(lapply(polygons, function(p) {
    if (!all(p[1,] == p[nrow(p),]))
      p <- rbind(p, p[1,])
    st_polygon(list(as.matrix(p)))
  }))

  sf_polys <- st_sf(geometry = sf_polys)

  rasterize(sf_polys, r, field = 1, background = 0, fun = "max")
}

invert_mask <- function(mask) {
  mask[] <- 1 - mask[]
  mask
}
sf_poly_to_df <- function(poly_sf) {
  coords <- st_coordinates(poly_sf)
  data.frame(
    x = coords[,1],
    y = coords[,2]
  )
}
mask_to_polygons <- function(mask) {
  
  mask[mask == 0] <- NA
  if (all(is.na(values(mask))))
    return(st_sf(geometry = st_sfc(), crs = NA))
  
  st_as_sf(
    raster::rasterToPolygons(mask, fun = function(x) x == 1, dissolve = TRUE)
  )
}

filter_gap_polygons <- function(polys_sf, min_area_px) {
  
  if (nrow(polys_sf) == 0) return(polys_sf)
  
  keep <- sapply(seq_len(nrow(polys_sf)), function(i) {
    df <- sf_poly_to_df(polys_sf[i, ])
    geom <- recompute_geometry(df)
    geom$area >= min_area_px
  })
  
  polys_sf[keep, , drop = FALSE]
}



erode_polygons <- function(polys_sf, erosion_px = 3) {
  
  if (nrow(polys_sf) == 0) return(polys_sf)
  
  st_make_valid(st_buffer(polys_sf, dist = -erosion_px))
}

normalize_sf <- function(sf_obj, type_label) {
  if (nrow(sf_obj) == 0) {
    st_sf(type = character(0), geometry = st_sfc())
  } else {
    st_sf(
      type = type_label,
      geometry = st_geometry(sf_obj)
    )
  }
}


postprocess_tile <- function(tile,
                             tile_size = 128,
                             erosion_px = 3,
                             min_gap_area = 200) {
  
  # --- object mask
  obj_mask <- polygons_to_mask_tile(tile$polygons, tile_size)
  
  # --- inverted mask (gaps)
  gap_mask <- invert_mask(obj_mask)
  
  # --- gap polygons
  gap_polys <- mask_to_polygons(gap_mask)
  gap_polys <- erode_polygons(gap_polys, erosion_px)
  gap_polys <- filter_gap_polygons(gap_polys, min_gap_area)
  
  # --- original object polygons
  obj_sf <- st_sfc(lapply(tile$polygons, function(p) {
    if (!all(p[1,] == p[nrow(p),]))
      p <- rbind(p, p[1,])
    st_polygon(list(as.matrix(p)))
  }))
  
  obj_sf <- st_sf(geometry = obj_sf)
  
  # --- combine objects + gaps
  obj_sf_norm  <- normalize_sf(obj_sf, "object")
  gap_sf_norm  <- normalize_sf(gap_polys, "gap")
  
  combined <- rbind(obj_sf_norm, gap_sf_norm)
  
  
  list(
    image             = tile$image,
    object_mask       = obj_mask,
    gap_mask          = gap_mask,
    combined_polygons = combined
  )
}


tiles_processed <- lapply(tiles_cropped, postprocess_tile)

