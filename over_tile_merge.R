# =========================================================
# Morphological stitching pipeline using SF and convex hull
# =========================================================

library(EBImage)
library(raster)
library(sf)


# --------------------------------------------------------
# Polygon list -> raster mask
# --------------------------------------------------------
poly_list_to_mask <- function(polys, img_dim){
  
  r <- raster(
    nrows = img_dim[1],
    ncols = img_dim[2],
    xmn = 0, xmx = img_dim[2],
    ymn = 0, ymx = img_dim[1]
  )
  
  r[] <- 0
  
  sp_polys <- list()
  
  for(i in seq_along(polys)){
    coords <- polys[[i]]
    if(!is.matrix(coords) || nrow(coords) < 3) next
    if(!all(coords[1,] == coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
    sp_polys[[length(sp_polys)+1]] <- Polygons(list(Polygon(coords)), ID = as.character(i))
  }
  
  if(length(sp_polys) == 0) return(r)
  
  sp_all <- SpatialPolygons(sp_polys)
  rasterize(sp_all, r, field = 1, background = 0)
}

# --------------------------------------------------------
# Convert raster mask to SF polygons using convex hull
# --------------------------------------------------------
mask_to_polygons_sf <- function(mask){
  
  mask_bin <- as.matrix(mask) > 0
  
  fg <- which(mask_bin, arr.ind = TRUE)
  if(nrow(fg) == 0) return(st_sf(geometry = st_sfc()))
  
  img_h <- nrow(mask_bin)

  pts <- st_as_sf(
    data.frame(
      x = fg[, "col"],
      y = img_h - fg[, "row"] + 1
    ),
    coords = c("x","y"),
    crs = NA
  )
  
  pts_buffer <- st_buffer(pts, 1)
  merged <- st_union(pts_buffer)
  
  polys <- st_cast(merged, "POLYGON")
  polys <- st_convex_hull(polys)
  
  st_sf(geometry = polys)
}

# --------------------------------------------------------
# Shrink polygon list using SF negative buffer
# --------------------------------------------------------
shrink_polygons_sf <- function(polys, shrink_dist){
  
  if(shrink_dist <= 0) return(polys)
  
  sf_polys <- st_as_sf(st_sfc(lapply(polys, function(p){
    if(!all(p[1,] == p[nrow(p),])) p <- rbind(p, p[1,])
    st_polygon(list(p))
  })))
  
  sf_shrunk <- st_buffer(sf_polys, -shrink_dist)
  sf_shrunk <- sf_shrunk[!st_is_empty(sf_shrunk), ]
  
  lapply(st_geometry(sf_shrunk), function(g){
    coords <- st_coordinates(g)[,1:2]
    if(!all(coords[1,] == coords[nrow(coords),]))
      coords <- rbind(coords, coords[1,])
    coords
  })
}

# --------------------------------------------------------
# NEW: Polygon dilation using buffer
# --------------------------------------------------------
dilate_polygons_sf <- function(sf_polys, dilate_dist){
  
  if(dilate_dist <= 0) return(sf_polys)
  
  sf_polys <- st_buffer(sf_polys, dilate_dist)
  
  # Clean geometry (prevents self-intersections)
  sf_polys <- st_make_valid(sf_polys)
  
  sf_polys
}

# --------------------------------------------------------
# Full pipeline
# --------------------------------------------------------
curve_stitch_sf <- function(
    polys,
    img_dim,
    shrink_dist = 3,
    dilate_radius = 3
){
  
  # 0. shrink polygons
  polys <- shrink_polygons_sf(polys, shrink_dist)
  
  # 1. rasterize polygons
  mask <- poly_list_to_mask(polys, img_dim)
  
  # 3. convert mask → polygons
  stitched_sf <- mask_to_polygons_sf(mask)
  
  # 4. polygon dilation (buffer shape)
  stitched_sf <- dilate_polygons_sf(stitched_sf, dilate_radius)
  
  stitched_sf
}

# =========================================================
# Run pipeline
# =========================================================
polys <- output[[1]]
img_dim <- dim(img)

stitched_sf <- curve_stitch_sf(
  polys,
  img_dim,
  shrink_dist = 10,
  dilate_radius = 0
)

# --------------------------------------------------------
# Convert SF polygons → list of Nx2 coords
# --------------------------------------------------------
sf_to_poly_list <- function(sfobj){
  lapply(st_geometry(sfobj), function(g){
    coords <- st_coordinates(g)[,1:2]
    if(!all(coords[1,] == coords[nrow(coords),]))
      coords <- rbind(coords, coords[1,])
    coords
  })
}

stitched_poly <- sf_to_poly_list(stitched_sf)

plot_overlay(img, stitched_poly, params$alpha)