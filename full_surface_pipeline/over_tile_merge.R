#this could be optimised using border aware merging 
#
# =========================================================
# Morphological stitching pipeline using SF and convex hull
# Percent-based metric buffers + size + edge filters
# =========================================================

library(EBImage)
library(raster)
library(sf)
library(sp)
library(ggplot2)
library(grid)

# --------------------------------------------------------
# Polygon metrics helpers
# --------------------------------------------------------
poly_area <- function(p){
  x <- p[,1]; y <- p[,2]
  0.5 * abs(sum(x[-1]*y[-length(y)] - x[-length(x)]*y[-1]))
}

poly_perimeter <- function(p){
  sum(sqrt(rowSums((p[-1,] - p[-nrow(p),])^2)))
}

polygon_metrics <- function(polys){
  data.frame(
    id = seq_along(polys),
    area = sapply(polys, poly_area),
    circularity = sapply(polys, function(p){
      a <- poly_area(p)
      per <- poly_perimeter(p)
      if(per == 0) return(0)
      4*pi*a/(per^2)
    })
  )
}

polygon_percent_buffer <- function(polys, percent){
  lapply(polys, function(p){
    if(!all(p[1,] == p[nrow(p),]))
      p <- rbind(p, p[1,])
    a <- poly_area(p)
    r_eq <- sqrt(a / pi)
    percent * r_eq
  })
}

# --------------------------------------------------------
# Filters
# --------------------------------------------------------
touches_image_border <- function(p, img_w, img_h, eps = 1){
  any(
    p[,1] <= 1 + eps |
      p[,2] <= 1 + eps |
      p[,1] >= img_w - eps |
      p[,2] >= img_h - eps
  )
}

filter_polygons <- function(polys,
                            img_dim,
                            min_area = 0,
                            max_area = Inf,
                            remove_edge = TRUE,
                            eps = 1){
  
  if(length(polys) == 0) return(list(polys = list(), metrics = data.frame()))
  
  img_h <- img_dim[1]
  img_w <- img_dim[2]
  
  metrics <- polygon_metrics(polys)
  
  keep <- rep(TRUE, length(polys))
  
  # --- Size filter ---
  keep <- keep &
    metrics$area >= min_area &
    metrics$area <= max_area
  
  # --- Edge filter ---
  if(remove_edge){
    edge_hit <- sapply(polys, touches_image_border, img_w, img_h, eps)
    keep <- keep & !edge_hit
  }
  
  list(
    polys = polys[keep],
    metrics = metrics[keep, , drop = FALSE]
  )
}

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
    
    if(!all(coords[1,] == coords[nrow(coords),]))
      coords <- rbind(coords, coords[1,])
    
    sp_polys[[length(sp_polys)+1]] <- 
      Polygons(list(Polygon(coords)), ID = as.character(i))
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
  
  if(nrow(fg) == 0)
    return(st_sf(geometry = st_sfc()))
  
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
  merged <- st_convex_hull(pts_buffer)
  
  polys <- st_cast(merged, "POLYGON")
  polys <- st_convex_hull(polys)
  
  st_sf(geometry = polys)
}

# --------------------------------------------------------
# Percent-based shrink
# --------------------------------------------------------
shrink_polygons_sf <- function(polys, shrink_percent){
  
  if(shrink_percent <= 0) return(polys)
  
  shrink_dists <- polygon_percent_buffer(polys, shrink_percent)
  out <- list()
  
  for(i in seq_along(polys)){
    
    p <- polys[[i]]
    
    if(!all(p[1,] == p[nrow(p),]))
      p <- rbind(p, p[1,])
    
    sf_poly <- st_sf(geometry = st_sfc(st_polygon(list(p))))
    sf_shrunk <- st_buffer(sf_poly, -shrink_dists[[i]])
    
    if(st_is_empty(sf_shrunk)) next
    
    coords <- st_coordinates(sf_shrunk)[,1:2]
    
    if(!all(coords[1,] == coords[nrow(coords),]))
      coords <- rbind(coords, coords[1,])
    
    out[[length(out)+1]] <- coords
  }
  
  out
}

# --------------------------------------------------------
# Percent-based dilation
# --------------------------------------------------------
dilate_polygons_sf <- function(sf_polys, dilate_percent){
  
  if(dilate_percent <= 0) return(sf_polys)
  
  poly_list <- lapply(st_geometry(sf_polys), function(g){
    st_coordinates(g)[,1:2]
  })
  
  dilate_dists <- polygon_percent_buffer(poly_list, dilate_percent)
  out <- list()
  
  for(i in seq_along(poly_list)){
    
    p <- poly_list[[i]]
    
    if(!all(p[1,] == p[nrow(p),]))
      p <- rbind(p, p[1,])
    
    sf_poly <- st_sf(geometry = st_sfc(st_polygon(list(p))))
    sf_dilated <- st_buffer(sf_poly, dilate_dists[[i]])
    sf_dilated <- st_make_valid(sf_dilated)
    
    out[[i]] <- sf_dilated
  }
  
  do.call(rbind, out) |> st_as_sf()
}

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

# --------------------------------------------------------
# Full stitching pipeline
# --------------------------------------------------------
curve_stitch_sf <- function(
    polys,
    img_dim,
    shrink_percent = 0.10,
    dilate_percent = 0,
    min_area = 0,
    max_area = Inf,
    remove_edge = TRUE,
    edge_eps = 1
){
  
  polys <- shrink_polygons_sf(polys, shrink_percent)
  mask <- poly_list_to_mask(polys, img_dim)
  stitched_sf <- mask_to_polygons_sf(mask)
  stitched_sf <- dilate_polygons_sf(stitched_sf, dilate_percent)
  
  stitched_poly <- sf_to_poly_list(stitched_sf)
  
  filtered <- filter_polygons(
    stitched_poly,
    img_dim,
    min_area,
    max_area,
    remove_edge,
    edge_eps
  )
  
  list(
    polys = filtered$polys,
    metrics = filtered$metrics,
    stitched_sf = stitched_sf
  )
}

# --------------------------------------------------------
# Overlay plot
# --------------------------------------------------------
plot_overlay <- function(img, polys, metrics, alpha, circ_thresh = 0.95){
  
  if(length(polys) != nrow(metrics))
    stop("Mismatch between polygon count and metrics rows")
  
  H <- dim(img)[1]; W <- dim(img)[2]
  
  img_rgb <- array(rep(img,3), dim = c(H,W,3))
  grob <- rasterGrob(img_rgb, width=unit(1,"npc"), height=unit(1,"npc"))
  
  df <- do.call(rbind, lapply(seq_along(polys), function(i){
    p <- polys[[i]]
    circ_val <- metrics$circularity[i]
    
    data.frame(
      x = p[,1],
      y = p[,2],
      id = i,
      circularity = rep(circ_val, nrow(p))
    )
  }))
  
  df$colour <- ifelse(df$circularity > circ_thresh, "blue", "red")
  
  ggplot(df, aes(x, y, group = id, fill = colour)) +
    annotation_custom(grob, 0, W, 0, H) +
    geom_polygon(alpha = alpha, colour = NA) +
    scale_fill_identity() +
    coord_equal() +
    theme_void() +
    scale_y_reverse()
}

# =========================================================
# Run pipeline
# =========================================================
# 
# polys <- output[[1]]
# img_dim <- dim(img)
# 
# stitched <- curve_stitch_sf(
#   polys,
#   img_dim,
#   shrink_percent = 0.40,
#   dilate_percent = 0.20,
#   min_area = 300,
#   max_area = 7500,
#   remove_edge = TRUE
# )
# stitched <-curve_stitch_blob(polys,img_dim = img_dim)

# plot_overlay(
#   img,
#   stitched$polys,
#   stitched$metrics,
#   params$alpha
# )
# hist(stitched$metrics$area)
