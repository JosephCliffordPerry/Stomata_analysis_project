# ============================================================
# YOLO Tiled Segmentation Pipeline (modular functions)
# ============================================================

library(reticulate)
library(tiff)
library(sf)
library(igraph)
library(ggplot2)
library(grid)
Sys.setenv(RETICULATE_PYTHON = "managed")
reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","ultralytics"), 
  python_version = "3.12.4"
)

# ------------------------------
# PARAMETERS
# ------------------------------
params <- list(
  image_path = "D:/stomata/November_images/November_image_data_8bit_tifs/A-T2R3_Ab_1_frame_0005.tif",
  model_path = "D:/stomata/surface_segment/weights/best.pt",
  tile_size  = 128,
  overlap    = 96,
  min_area   = 10,
  max_area   = 5000,
  min_circ   = 0.2,
  max_circ   = 1.0,
  alpha      = 0.4
)

params$stride <- params$tile_size - params$overlap

# ------------------------------
# PYTHON YOLO MODEL
# ------------------------------
py_run_string(sprintf("
import numpy as np
from ultralytics import YOLO

model = YOLO(r'''%s''')

def segment_tile_yolo(tile):
    if tile.ndim == 2:
        tile = np.stack([tile]*3, axis=-1)
    elif tile.shape[2] == 1:
        tile = np.concatenate([tile]*3, axis=2)
    if tile.dtype != np.uint8:
        tile = (255 * tile / tile.max()).astype(np.uint8)
    results = model.predict(tile, task='seg', save=False, verbose=False)[0]
    polys = []
    if results.masks is not None:
        for m in results.masks.xy:
            polys.append(np.asarray(m))
    return polys
", params$model_path))

# ------------------------------
# 1. IMAGE LOADING
# ------------------------------
load_image <- function(path){
  img <- tiff::readTIFF(path)
  if(length(dim(img))==2) img <- array(img, dim=c(dim(img),1))
  img
}

# ------------------------------
# 2. PAD IMAGE
# ------------------------------
pad_image <- function(img, tile_size, stride){
  h <- dim(img)[1]; w <- dim(img)[2]; c <- dim(img)[3]
  pad_h <- ceiling((h - tile_size)/stride)*stride + tile_size
  pad_w <- ceiling((w - tile_size)/stride)*stride + tile_size
  padded <- array(0, dim=c(pad_h, pad_w, c))
  padded[1:h, 1:w, ] <- img
  padded
}

# ------------------------------
# 3. GENERATE TILES
# ------------------------------
generate_tiles <- function(img_dim, tile_size, stride){
  H <- img_dim[1]; W <- img_dim[2]
  x_starts <- seq(1, W - tile_size + 1, by=stride)
  y_starts <- seq(1, H - tile_size + 1, by=stride)
  tiles <- list()
  for(y0 in y_starts) for(x0 in x_starts)
    tiles[[length(tiles)+1]] <- list(x0=x0, y0=y0, x1=x0+tile_size-1, y1=y0+tile_size-1)
  tiles
}

# ------------------------------
# 4. CLEAN POLYGON (robust)
# ------------------------------
clean_polygon <- function(p){
  if(!all(p[1,]==p[nrow(p),])) p <- rbind(p, p[1,])
  poly <- tryCatch({
    poly <- st_polygon(list(as.matrix(data.frame(x=p[,1], y=p[,2]))))
    poly <- st_zm(poly)
    poly <- st_make_valid(poly)
    coords <- st_coordinates(poly)[,1:2]
    if(!all(coords[1,]==coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
    st_polygon(list(coords))
  }, error=function(e) NULL)
  poly
}

# ------------------------------
# 5. DEDUPLICATE WITHIN TILE
# ------------------------------
polygon_iou <- function(poly1, poly2){
  inter <- st_area(st_intersection(poly1, poly2))
  union <- st_area(st_union(poly1, poly2))
  as.numeric(inter/union)
}

deduplicate_tile <- function(tile_polys_sf, iou_thresh=0.5){
  if(nrow(tile_polys_sf)<=1) return(tile_polys_sf)
  keep <- rep(TRUE, nrow(tile_polys_sf))
  for(i in seq_len(nrow(tile_polys_sf)-1)){
    if(!keep[i]) next
    geom_i <- st_geometry(tile_polys_sf[i])[[1]]
    for(j in (i+1):nrow(tile_polys_sf)){
      if(!keep[j]) next
      geom_j <- st_geometry(tile_polys_sf[j])[[1]]
      if(polygon_iou(st_sf(geometry=geom_i), st_sf(geometry=geom_j)) >= iou_thresh){
        geom_i <- st_union(geom_i, geom_j)
        tile_polys_sf[i,] <- st_sf(geometry=st_sfc(geom_i))
        keep[j] <- FALSE
      }
    }
  }
  tile_polys_sf[keep,]
}

# ------------------------------
# 6. SHIFT TO GLOBAL COORDINATES
# ------------------------------
shift_geometries <- function(tile_polys_sf, xoff, yoff){
  geoms_shifted <- lapply(st_geometry(tile_polys_sf), function(poly){
    coords <- st_coordinates(poly)[,1:2]
    coords[,1] <- coords[,1] + xoff - 1
    coords[,2] <- coords[,2] + yoff - 1
    if(!all(coords[1,]==coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
    poly_shift <- st_polygon(list(coords))
    st_zm(st_make_valid(poly_shift))
  })
  st_geometry(tile_polys_sf) <- st_sfc(geoms_shifted)
  tile_polys_sf
}

# ------------------------------
# 7. MERGE ACROSS SEAMS
# ------------------------------
merge_across_seams <- function(polys_sf){
  if(nrow(polys_sf)<=1) return(polys_sf)
  adj <- st_touches(polys_sf)
  g <- igraph::graph_from_adj_list(adj)
  comps <- igraph::components(g)$membership
  merged_list <- lapply(unique(comps), function(i) st_union(polys_sf[comps==i,]))
  st_sf(geometry=st_sfc(merged_list))
}

# ------------------------------
# 8. FILTER BY AREA & CIRCULARITY
# ------------------------------
filter_polygons <- function(polys_sf, min_area, max_area, min_circ, max_circ){
  poly_area <- st_area(polys_sf)
  poly_perim <- st_length(st_cast(polys_sf, "MULTILINESTRING"))
  circ <- as.numeric(4*pi*poly_area/(poly_perim^2))
  polys_sf[
    poly_area>=min_area & poly_area<=max_area &
      circ>=min_circ & circ<=max_circ,
  ]
}

# ------------------------------
# 9. PLOT OVERLAY
# ------------------------------
plot_polygons <- function(img, polys_sf, alpha=0.4){
  orig_h <- dim(img)[1]; orig_w <- dim(img)[2]
  img_rgb <- array(rep(img,3), dim=c(orig_h, orig_w,3))
  g <- rasterGrob(img_rgb, width=unit(1,"npc"), height=unit(1,"npc"))
  poly_df <- do.call(rbind, lapply(seq_len(nrow(polys_sf)), function(i){
    c <- st_coordinates(polys_sf[i,])
    data.frame(x=c[,1], y=c[,2], group=i)
  }))
  ggplot(poly_df, aes(x,y,group=group)) +
    annotation_custom(g, xmin=0, xmax=orig_w, ymin=0, ymax=orig_h) +
    geom_polygon(fill="red", color="black", alpha=alpha) +
    coord_equal() +
    theme_void()
}

# do bar ------------------------------------------------------------------


init_progress <- function(n){
  pb <- txtProgressBar(min = 0, max = n, style = 3)
  list(
    pb = pb,
    tick = function(i) setTxtProgressBar(pb, i),
    close = function() close(pb)
  )
}

# ------------------------------
# 10. WRAPPER FUNCTION
# ------------------------------
run_yolo_pipeline <- function(params){
  
  img <- load_image(params$image_path)
  img_pad <- pad_image(img, params$tile_size, params$stride)
  tiles <- generate_tiles(dim(img_pad), params$tile_size, params$stride)
  
  all_tiles_polys <- list()
  
  # ---- progress bar ----
  prog <- init_progress(length(tiles))
  
  for(i in seq_along(tiles)){
    t <- tiles[[i]]
    prog$tick(i)
    
    tile_img <- img_pad[t$y0:t$y1, t$x0:t$x1,,drop=FALSE]
    tile_polys <- py$segment_tile_yolo(tile_img)
    if(length(tile_polys)==0) next
    
    clean_polys <- lapply(tile_polys, clean_polygon)
    clean_polys <- Filter(Negate(is.null), clean_polys)
    if(length(clean_polys)==0) next
    
    tile_polys_sf <- st_sf(geometry = st_sfc(clean_polys))
    
    #tile_polys_sf <- deduplicate_tile(tile_polys_sf, iou_thresh = 0.5)
    if(nrow(tile_polys_sf)==0) next
    
    tile_polys_sf <- shift_geometries(tile_polys_sf, t$x0, t$y0)
    
    all_tiles_polys <- c(all_tiles_polys, tile_polys_sf)
  }
  
  prog$close()
  
  if(length(all_tiles_polys)==0){
    stop("No polygons detected")
  }
  
  all_polys_sf <- st_sf(
    geometry = st_sfc(lapply(all_tiles_polys, function(x) x$geometry[[1]]))
  )
  
  all_polys_sf <- st_make_valid(all_polys_sf)
  all_polys_sf <- all_polys_sf[!st_is_empty(all_polys_sf), ]
  
  merged_sf <- merge_across_seams(all_polys_sf)
  
  filtered_sf <- filter_polygons(
    merged_sf,
    params$min_area,
    params$max_area,
    params$min_circ,
    params$max_circ
  )
  
  plot_polygons(img, filtered_sf, params$alpha)
}

# ------------------------------
# RUN PIPELINE
# ------------------------------
run_yolo_pipeline(params)
