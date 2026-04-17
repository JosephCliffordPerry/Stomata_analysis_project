# =========================================================
# HYBRID MERGE: AXIS-ALIGNED EDGE + IOU
# =========================================================

merge_adjacent_overlapping_polygons <- function(
    polygons,
    df,
    iou_threshold = 0.3,
    bbox_iou_threshold = 0.05,
    min_shared_length = 10,
    distance_tol = 2,
    precision = 1e3
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(dplyr)
    library(igraph)
    library(future.apply)
  })
  
  stopifnot(length(polygons) == nrow(df))
  
  # =====================================================
  # PARALLEL CLEAN
  # =====================================================
  cleaned <- future_lapply(polygons, function(p){
    
    if (is.null(p) || nrow(p) < 3) return(NULL)
    
    p <- p[complete.cases(p), , drop = FALSE]
    if (nrow(p) < 3) return(NULL)
    
    keep <- c(TRUE, rowSums(abs(diff(p)) > 1e-9) > 0)
    p <- p[keep, , drop = FALSE]
    
    if (sqrt(sum((p[1,] - p[nrow(p),])^2)) > 1e-6)
      p <- rbind(p, p[1,])
    
    if (nrow(p) < 4) return(NULL)
    
    p[nrow(p),] <- p[1,]
    
    st_polygon(list(p))
  })
  
  valid <- !sapply(cleaned, is.null)
  polygons <- cleaned[valid]
  df <- df[valid, ]
  
  # =====================================================
  # BUILD SF
  # =====================================================
  polys_sf <- st_sf(
    id = seq_along(polygons),
    area = df$area,
    tile_x = df$tile_x,
    tile_y = df$tile_y,
    geometry = st_sfc(polygons)
  )
  
  polys_sf <- st_set_precision(polys_sf, precision)
  polys_sf <- st_make_valid(polys_sf)
  polys_sf <- polys_sf[!st_is_empty(polys_sf), ]
  
  # =====================================================
  # BBOX PRECOMPUTE
  # =====================================================
  bbox_mat <- t(sapply(st_geometry(polys_sf), function(g){
    bb <- st_bbox(g)
    c(bb$xmin, bb$ymin, bb$xmax, bb$ymax)
  }))
  
  bbox_iou <- function(i, j){
    xa <- max(bbox_mat[i,1], bbox_mat[j,1])
    ya <- max(bbox_mat[i,2], bbox_mat[j,2])
    xb <- min(bbox_mat[i,3], bbox_mat[j,3])
    yb <- min(bbox_mat[i,4], bbox_mat[j,4])
    
    inter <- max(0, xb-xa) * max(0, yb-ya)
    if (inter == 0) return(0)
    
    ai <- (bbox_mat[i,3]-bbox_mat[i,1])*(bbox_mat[i,4]-bbox_mat[i,2])
    aj <- (bbox_mat[j,3]-bbox_mat[j,1])*(bbox_mat[j,4]-bbox_mat[j,2])
    
    inter/(ai+aj-inter)
  }
  
  # =====================================================
  # TILE GROUPING
  # =====================================================
  polys_sf$tile_id <- paste(polys_sf$tile_x, polys_sf$tile_y)
  tile_groups <- split(polys_sf, polys_sf$tile_id)
  
  get_neighbors <- function(tx, ty){
    expand.grid(tile_x=(tx-1):(tx+1), tile_y=(ty-1):(ty+1))
  }
  
  # =====================================================
  # AXIS-ALIGNED CHECK
  # =====================================================
  is_axis_aligned <- function(coords, tol = 1e-6){
    dx <- abs(diff(coords[,1]))
    dy <- abs(diff(coords[,2]))
    all(dx < tol) || all(dy < tol)
  }
  
  # =====================================================
  # PROCESS TILE (PARALLEL)
  # =====================================================
  process_tile <- function(tile_name){
    
    tile <- tile_groups[[tile_name]]
    tx <- tile$tile_x[1]
    ty <- tile$tile_y[1]
    
    neigh <- get_neighbors(tx, ty)
    keys <- paste(neigh$tile_x, neigh$tile_y)
    keys <- intersect(keys, names(tile_groups))
    
    candidates <- do.call(rbind, tile_groups[keys])
    if (is.null(candidates)) return(NULL)
    
    edges <- list()
    k <- 1
    
    for(i in seq_len(nrow(tile))){
      for(j in seq_len(nrow(candidates))){
        
        id1 <- tile$id[i]
        id2 <- candidates$id[j]
        if (id1 >= id2) next
        
        # =================================================
        # 1. AXIS-ALIGNED EDGE MERGE (FIRST)
        # =================================================
        
        if (st_distance(tile[i,], candidates[j,]) < distance_tol){
          
          shared <- tryCatch(
            st_intersection(
              st_boundary(tile[i,]),
              st_boundary(candidates[j,])
            ),
            error = function(e) NULL
          )
          
          if (!is.null(shared) && length(shared) > 0){
            
            coords <- st_coordinates(shared)
            
            if (nrow(coords) > 1 && is_axis_aligned(coords)){
              
              if (as.numeric(st_length(shared)) > min_shared_length){
                edges[[k]] <- c(id1, id2)
                k <- k + 1
                next  # skip IoU if already merged
              }
            }
          }
        }
        
        # =================================================
        # 2. BBOX PREFILTER
        # =================================================
        if (bbox_iou(id1, id2) < bbox_iou_threshold) next
        
        # =================================================
        # 3. IOU MERGE
        # =================================================
        inter <- tryCatch(
          st_intersection(tile[i,], candidates[j,]),
          error=function(e) NULL
        )
        
        if (is.null(inter) || nrow(inter)==0) next
        
        ia <- as.numeric(st_area(inter))
        if (ia == 0) next
        
        uni <- tryCatch(
          st_union(tile[i,], candidates[j,]),
          error=function(e) NULL
        )
        
        if (is.null(uni)) next
        
        ua <- as.numeric(st_area(uni))
        if (ua == 0) next
        
        if (ia/ua >= iou_threshold){
          edges[[k]] <- c(id1, id2)
          k <- k + 1
        }
      }
    }
    
    if (length(edges)==0) return(NULL)
    do.call(rbind, edges)
  }
  
  # =====================================================
  # PARALLEL EXECUTION
  # =====================================================
  edge_list <- future_lapply(names(tile_groups), process_tile)
  edge_list <- edge_list[!sapply(edge_list, is.null)]
  
  # =====================================================
  # GRAPH MERGE
  # =====================================================
  if (length(edge_list) == 0){
    polys_sf$group <- polys_sf$id
  } else {
    edges <- do.call(rbind, edge_list)
    g <- graph_from_edgelist(edges, directed = FALSE)
    comps <- components(g)$membership
    
    polys_sf$group <- polys_sf$id
    polys_sf$group[as.numeric(names(comps))] <- comps
  }
  
  # =====================================================
  # FINAL UNION
  # =====================================================
  merged <- polys_sf %>%
    group_by(group) %>%
    summarise(
      geometry = st_union(geometry),
      area = sum(area),
      .groups = "drop"
    )
  
  list(
    merged = merged,
    mapping = polys_sf[,c("id","group")]
  )
}


# Debug section  ----------------------------------------------------------

subset_tiles <- function(polygons, df, tx_range, ty_range) {
  
  keep <- df$tile_x %in% tx_range & df$tile_y %in% ty_range
  
  list(
    polygons = polygons[keep],
    df = df[keep, ]
  )
}
sub <- subset_tiles(
  polygons = output$polygons,
  df = output$metrics,
  tx_range = 3:7,
  ty_range = 8:12
)

length(sub$polygons)  # should now be manageable (~50–200)
merged_sub<-merge_adjacent_overlapping_polygons(polygons = sub$polygons,df = sub$df)


ggplot() +
  annotation_custom(
    rasterGrob(array(rep(img,3), dim=c(dim(img)[1],dim(img)[2],3))),
    xmin = 0, xmax = dim(img)[2],
    ymin = 0, ymax = dim(img)[1]
  ) +
  geom_sf(data = merged$merged, fill = "red", alpha = 0.4, colour = "yellow") +
  coord_sf(expand = FALSE) +
  scale_y_reverse() +
  theme_void()

# 
# ggplot() +
#   annotation_custom(
#     rasterGrob(array(rep(img,3), dim=c(dim(img)[1],dim(img)[2],3))),
#     xmin = 0, xmax = dim(img)[2],
#     ymin = 0, ymax = dim(img)[1]
#   ) +
#   geom_sf(
#     data = merged$merged,
#     aes(fill = factor(group)),
#     alpha = 0.4,
#     colour = NA
#   ) +
#   coord_sf(expand = FALSE) +
#   scale_y_reverse() +
#   theme_void() +
#   guides(fill = "none")
image_path <- "E:/Stomata/Sugarbeet_stomata_imaging/mips2/V1T2R2-Ab-mip.nd2 - V1T2R2-Ab-mip.nd2 (series 1)_MIP.tif"
load_image <- function(path){
  img <- tiff::readTIFF(path)
  if (length(dim(img)) == 2)
    img <- array(img, dim = c(dim(img), 1))
  img
}
img<-load_image(image_path)
# =========================================================
# OVERLAY SUB-REGION: IMAGE + TILES + MERGED POLYGONS
# (ROBUST + FIXED CHANNEL HANDLING)
# =========================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(sf)
})

# -------------------------------
# PARAMETERS
# -------------------------------
tile_size <- 128
overlap   <- 96
stride    <- tile_size - overlap

# -------------------------------
# SUBSET
# -------------------------------
tile_x_vals <- unique(sub$df$tile_x)
tile_y_vals <- unique(sub$df$tile_y)

# -------------------------------
# COMPUTE BOUNDS
# -------------------------------
xmin <- min((tile_x_vals - 1) * stride) + 1
xmax <- max((tile_x_vals - 1) * stride) + tile_size

ymin <- min((tile_y_vals - 1) * stride) + 1
ymax <- max((tile_y_vals - 1) * stride) + tile_size

xmin <- max(1, xmin)
ymin <- max(1, ymin)
xmax <- min(dim(img)[2], xmax)
ymax <- min(dim(img)[1], ymax)

# -------------------------------
# CROP IMAGE (FULLY ROBUST)
# -------------------------------
dims <- dim(img)

if (length(dims) == 2) {
  # grayscale matrix
  img_crop <- img[ymin:ymax, xmin:xmax]
  H <- dim(img_crop)[1]
  W <- dim(img_crop)[2]
  img_rgb <- array(rep(img_crop, 3), dim = c(H, W, 3))
  
} else if (length(dims) == 3) {
  
  img_crop <- img[ymin:ymax, xmin:xmax, , drop = FALSE]
  H <- dim(img_crop)[1]
  W <- dim(img_crop)[2]
  C <- dim(img_crop)[3]
  
  if (C == 1) {
    img_rgb <- array(rep(img_crop[,,1], 3), dim = c(H, W, 3))
    
  } else if (C >= 3) {
    img_rgb <- img_crop[,,1:3]
    
  } else {
    # fallback
    img_rgb <- array(rep(img_crop[,,1], 3), dim = c(H, W, 3))
  }
  
} else {
  stop("Unsupported image format")
}

# -------------------------------
# NORMALISE (important for plotting)
# -------------------------------
if (max(img_rgb, na.rm = TRUE) > 1) {
  img_rgb <- img_rgb / max(img_rgb, na.rm = TRUE)
}

# -------------------------------
# FLIP RASTER (ONLY FIX NEEDED)
# -------------------------------
img_rgb <- img_rgb[dim(img_rgb)[1]:1, , , drop = FALSE]

grob <- rasterGrob(
  img_rgb,
  width = unit(1, "npc"),
  height = unit(1, "npc")
)

# -------------------------------
# GENERATE TILES
# -------------------------------
generate_tiles <- function(dim, tile, stride){
  
  ys <- seq(1, dim[1] - tile + 1, by = stride)
  xs <- seq(1, dim[2] - tile + 1, by = stride)
  
  expand.grid(
    tile_y = seq_along(ys),
    tile_x = seq_along(xs)
  ) |>
    transform(
      y0 = ys[tile_y],
      x0 = xs[tile_x]
    )
}

tiles <- generate_tiles(dim(img), tile_size, stride)

tiles_sub <- tiles[
  tiles$tile_x %in% tile_x_vals &
    tiles$tile_y %in% tile_y_vals,
]

tiles_df <- data.frame(
  xmin = tiles_sub$x0,
  xmax = tiles_sub$x0 + tile_size,
  ymin = tiles_sub$y0,
  ymax = tiles_sub$y0 + tile_size
)

# -------------------------------
# SHIFT GEOMETRY
# -------------------------------
shift_geom <- function(sf_obj, xmin, ymin) {
  st_geometry(sf_obj) <- st_geometry(sf_obj) - c(xmin - 1, ymin - 1)
  sf_obj
}

merged_shift <- shift_geom(merged_sub$merged, xmin, ymin)

tiles_df$xmin <- tiles_df$xmin - xmin + 1
tiles_df$xmax <- tiles_df$xmax - xmin + 1
tiles_df$ymin <- tiles_df$ymin - ymin + 1
tiles_df$ymax <- tiles_df$ymax - ymin + 1

# -------------------------------
# PLOT
# -------------------------------
ggplot() +
  
  annotation_custom(
    grob,
    xmin = 0, xmax = W,
    ymin = 0, ymax = H
  ) +
  
  geom_rect(
    data = tiles_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "blue",
    alpha = 0.2,
    colour = "cyan",
    linewidth = 0.4
  ) +
  
  geom_sf(
    data = merged_shift,
    fill = "red",
    alpha = 0.4,
    colour = "yellow",
    inherit.aes = FALSE
  ) +
  
  coord_sf(expand = FALSE) +
  scale_y_reverse() +
  theme_void()

