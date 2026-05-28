filter_polygons <- function(
    polygons,
    df,
    area_min = 500,
    area_max = 4500,
    circ_min = 0.2,
    circ_max = 1
){
  
  stopifnot(length(polygons) == nrow(df))
  stopifnot(all(c("area", "circularity") %in% colnames(df)))
  
  keep_metrics <- with(df,
                       area >= area_min &
                         area <= area_max &
                         circularity >= circ_min &
                         circularity <= circ_max)
  
  polygons <- polygons[keep_metrics]
  df <- df[keep_metrics, , drop = FALSE]
  
  valid_geom <- sapply(polygons, function(p){
    
    if (is.null(p)) return(FALSE)
    
    if (inherits(p, "sfg")) {
      return(length(p) > 0 && nrow(p[[1]]) >= 3)
    }
    
    if (is.numeric(p) && !is.null(dim(p))) {
      return(nrow(p) >= 3 && ncol(p) >= 2)
    }
    
    FALSE
  })
  
  polygons <- polygons[valid_geom]
  df <- df[valid_geom, , drop = FALSE]
  
  list(
    polygons = polygons,
    df = df
  )
}


merge_polygons_convex_hull <- function(polygons, axis_edges, precision = 1e3){
  
  suppressPackageStartupMessages({
    library(sf)
    library(igraph)
    library(data.table)
  })
  
  # =====================================================
  # CLEAN INPUT + INITIALISE ANCESTRY
  # =====================================================
  
  cleaned <- lapply(seq_along(polygons), function(i){
    
    p <- polygons[[i]]
    
    if (inherits(p, "sfg")) return(p)
    
    if (is.null(p) || !is.numeric(p) ||
        is.null(dim(p)) || nrow(p) < 3 || ncol(p) < 2)
      return(NULL)
    
    p <- p[complete.cases(p), , drop = FALSE]
    if (nrow(p) < 3) return(NULL)
    
    if (sqrt(sum((p[1,] - p[nrow(p),])^2)) > 1e-6)
      p <- rbind(p, p[1,])
    
    if (nrow(p) < 4) return(NULL)
    
    p[nrow(p),] <- p[1,]
    
    st_polygon(list(p))
  })
  
  valid <- !sapply(cleaned, is.null)
  if (!any(valid)) return(NULL)
  
  polys_sf <- st_sf(
    node_id = seq_len(sum(valid)),
    origin_ids = lapply(seq_len(sum(valid)), function(i) i),
    geometry = st_sfc(cleaned[valid])
  )
  
  polys_sf <- st_set_precision(polys_sf, precision)
  polys_sf <- st_make_valid(polys_sf)
  
  # =====================================================
  # BUILD GRAPH (robust: no NA groups)
  # =====================================================
  
  polys_sf$group <- polys_sf$node_id
  
  if (!is.null(axis_edges) && nrow(axis_edges) > 0) {
    
    edges <- as.matrix(axis_edges)
    edges <- edges[complete.cases(edges), , drop = FALSE]
    
    edges <- data.table(
      a = as.integer(edges[,1]),
      b = as.integer(edges[,2])
    )
    
    edges <- edges[
      a %in% polys_sf$node_id &
        b %in% polys_sf$node_id
    ]
    
    if (nrow(edges) > 0) {
      
      g <- graph_from_data_frame(
        edges,
        directed = FALSE,
        vertices = data.frame(
          name = as.character(polys_sf$node_id)
        )
      )
      
      comps <- components(g)$membership
      
      # enforce full vector alignment (positional safety)
      group_vec <- polys_sf$node_id
      
      names(comps) <- as.character(names(comps))
      
      for (i in seq_along(group_vec)) {
        
        nid <- as.character(polys_sf$node_id[i])
        
        if (nid %in% names(comps)) {
          group_vec[i] <- comps[[nid]]
        } else {
          group_vec[i] <- max(comps) + i  # isolate safely
        }
      }
      
      polys_sf$group <- group_vec
    }
  }
  
  # =====================================================
  # MERGE WITH ANCESTRY PROPAGATION
  # =====================================================
  
  group_ids <- unique(polys_sf$group)
  
  merged_geom <- vector("list", length(group_ids))
  merged_origin <- vector("list", length(group_ids))
  
  for (ii in seq_along(group_ids)) {
    
    gid <- group_ids[ii]
    
    group_sf <- polys_sf[
      polys_sf$group == gid,
    ]
    
    geom_union <- st_union(group_sf$geometry)
    hull <- st_convex_hull(geom_union)
    
    merged_geom[[ii]] <- hull[[1]]
    
    merged_origin[[ii]] <- unique(
      unlist(group_sf$origin_ids)
    )
  }
  
  merged <- st_sf(
    node_id = seq_along(merged_geom),
    geometry = st_sfc(merged_geom)
  )
  
  merged$origin_ids <- merged_origin
  
  return(merged)
}

build_bbox_iou_edges <- function(polys_sf,
                                 centroid_dist_tol = 10,
                                 bbox_iou_threshold = 0.05){
  
  if (is.null(polys_sf) || nrow(polys_sf) < 2)
    return(NULL)
  
  n <- nrow(polys_sf)
  ids <- polys_sf$node_id
  
  centroids <- st_centroid(polys_sf)
  centroid_xy <- st_coordinates(centroids)[,1:2]
  
  bbox <- t(vapply(st_geometry(polys_sf), function(g){
    bb <- st_bbox(g)
    c(bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"])
  }, numeric(4)))
  
  bbox_iou <- function(i, j){
    
    xA <- max(bbox[i,1], bbox[j,1])
    yA <- max(bbox[i,2], bbox[j,2])
    xB <- min(bbox[i,3], bbox[j,3])
    yB <- min(bbox[i,4], bbox[j,4])
    
    if (xA >= xB || yA >= yB) return(0)
    
    inter <- (xB-xA)*(yB-yA)
    
    ai <- (bbox[i,3]-bbox[i,1])*(bbox[i,4]-bbox[i,2])
    aj <- (bbox[j,3]-bbox[j,1])*(bbox[j,4]-bbox[j,2])
    
    inter/(ai+aj-inter)
  }
  
  dist_fun <- function(i,j){
    dx <- centroid_xy[i,1] - centroid_xy[j,1]
    dy <- centroid_xy[i,2] - centroid_xy[j,2]
    sqrt(dx*dx + dy*dy)
  }
  
  edges <- vector("list", n * (n - 1) / 2)
  e <- 1
  
  for (i in 1:(n-1)){
    for (j in (i+1):n){
      
      if (dist_fun(i,j) > centroid_dist_tol)
        next
      
      if (bbox_iou(i,j) < bbox_iou_threshold)
        next
      
      edges[[e]] <- c(ids[i], ids[j])
      e <- e + 1
    }
  }
  
  if (e == 1) return(NULL)
  
  do.call(rbind, edges[1:(e-1)])
}

# =========================================================
# SPLIT ISOLATED POLYGONS
#
# RETURNS:
#
# 1. isolated_polygons
#    clean final polygons
#
# 2. remaining_original_polygons
#    original polygons NOT used
#    to build isolated polygons
#
# 3. used_original_polygons
#    original polygons consumed
#
# =========================================================
split_isolated_polygons <- function(
    polygons,
    original_polygons = NULL,
    k = 10,
    overlap_tol = 0,
    area_min = 0,
    axis_length_threshold = 20,
    axis_tol = 1e-6
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(FNN)
  })
  
  # =====================================================
  # SAFE INPUT: polygons -> sf_polys
  # =====================================================
  
  if (inherits(polygons, "sf")) {
    
    sf_polys <- polygons
    
  } else if (inherits(polygons, "sfc")) {
    
    sf_polys <- st_sf(geometry = polygons)
    
  } else if (is.list(polygons)) {
    
    sfg_list <- lapply(polygons, function(p){
      
      if (inherits(p, "sfg"))
        return(p)
      
      p <- as.matrix(p)
      
      if (any(p[1,] != p[nrow(p),])) {
        p <- rbind(p, p[1,])
      }
      
      st_polygon(list(p))
    })
    
    sf_polys <- st_sf(
      node_id = seq_along(sfg_list),
      geometry = st_sfc(sfg_list)
    )
    
  } else {
    stop("Unsupported polygons format")
  }
  
  sf_polys <- st_make_valid(sf_polys)
  
  n <- nrow(sf_polys)
  
  # ensure IDs exist
  if (is.null(sf_polys$node_id)) {
    sf_polys$node_id <- seq_len(n)
  }
  
  if (is.null(sf_polys$origin_ids)) {
    sf_polys$origin_ids <- lapply(sf_polys$node_id, function(x) x)
  }
  
  # =====================================================
  # SAFE INPUT: original_polygons -> original_sf
  # =====================================================
  
  if (is.null(original_polygons)) {
    
    original_sf <- sf_polys
    
  } else if (inherits(original_polygons, "sf")) {
    
    original_sf <- original_polygons
    
  } else if (inherits(original_polygons, "sfc")) {
    
    original_sf <- st_sf(geometry = original_polygons)
    
  } else if (is.list(original_polygons)) {
    
    sfg_list <- lapply(original_polygons, function(p){
      
      if (inherits(p, "sfg"))
        return(p)
      
      p <- as.matrix(p)
      
      if (any(p[1,] != p[nrow(p),])) {
        p <- rbind(p, p[1,])
      }
      
      st_polygon(list(p))
    })
    
    original_sf <- st_sf(
      node_id = seq_along(sfg_list),
      geometry = st_sfc(sfg_list)
    )
    
  } else {
    stop("Unsupported original_polygons format")
  }
  
  # ensure id exists
  if (is.null(original_sf$node_id)) {
    original_sf$node_id <- seq_len(nrow(original_sf))
  }
  
  # =====================================================
  # EDGE CASE
  # =====================================================
  
  if (n == 0) {
    
    return(list(
      isolated_polygons = sf_polys,
      remaining_original_polygons = original_sf,
      used_original_polygons = original_sf[0,]
    ))
  }
  
  # =====================================================
  # KNN
  # =====================================================
  
  cent <- st_coordinates(st_centroid(sf_polys))
  
  knn <- FNN::get.knn(
    cent,
    k = min(k, max(1, n - 1))
  )
  
  areas <- as.numeric(st_area(sf_polys))
  
  isolated <- logical(n)
  
  # =====================================================
  # ISOLATION TEST
  # =====================================================
  
  for (i in seq_len(n)) {
    
    if (areas[i] < area_min)
      next
    
    g1 <- sf_polys$geometry[i]
    
    coords <- st_coordinates(g1)[,1:2,drop=FALSE]
    
    dx <- diff(coords[,1])
    dy <- diff(coords[,2])
    seg_len <- sqrt(dx^2 + dy^2)
    
    has_axis_edge <- any(
      seg_len >= axis_length_threshold &
        (abs(dx) <= axis_tol | abs(dy) <= axis_tol)
    )
    
    if (has_axis_edge)
      next
    
    nbrs <- knn$nn.index[i,]
    
    overlap_found <- FALSE
    
    for (j in nbrs) {
      
      g2 <- sf_polys$geometry[j]
      
      inter <- suppressWarnings(st_intersection(g1, g2))
      
      if (length(inter) == 0)
        next
      
      a <- suppressWarnings(as.numeric(st_area(inter)))
      
      if (length(a) > 0 &&
          max(a, na.rm = TRUE) > overlap_tol) {
        
        overlap_found <- TRUE
        break
      }
    }
    
    isolated[i] <- !overlap_found
  }
  
  # =====================================================
  # OUTPUT ISOLATED
  # =====================================================
  
  isolated_sf <- sf_polys[isolated,]
  
  # =====================================================
  # MAP BACK TO ORIGINALS
  # =====================================================
  
  used_ids <- unique(unlist(isolated_sf$origin_ids))
  
  used_original <- original_sf[
    original_sf$node_id %in% used_ids,
  ]
  
  remaining_original <- original_sf[
    !original_sf$node_id %in% used_ids,
  ]
  
  return(list(
    isolated_polygons = isolated_sf,
    remaining_original_polygons = remaining_original,
    used_original_polygons = used_original
  ))
}

# =========================================================
# DEBUG PLOT
#
# BLACK  = original
# RED    = removed originals
# GREEN  = remaining originals
# CYAN   = isolated final polygons
# =========================================================

plot_isolation_debug <- function(
    img,
    original_polygons,
    used_original_polygons,
    remaining_original_polygons,
    isolated_polygons
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(ggplot2)
    library(grid)
  })
  
  # =====================================================
  # SAFE CONVERTER
  # =====================================================
  
  to_sf <- function(x){
    
    if (inherits(x, "sf")) return(x)
    
    if (inherits(x, "sfc")) return(st_sf(geometry = x))
    
    if (is.list(x)) {
      
      sfg_list <- lapply(x, function(p){
        
        if (inherits(p, "sfg"))
          return(p)
        
        p <- as.matrix(p)
        
        if (any(p[1,] != p[nrow(p),])) {
          p <- rbind(p, p[1,])
        }
        
        st_polygon(list(p))
      })
      
      return(st_sf(geometry = st_sfc(sfg_list)))
    }
    
    stop("Unsupported geometry type")
  }
  
  original_polygons <- to_sf(original_polygons)
  used_original_polygons <- to_sf(used_original_polygons)
  remaining_original_polygons <- to_sf(remaining_original_polygons)
  isolated_polygons <- to_sf(isolated_polygons)
  
  # =====================================================
  # IMAGE PREP
  # =====================================================
  
  if (length(dim(img)) == 2) {
    img <- array(rep(img, 3), dim = c(dim(img), 3))
  }
  
  H <- dim(img)[1]
  W <- dim(img)[2]
  
  grob <- rasterGrob(
    img,
    width = unit(1, "npc"),
    height = unit(1, "npc"),
    interpolate = FALSE
  )
  
  # =====================================================
  # FLIP COORDS
  # =====================================================
  
  flip_sf <- function(sf_obj){
    
    geom_fixed <- lapply(
      st_geometry(sf_obj),
      function(g){
        
        coords <- st_coordinates(g)[,1:2,drop=FALSE]
        coords[,2] <- H - coords[,2]
        
        if (any(coords[1,] != coords[nrow(coords),])) {
          coords <- rbind(coords, coords[1,])
        }
        
        st_polygon(list(coords))
      }
    )
    
    st_sf(geometry = st_sfc(geom_fixed))
  }
  
  original_polygons <- flip_sf(original_polygons)
  used_original_polygons <- flip_sf(used_original_polygons)
  remaining_original_polygons <- flip_sf(remaining_original_polygons)
  isolated_polygons <- flip_sf(isolated_polygons)
  
  # =====================================================
  # PLOT
  # =====================================================
  
  ggplot() +
    annotation_custom(grob, xmin = 0, xmax = W, ymin = 0, ymax = H) +
    
    geom_sf(data = original_polygons, fill = NA, colour = "black", linewidth = 0.2) +
    geom_sf(data = used_original_polygons, fill = "red", colour = "red", alpha = 0.4) +
    geom_sf(data = remaining_original_polygons, fill = "green", colour = "green", alpha = 0.3) +
    geom_sf(data = isolated_polygons, fill = NA, colour = "cyan", linewidth = 1) +
    
    coord_sf(xlim = c(0, W), ylim = c(0, H), expand = FALSE) +
    theme_void()
}

