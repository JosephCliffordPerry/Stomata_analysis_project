suppressPackageStartupMessages({
  library(sf)
  library(FNN)
})

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
    
    tryCatch(st_polygon(list(p)), error = function(e) NULL)
  })
  
  valid <- !sapply(cleaned, is.null)
  if (!any(valid)) return(NULL)
  
  polys_sf <- st_sf(
    node_id = seq_len(sum(valid)),
    geometry = st_sfc(cleaned[valid])
  )
  
  polys_sf <- st_set_precision(polys_sf, precision)
  polys_sf <- st_make_valid(polys_sf)
  
  # ===============================
  # GRAPH
  # ===============================
  if (is.null(axis_edges) || nrow(axis_edges) == 0) {
    
    polys_sf$group <- polys_sf$node_id
    
  } else {
    
    edges <- as.matrix(axis_edges)
    edges <- edges[complete.cases(edges), , drop = FALSE]
    
    # IMPORTANT: edges already in node_id space
    edges <- data.table(
      a = as.integer(edges[,1]),
      b = as.integer(edges[,2])
    )
    
    edges <- edges[a %in% polys_sf$node_id &
                     b %in% polys_sf$node_id]
    
    if (nrow(edges) == 0) {
      
      polys_sf$group <- polys_sf$node_id
      
    } else {
      
      g <- graph_from_data_frame(
        edges,
        directed = FALSE,
        vertices = data.frame(name = polys_sf$node_id)
      )
      
      comps <- components(g)$membership
      
      polys_sf$group <- comps[as.character(polys_sf$node_id)]
    }
  }
  
  # ===============================
  # CONVEX HULL MERGE
  # ===============================
  merged_list <- lapply(
    split(polys_sf, polys_sf$group),
    function(group_sf){
      
      geom_union <- st_union(group_sf$geometry)
      hull <- st_convex_hull(geom_union)
      
      st_sf(
        node_id = min(group_sf$id),
        members = paste(group_sf$id, collapse = ","),
        geometry = st_sfc(hull)
      )
    }
  )
  
  merged <- do.call(rbind, merged_list)
  
  merged$node_id <- seq_len(nrow(merged))
  
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



# filter non overlapping polygons -----------------------------------------

# =========================================================
# SPLIT POLYGONS INTO:
#
# 1. isolated_polygons
#    - no overlap with k nearest neighbours
#    - passes area threshold
#    - contains NO long axis-aligned edges
#
# 2. remaining_polygons
# =========================================================

split_isolated_polygons <- function(
    polygons,
    k = 10,
    overlap_tol = 0,
    area_min = 0,
    axis_length_threshold = 20,
    axis_tol = 1e-6
){

  
  
  has_long_axis_edge <- function(coords){
    
    dx <- diff(coords[,1])
    dy <- diff(coords[,2])
    
    seg_len <- sqrt(dx^2 + dy^2)
    
    vertical <- abs(dx) <= axis_tol
    horizontal <- abs(dy) <= axis_tol
    
    any(
      seg_len >= axis_length_threshold &
        (vertical | horizontal)
    )
  }
  

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
      geometry = st_sfc(sfg_list)
    )
    
  } else {
    
    stop("Unsupported polygon format")
  }
  
  sf_polys <- st_make_valid(sf_polys)
  
  n <- nrow(sf_polys)
  
  if (n == 0) {
    
    return(list(
      isolated_polygons = sf_polys,
      remaining_polygons = sf_polys
    ))
  }
 
  cent <- st_coordinates(
    st_centroid(sf_polys)
  )
  
  knn <- FNN::get.knn(
    cent,
    k = min(k, max(1, n - 1))
  )
  
  areas <- as.numeric(
    st_area(sf_polys)
  )
  
  isolated <- logical(n)
  
  
  for (i in seq_len(n)) {
    
  
    if (areas[i] < area_min)
      next
    
    g1 <- sf_polys$geometry[i]
    
    coords <- st_coordinates(g1)[,1:2,drop=FALSE]
    
    if (has_long_axis_edge(coords))
      next
    
   
    nbrs <- knn$nn.index[i,]
    
    has_overlap <- FALSE
    
    for (j in nbrs) {
      
      g2 <- sf_polys$geometry[j]
      
      inter <- suppressWarnings(
        st_intersection(g1, g2)
      )
      
      if (length(inter) == 0)
        next
      
      a <- suppressWarnings(
        as.numeric(st_area(inter))
      )
      
      if (length(a) > 0 &&
          any(is.finite(a)) &&
          max(a, na.rm = TRUE) > overlap_tol) {
        
        has_overlap <- TRUE
        break
      }
    }
    
    isolated[i] <- !has_overlap
  }
  
  list(
    isolated_polygons = sf_polys[isolated,],
    remaining_polygons = sf_polys[!isolated,]
  )
}
