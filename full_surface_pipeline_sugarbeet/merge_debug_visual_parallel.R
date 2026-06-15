merge_debug_visual_parallel <- function(
    polygons,
    df,
    iou_threshold = 0.3,
    bbox_iou_threshold = 0.05,
    min_shared_length = 10,
    distance_tol = 20,
    precision = 1e3,
    workers = parallel::detectCores() - 1
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(dplyr)
    library(igraph)
    library(ggplot2)
    library(future)
    library(future.apply)
  })
  
  plan(multisession, workers = workers)
  
  # ===============================
  # FILTER
  # ===============================
  stopifnot(all(c("area", "circularity") %in% colnames(df)))
  
  keep <- with(df,
               area >= 500 &
                 area <= 4500 &
                 circularity >= 0.2 &
                 circularity <= 1)
  
  polygons <- polygons[keep]
  df <- df[keep, ]
  
  valid <- !sapply(polygons, is.null)
  polygons <- polygons[valid]
  df <- df[valid, ]
  
  # ===============================
  # CLEAN
  # ===============================
  cleaned <- lapply(polygons, function(p){
    if (is.null(p) || nrow(p) < 3) return(NULL)
    p <- p[complete.cases(p), , drop = FALSE]
    if (nrow(p) < 3) return(NULL)
    
    if (sqrt(sum((p[1,] - p[nrow(p),])^2)) > 1e-6)
      p <- rbind(p, p[1,])
    
    if (nrow(p) < 4) return(NULL)
    p[nrow(p),] <- p[1,]
    
    st_polygon(list(p))
  })
  
  valid <- !sapply(cleaned, is.null)
  polygons <- cleaned[valid]
  df <- df[valid,]
  
  polys_sf <- st_sf(
    id = seq_along(polygons),
    geometry = st_sfc(polygons)
  )
  
  polys_sf <- st_set_precision(polys_sf, precision)
  polys_sf <- st_make_valid(polys_sf)
  
  n <- nrow(polys_sf)
  if (n < 2) {
    polys_sf$group <- polys_sf$id
    return(NULL)
  }
  
  # ===============================
  # PRECOMPUTE (SHARED)
  # ===============================
  centroids <- st_coordinates(st_centroid(polys_sf))[,1:2]
  
  bbox_mat <- t(sapply(st_geometry(polys_sf), function(g){
    bb <- st_bbox(g)
    c(bb$xmin, bb$ymin, bb$xmax, bb$ymax)
  }))
  
  # SEGMENTS
  get_axis_segments <- function(coords, tol = 1e-6){
    segs <- list(); k <- 1
    for(i in 1:(nrow(coords)-1)){
      x1 <- coords[i,1]; y1 <- coords[i,2]
      x2 <- coords[i+1,1]; y2 <- coords[i+1,2]
      
      if (abs(x1-x2) < tol){
        segs[[k]] <- c(1, x1, min(y1,y2), max(y1,y2)); k<-k+1
      } else if (abs(y1-y2) < tol){
        segs[[k]] <- c(0, y1, min(x1,x2), max(x1,x2)); k<-k+1
      }
    }
    if (k==1) return(NULL)
    do.call(rbind, segs)
  }
  
  overlap_1d <- function(a1,a2,b1,b2){
    max(0, min(a2,b2) - max(a1,b1))
  }
  
  segments_touch <- function(segsA, segsB){
    if (is.null(segsA) || is.null(segsB)) return(FALSE)
    for(i in seq_len(nrow(segsA))){
      for(j in seq_len(nrow(segsB))){
        if (segsA[i,1] != segsB[j,1]) next
        if (abs(segsA[i,2] - segsB[j,2]) > distance_tol) next
        if (overlap_1d(segsA[i,3], segsA[i,4], segsB[j,3], segsB[j,4]) > min_shared_length)
          return(TRUE)
      }
    }
    FALSE
  }
  
  seg_list <- lapply(st_geometry(polys_sf), function(g){
    get_axis_segments(st_coordinates(g)[,1:2])
  })
  
  bbox_iou <- function(i,j){
    xa <- max(bbox_mat[i,1], bbox_mat[j,1])
    ya <- max(bbox_mat[i,2], bbox_mat[j,2])
    xb <- min(bbox_mat[i,3], bbox_mat[j,3])
    yb <- min(bbox_mat[i,4], bbox_mat[j,4])
    
    if (xa >= xb || ya >= yb) return(0)
    
    inter <- (xb-xa)*(yb-ya)
    ai <- (bbox_mat[i,3]-bbox_mat[i,1])*(bbox_mat[i,4]-bbox_mat[i,2])
    aj <- (bbox_mat[j,3]-bbox_mat[j,1])*(bbox_mat[j,4]-bbox_mat[j,2])
    
    inter/(ai+aj-inter)
  }
  
  centroid_dist <- function(i,j){
    dx <- centroids[i,1] - centroids[j,1]
    dy <- centroids[i,2] - centroids[j,2]
    sqrt(dx*dx + dy*dy)
  }
  
  # ===============================
  # PARALLEL EDGE BUILD
  # ===============================
  edge_chunks <- future_lapply(1:(n-1), function(i){
    
    axis_local <- list()
    approx_local <- list()
    k1 <- 1; k2 <- 1
    
    for(j in (i+1):n){
      
      if (centroid_dist(i,j) < distance_tol){
        if (segments_touch(seg_list[[i]], seg_list[[j]])){
          axis_local[[k1]] <- c(i,j)
          k1 <- k1 + 1
          next
        }
      }
      
      if (bbox_iou(i,j) < bbox_iou_threshold) next
      
      if (centroid_dist(i,j) < distance_tol / 6){
        approx_local[[k2]] <- c(i,j)
        k2 <- k2 + 1
      }
    }
    
    list(
      axis = if(length(axis_local)) do.call(rbind, axis_local) else NULL,
      approx = if(length(approx_local)) do.call(rbind, approx_local) else NULL
    )
  }, future.seed = TRUE)
  
  # ===============================
  # MERGE EDGES SAFELY
  # ===============================
  collect_edges <- function(lst, key){
    mats <- lapply(lst, function(x) x[[key]])
    mats <- mats[!sapply(mats, is.null)]
    if (length(mats) == 0) return(NULL)
    do.call(rbind, mats)
  }
  
  axis_edges   <- collect_edges(edge_chunks, "axis")
  approx_edges <- collect_edges(edge_chunks, "approx")
  
  edges <- rbind(axis_edges, approx_edges)
  
  # ===============================
  # GRAPH
  # ===============================
  if (is.null(edges)){
    polys_sf$group <- polys_sf$id
  } else {
    
    edges <- edges[complete.cases(edges), , drop = FALSE]
    
    g <- graph_from_edgelist(edges, directed = FALSE)
    comps <- components(g)$membership
    
    polys_sf$group <- polys_sf$id
    polys_sf$group[as.integer(names(comps))] <- comps
  }
  
  # ===============================
  # DEBUG VISUALS
  # ===============================
  make_edge_sf <- function(edges){
    if (is.null(edges)) return(NULL)
    
    do.call(rbind, lapply(seq_len(nrow(edges)), function(i){
      a <- edges[i,1]; b <- edges[i,2]
      st_sf(geometry = st_sfc(st_linestring(rbind(
        centroids[a,], centroids[b,]
      ))))
    }))
  }
  
  axis_sf <- make_edge_sf(axis_edges)
  approx_sf <- make_edge_sf(approx_edges)
  
  p1 <- ggplot() + geom_sf(data = polys_sf, fill = NA)
  p2 <- ggplot() + geom_sf(data = polys_sf, fill = NA) +
    geom_sf(data = axis_sf, colour="green")
  p3 <- ggplot() + geom_sf(data = polys_sf, fill = NA) +
    geom_sf(data = approx_sf, colour="blue")
  p4 <- ggplot() + geom_sf(data = polys_sf, aes(fill=factor(group))) +
    theme(legend.position="none")
  
  merged <- polys_sf %>%
    group_by(group) %>%
    summarise(geometry = st_union(geometry), .groups="drop")
  
  p5 <- ggplot() + geom_sf(data = merged, fill="red", alpha=0.4)
  
  list(raw=p1, axis=p2, approx=p3, components=p4, merged=p5)
}

data<-merge_debug_visual_parallel(output$polygons,output$metrics)
data$raw
data$axis
data$approx

data$merged
