merge_debug_visual_fast_bbox <- function(
    polygons,
    df,
    iou_threshold = 0.3,
    bbox_iou_threshold = 0.05,
    min_shared_length = 20,
    distance_tol = 2,
    precision = 1e3,
    use_exact_check = FALSE
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(dplyr)
    library(igraph)
    library(ggplot2)
  })
  
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
    return(list(raw=NULL, axis=NULL, approx=NULL, components=NULL, merged=NULL))
  }
  
  # ===============================
  # PRECOMPUTE
  # ===============================
  centroids <- st_centroid(polys_sf)
  areas <- as.numeric(st_area(polys_sf))
  
  bbox_mat <- t(sapply(st_geometry(polys_sf), function(g){
    bb <- st_bbox(g)
    c(bb$xmin, bb$ymin, bb$xmax, bb$ymax)
  }))
  
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
    c1 <- st_coordinates(centroids[i,])[1,1:2]
    c2 <- st_coordinates(centroids[j,])[1,1:2]
    sqrt(sum((c1-c2)^2))
  }
  
  # ===============================
  # AXIS SEGMENTS
  # ===============================
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
    
    if (k == 1) return(NULL)
    do.call(rbind, segs)
  }
  
  overlap_1d <- function(a1,a2,b1,b2){
    max(0, min(a2,b2) - max(a1,b1))
  }
  
  segments_touch <- function(segsA, segsB, tol, min_len){
    if (is.null(segsA) || is.null(segsB)) return(FALSE)
    
    for(i in seq_len(nrow(segsA))){
      s1 <- segsA[i,]
      for(j in seq_len(nrow(segsB))){
        s2 <- segsB[j,]
        if (s1[1] != s2[1]) next
        if (abs(s1[2] - s2[2]) > tol) next
        
        if (overlap_1d(s1[3], s1[4], s2[3], s2[4]) > min_len)
          return(TRUE)
      }
    }
    FALSE
  }
  
  seg_list <- lapply(st_geometry(polys_sf), function(g){
    get_axis_segments(st_coordinates(g)[,1:2])
  })
  
  # ===============================
  # EDGE COLLECTION
  # ===============================
  axis_edges <- list()
  approx_edges <- list()
  
  k1 <- 1
  k2 <- 1
  
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      
      # AXIS
      if (centroid_dist(i,j) < distance_tol){
        if (segments_touch(seg_list[[i]], seg_list[[j]],
                           distance_tol, min_shared_length)){
          axis_edges[[k1]] <- c(i,j)
          k1 <- k1 + 1
          next
        }
      }
      
      # BBOX FILTER
      if (bbox_iou(i,j) < bbox_iou_threshold) next
      
      # APPROX MERGE
      if (centroid_dist(i,j) < distance_tol * 6){
        approx_edges[[k2]] <- c(i,j)
        k2 <- k2 + 1
      }
    }
  }
  
  axis_edges <- if(length(axis_edges)) do.call(rbind, axis_edges) else NULL
  approx_edges <- if(length(approx_edges)) do.call(rbind, approx_edges) else NULL
  
  edges <- rbind(axis_edges, approx_edges)
  
  # ===============================
  # GRAPH
  # ===============================
  if (is.null(edges)){
    polys_sf$group <- polys_sf$id
  } else {
    
    edges <- as.matrix(edges)
    
    g <- graph_from_edgelist(
      matrix(as.character(edges), ncol = 2),
      directed = FALSE
    )
    
    comps <- components(g)$membership
    
    polys_sf$group <- polys_sf$id
    polys_sf$group[as.integer(names(comps))] <- comps
  }
  
  # ===============================
  # VISUALS
  # ===============================
  make_edge_sf <- function(edges){
    if (is.null(edges)) return(NULL)
    
    do.call(rbind, lapply(seq_len(nrow(edges)), function(i){
      a <- edges[i,1]; b <- edges[i,2]
      
      st_sf(geometry = st_sfc(st_linestring(rbind(
        st_coordinates(centroids[a,])[1,1:2],
        st_coordinates(centroids[b,])[1,1:2]
      ))))
    }))
  }
  
  axis_sf <- make_edge_sf(axis_edges)
  approx_sf <- make_edge_sf(approx_edges)
  
  p1 <- ggplot() +
    geom_sf(data = polys_sf, fill = NA) +
    ggtitle("Raw")
  
  p2 <- ggplot() +
    geom_sf(data = polys_sf, fill = NA, colour="grey") +
    geom_sf(data = axis_sf, colour="green") +
    ggtitle("Axis")
  
  p3 <- ggplot() +
    geom_sf(data = polys_sf, fill = NA, colour="grey") +
    geom_sf(data = approx_sf, colour="blue") +
    ggtitle("Approx")
  
  p4 <- ggplot() +
    geom_sf(data = polys_sf, aes(fill=factor(group)), colour=NA) +
    theme(legend.position="none") +
    ggtitle("Components")
  
  merged <- polys_sf %>%
    group_by(group) %>%
    summarise(geometry = st_union(geometry), .groups="drop")
  
  p5 <- ggplot() +
    geom_sf(data = merged, fill="red", alpha=0.4) +
    ggtitle("Merged")
  
  list(raw=p1, axis=p2, approx=p3, components=p4, merged=p5)
}

plots <- merge_debug_visual_fast_bbox(output$polygons,output$metrics)
plots$raw
plots$axis
plots$iou
plots$components
plots$merged
