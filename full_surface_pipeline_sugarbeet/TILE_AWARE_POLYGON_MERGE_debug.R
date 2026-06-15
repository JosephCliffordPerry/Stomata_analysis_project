merge_debug_visual <- function(polygons, df,
                               iou_threshold = 0.3,
                               bbox_iou_threshold = 0.05,
                               min_shared_length = 20,
                               distance_tol = 2,
                               precision = 1e3){
  
  suppressPackageStartupMessages({
    library(sf)
    library(dplyr)
    library(igraph)
    library(ggplot2)
  })
  # =====================================================
  # SIMPLE FILTERS (AREA + CIRCULARITY)
  # =====================================================
  
  # require expected columns
  stopifnot(all(c("area", "circularity") %in% colnames(df)))
  
  keep <- with(df,
               area >= 500 &  
                 area <= 4500 &
                 circularity >= 0.2 & 
                 circularity <= 1
  )
  
  
  
  polygons <- polygons[keep]
  df <- df[keep, ]
  
  # drop empty after filtering
  valid <- !sapply(polygons, is.null)
  polygons <- polygons[valid]
  df <- df[valid, ]
  # ===============================
  # CLEAN + BUILD SF
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
  
  # ===============================
  # SEGMENTS
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
    if (k==1) return(NULL)
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
        
        ov <- overlap_1d(s1[3], s1[4], s2[3], s2[4])
        if (ov > min_len) return(TRUE)
      }
    }
    FALSE
  }
  
  seg_list <- lapply(st_geometry(polys_sf), function(g){
    get_axis_segments(st_coordinates(g)[,1:2])
  })
  
  # ===============================
  # BBOX IOU
  # ===============================
  bbox_mat <- t(sapply(st_geometry(polys_sf), function(g){
    bb <- st_bbox(g)
    c(bb$xmin, bb$ymin, bb$xmax, bb$ymax)
  }))
  
  bbox_iou <- function(i,j){
    xa <- max(bbox_mat[i,1], bbox_mat[j,1])
    ya <- max(bbox_mat[i,2], bbox_mat[j,2])
    xb <- min(bbox_mat[i,3], bbox_mat[j,3])
    yb <- min(bbox_mat[i,4], bbox_mat[j,4])
    
    inter <- max(0, xb-xa) * max(0, yb-ya)
    if (inter==0) return(0)
    
    ai <- (bbox_mat[i,3]-bbox_mat[i,1])*(bbox_mat[i,4]-bbox_mat[i,2])
    aj <- (bbox_mat[j,3]-bbox_mat[j,1])*(bbox_mat[j,4]-bbox_mat[j,2])
    
    inter/(ai+aj-inter)
  }
  
  # ===============================
  # EDGE COLLECTION
  # ===============================
  axis_edges <- list()
  iou_edges  <- list()
  k1 <- 1; k2 <- 1
  
  n <- nrow(polys_sf)
  
  for(i in 1:n){
    for(j in 1:n){
      
      if (i >= j) next
      
      # --- AXIS ---
      if (st_distance(polys_sf[i,], polys_sf[j,]) < distance_tol){
        if (segments_touch(seg_list[[i]], seg_list[[j]], distance_tol, min_shared_length)){
          axis_edges[[k1]] <- c(i,j)
          k1 <- k1 + 1
          next
        }
      }
      
      # --- BBOX ---
      if (bbox_iou(i,j) < bbox_iou_threshold) next
      
      # --- IOU ---
      inter <- tryCatch(st_intersection(polys_sf[i,], polys_sf[j,]), error=function(e) NULL)
      if (is.null(inter) || nrow(inter)==0) next
      
      ia <- as.numeric(st_area(inter))
      if (ia == 0) next
      
      uni <- st_union(polys_sf[i,], polys_sf[j,])
      ua <- as.numeric(st_area(uni))
      
      if (ia/ua >= iou_threshold){
        iou_edges[[k2]] <- c(i,j)
        k2 <- k2 + 1
      }
    }
  }
  
  axis_edges <- if(length(axis_edges)>0) do.call(rbind, axis_edges) else NULL
  iou_edges  <- if(length(iou_edges)>0) do.call(rbind, iou_edges) else NULL
  
  cat("Axis edges:", ifelse(is.null(axis_edges),0,nrow(axis_edges)), "\n")
  cat("IoU edges:", ifelse(is.null(iou_edges),0,nrow(iou_edges)), "\n")
  
  # ===============================
  # GRAPH
  # ===============================
  edges <- rbind(axis_edges, iou_edges)
  
  if (is.null(edges)){
    polys_sf$group <- polys_sf$id
  } else {
    
    g <- graph_from_edgelist(matrix(as.character(edges), ncol=2), directed=FALSE)
    comps <- components(g)$membership
    
    polys_sf$group <- polys_sf$id
    polys_sf$group[as.integer(names(comps))] <- comps
  }
  
  # ===============================
  # VISUALS
  # ===============================
  
  # edge lines
  make_edge_sf <- function(edges){
    if (is.null(edges)) return(NULL)
    do.call(rbind, lapply(1:nrow(edges), function(i){
      a <- edges[i,1]; b <- edges[i,2]
      st_sf(geometry = st_sfc(st_linestring(rbind(
        st_coordinates(st_centroid(polys_sf[a,]))[1,1:2],
        st_coordinates(st_centroid(polys_sf[b,]))[1,1:2]
      ))))
    }))
  }
  
  axis_sf <- make_edge_sf(axis_edges)
  iou_sf  <- make_edge_sf(iou_edges)
  
  # ===============================
  # PLOTS
  # ===============================
  
  p1 <- ggplot() +
    geom_sf(data = polys_sf, fill = NA) +
    ggtitle("Raw polygons")
  
  p2 <- ggplot() +
    geom_sf(data = polys_sf, fill = NA, colour="grey") +
    geom_sf(data = axis_sf, colour="green", linewidth=0.7) +
    ggtitle("Axis-aligned merges")
  
  p3 <- ggplot() +
    geom_sf(data = polys_sf, fill = NA, colour="grey") +
    geom_sf(data = iou_sf, colour="blue", linewidth=0.7) +
    ggtitle("IoU merges")
  
  p4 <- ggplot() +
    geom_sf(data = polys_sf, aes(fill=factor(group)), colour=NA, alpha=0.6) +
    ggtitle("Graph components")+
    theme(legend.position = "none")
  
  merged <- polys_sf %>%
    group_by(group) %>%
    summarise(geometry = st_union(geometry), .groups="drop")
  
  p5 <- ggplot() +
    geom_sf(data = merged, fill="red", alpha=0.4, colour="yellow") +
    ggtitle("Final merged")
  
  list(
    raw = p1,
    axis = p2,
    iou = p3,
    components = p4,
    merged = p5
  )
}

plots <- merge_debug_visual(output$polygons,output$metrics)

plots$raw
plots$axis
plots$iou
plots$components
plots$merged
