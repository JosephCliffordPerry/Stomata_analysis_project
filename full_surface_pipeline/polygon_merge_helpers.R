filter_polygons <- function(
    polygons,
    df,
    area_min = 500,
    area_max = 4500,
    circ_min = 0.2,
    circ_max = 1
){
  
  # -------------------------------
  # Validate inputs
  # -------------------------------
  stopifnot(length(polygons) == nrow(df))
  stopifnot(all(c("area", "circularity") %in% colnames(df)))
  
  # -------------------------------
  # Metric filter
  # -------------------------------
  keep_metrics <- with(df,
                       area >= area_min &
                         area <= area_max &
                         circularity >= circ_min &
                         circularity <= circ_max)
  
  polygons <- polygons[keep_metrics]
  df <- df[keep_metrics, , drop = FALSE]
  
  # -------------------------------
  # Geometry validity filter
  # -------------------------------
  valid_geom <- sapply(polygons, function(p){
    
    if (is.null(p)) return(FALSE)
    
    # sfg polygon
    if (inherits(p, "sfg")) {
      return(length(p) > 0 && nrow(p[[1]]) >= 3)
    }
    
    # matrix fallback
    if (is.numeric(p) && !is.null(dim(p))) {
      return(nrow(p) >= 3 && ncol(p) >= 2)
    }
    
    FALSE
  })
  
  polygons <- polygons[valid_geom]
  df <- df[valid_geom, , drop = FALSE]
  
  # -------------------------------
  # Return
  # -------------------------------
  list(
    polygons = polygons,
    df = df
  )
}

merge_polygons_convex_hull <- function(polygons, axis_edges, precision = 1e3){
  
  suppressPackageStartupMessages({
    library(sf)
    library(dplyr)
    library(igraph)
  })
  
  # ===============================
  # CLEAN → SF
  # ===============================
  cleaned <- lapply(seq_along(polygons), function(i){
    
    p <- polygons[[i]]
    
    # already sfg
    if (inherits(p, "sfg")) {
      return(p)
    }
    
    # matrix case
    if (
      is.null(p) ||
      !is.numeric(p) ||
      is.null(dim(p)) ||
      nrow(p) < 3 ||
      ncol(p) < 2
    ) return(NULL)
    
    p <- p[complete.cases(p), , drop = FALSE]
    if (nrow(p) < 3) return(NULL)
    
    if (sqrt(sum((p[1,] - p[nrow(p),])^2)) > 1e-6)
      p <- rbind(p, p[1,])
    
    if (nrow(p) < 4) return(NULL)
    
    p[nrow(p),] <- p[1,]
    
    tryCatch(
      st_polygon(list(p)),
      error = function(e) NULL
    )
  })
  
  valid <- !sapply(cleaned, is.null)
  if (!any(valid)) return(NULL)
  
  polys_sf <- st_sf(
    id = which(valid),
    geometry = st_sfc(cleaned[valid])
  )
  
  polys_sf <- st_set_precision(polys_sf, precision)
  polys_sf <- st_make_valid(polys_sf)
  
  # ===============================
  # GRAPH COMPONENTS
  # ===============================
  if (is.null(axis_edges) || nrow(axis_edges) == 0){
    
    polys_sf$group <- polys_sf$id
    
  } else {
    
    edges <- matrix(as.integer(axis_edges), ncol = 2)
    edges <- edges[complete.cases(edges), , drop = FALSE]
    
    edges <- edges[
      edges[,1] %in% polys_sf$id &
        edges[,2] %in% polys_sf$id,
      , drop = FALSE
    ]
    
    if (nrow(edges) == 0){
      
      polys_sf$group <- polys_sf$id
      
    } else {
      
      g <- graph_from_edgelist(edges, directed = FALSE)
      comps <- components(g)$membership
      
      polys_sf$group <- polys_sf$id
      polys_sf$group[as.integer(names(comps))] <- comps
    }
  }
  
  # ===============================
  # CONVEX HULL MERGE
  # ===============================
  merged_list <- lapply(split(polys_sf, polys_sf$group), function(group_sf){
    
    # union first (collect all geometry)
    geom_union <- st_union(group_sf$geometry)
    
    # convex hull of union
    hull <- st_convex_hull(geom_union)
    
    # FIX: ensure sfc
    hull_sfc <- st_sfc(hull)
    
    st_sf(
      group = unique(group_sf$group),
      geometry = hull_sfc
    )
  })
  
  merged <- do.call(rbind, merged_list)
  
 
}