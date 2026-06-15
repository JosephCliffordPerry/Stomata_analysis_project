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


merge_polygons_convex_hull <- function(
    polygons,
    axis_edges,
    max_group_size = 50,
    max_bbox_width = Inf,
    max_bbox_height = Inf,
    max_area = Inf
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(igraph)
  })
  
  # =====================================================
  # INPUT HANDLING
  # =====================================================
  
  if (inherits(polygons, "sf")) {
    
    geom_sf <- polygons
    
  } else if (inherits(polygons, "sfc")) {
    
    geom_sf <- st_sf(
      geometry = polygons
    )
    
  } else if (is.list(polygons)) {
    
    geom_sf <- st_sf(
      geometry = st_sfc(
        lapply(polygons, function(x){
          
          if (inherits(x, "POLYGON"))
            return(x)
          
          x <- as.matrix(x)
          
          if (any(x[1, ] != x[nrow(x), ])) {
            x <- rbind(x, x[1, ])
          }
          
          st_polygon(list(x))
        })
      )
    )
    
  } else {
    stop("Unsupported polygon format")
  }
  
  geom_sf <- st_make_valid(geom_sf)
  
  n <- nrow(geom_sf)
  
  if (n == 0) {
    stop("No polygons supplied")
  }
  
  # =====================================================
  # IMMUTABLE ORIGINAL IDS
  # =====================================================
  
  if (!"poly_id" %in% names(geom_sf)) {
    geom_sf$poly_id <- seq_len(n)
  }
  
  if (!"origin_ids" %in% names(geom_sf)) {
    geom_sf$origin_ids <- lapply(
      geom_sf$poly_id,
      identity
    )
  }
  
  # =====================================================
  # GRAPH
  # =====================================================
  
  if (is.null(axis_edges) || nrow(axis_edges) == 0) {
    
    g <- make_empty_graph(n = n)
    
  } else {
    
    edges_mat <- as.matrix(axis_edges[,1:2])
    
    g <- graph_from_edgelist(
      edges_mat,
      directed = FALSE
    )
    
    if (vcount(g) < n) {
      g <- add_vertices(g, n - vcount(g))
    }
  }
  
  comps <- components(g)$membership
  
  # =====================================================
  # SPLIT LARGE COMPONENTS
  # =====================================================
  
  split_groups <- list()
  
  for (grp in unique(comps)) {
    
    idx <- which(comps == grp)
    
    if (length(idx) <= max_group_size) {
      
      split_groups[[length(split_groups)+1]] <- idx
      next
    }
    
    cent <- st_coordinates(
      st_centroid(geom_sf[idx,])
    )
    
    ord <- order(
      cent[,1],
      cent[,2]
    )
    
    idx <- idx[ord]
    
    chunks <- split(
      idx,
      ceiling(seq_along(idx)/max_group_size)
    )
    
    split_groups <- c(split_groups, chunks)
  }
  
  # =====================================================
  # MERGE
  # =====================================================
  
  out_geom <- list()
  out_origin_ids <- list()
  
  for (i in seq_along(split_groups)) {
    
    ids <- split_groups[[i]]
    
    gsub <- geom_sf[ids,]
    
    merged <- tryCatch(
      suppressWarnings(
        st_union(gsub)
      ),
      error = function(e) NULL
    )
    
    if (is.null(merged))
      next
    
    hull <- tryCatch(
      st_convex_hull(merged),
      error = function(e) NULL
    )
    
    if (is.null(hull))
      next
    
    bb <- st_bbox(hull)
    
    width  <- bb$xmax - bb$xmin
    height <- bb$ymax - bb$ymin
    
    area <- as.numeric(
      st_area(hull)
    )
    
    too_big <-
      width  > max_bbox_width ||
      height > max_bbox_height ||
      area   > max_area
    
    # =================================================
    # REVERT TO INDIVIDUAL POLYGONS
    # =================================================
    
    if (too_big) {
      
      for (j in seq_along(ids)) {
        
        out_geom[[length(out_geom)+1]] <-
          st_geometry(geom_sf[ids[j],])[[1]]
        
        out_origin_ids[[length(out_origin_ids)+1]] <-
          geom_sf$origin_ids[[ids[j]]]
      }
      
    } else {
      
      merged_origins <- unique(
        unlist(
          geom_sf$origin_ids[ids],
          recursive = TRUE
        )
      )
      
      out_geom[[length(out_geom)+1]] <- hull[[1]]
      
      out_origin_ids[[length(out_origin_ids)+1]] <-
        merged_origins
    }
  }
  
  # =====================================================
  # OUTPUT
  # =====================================================
  
  out_sf <- st_sf(
    component_id = seq_along(out_geom),
    geometry = st_sfc(
      out_geom,
      crs = st_crs(geom_sf)
    )
  )
  
  out_sf$group_size <- sapply(
    out_origin_ids,
    length
  )
  
  out_sf$origin_ids <- out_origin_ids
  
  out_sf
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
    axis_tol = 1e-6,
    debug = TRUE
){
  
  suppressPackageStartupMessages({
    library(sf)
    library(FNN)
  })
  
  # =====================================================
  # SAFE INPUT
  # =====================================================
  
  if (inherits(polygons, "sf")) {
    
    sf_polys <- polygons
    
  } else if (inherits(polygons, "sfc")) {
    
    sf_polys <- st_sf(
      geometry = polygons
    )
    
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
    stop("Unsupported polygons format")
  }
  
  sf_polys <- st_make_valid(sf_polys)
  
  n <- nrow(sf_polys)
  
  # =====================================================
  # IDS
  # =====================================================
  
  if (!"poly_id" %in% names(sf_polys)) {
    sf_polys$poly_id <- seq_len(n)
  }
  
  if (!"origin_ids" %in% names(sf_polys)) {
    sf_polys$origin_ids <- lapply(
      sf_polys$poly_id,
      identity
    )
  }
  
  # =====================================================
  # ORIGINAL POLYGONS
  # =====================================================
  
  if (is.null(original_polygons)) {
    
    original_sf <- sf_polys
    
  } else if (inherits(original_polygons, "sf")) {
    
    original_sf <- original_polygons
    
  } else if (inherits(original_polygons, "sfc")) {
    
    original_sf <- st_sf(
      geometry = original_polygons
    )
    
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
      geometry = st_sfc(sfg_list)
    )
    
  } else {
    stop("Unsupported original_polygons format")
  }
  
  if (!"poly_id" %in% names(original_sf)) {
    original_sf$poly_id <- seq_len(nrow(original_sf))
  }
  
  # =====================================================
  # EDGE CASE
  # =====================================================
  
  if (n == 0) {
    
    return(list(
      isolated_polygons = sf_polys,
      remaining_original_polygons = original_sf,
      used_original_polygons = original_sf[0,],
      debug_table = NULL
    ))
  }
  
  # =====================================================
  # KNN
  # =====================================================
  
  cent <- st_coordinates(
    st_centroid(sf_polys)
  )
  
  knn <- FNN::get.knn(
    cent,
    k = min(k, max(1, n - 1))
  )
  
  # =====================================================
  # METRICS
  # =====================================================
  
  areas <- as.numeric(
    st_area(sf_polys)
  )
  
  isolated <- logical(n)
  
  # =====================================================
  # DEBUG STORAGE
  # =====================================================
  
  debug_df <- data.frame(
    poly_id = sf_polys$poly_id,
    area = areas,
    
    failed_area = FALSE,
    failed_axis = FALSE,
    failed_overlap = FALSE,
    
    max_overlap = 0,
    overlap_neighbor = NA_integer_,
    
    has_axis_edge = FALSE,
    accepted = FALSE,
    
    stringsAsFactors = FALSE
  )
  
  # =====================================================
  # ISOLATION TEST
  # =====================================================
  
  for (i in seq_len(n)) {
    
    g1 <- sf_polys$geometry[i]
    
    # ---------------------------------------------------
    # AREA FILTER
    # ---------------------------------------------------
    
    if (areas[i] < area_min) {
      
      debug_df$failed_area[i] <- TRUE
      
      if (debug)
        cat("[AREA FAIL] poly:", i,
            " area:", areas[i], "\n")
      
      next
    }
    
    # ---------------------------------------------------
    # AXIS EDGE FILTER
    # ---------------------------------------------------
    
    coords <- st_coordinates(g1)[,1:2,drop=FALSE]
    
    dx <- diff(coords[,1])
    dy <- diff(coords[,2])
    
    seg_len <- sqrt(dx^2 + dy^2)
    
    axis_mask <-
      seg_len >= axis_length_threshold &
      (
        abs(dx) <= axis_tol |
          abs(dy) <= axis_tol
      )
    
    has_axis_edge <- any(axis_mask)
    
    debug_df$has_axis_edge[i] <- has_axis_edge
    
    if (has_axis_edge) {
      
      debug_df$failed_axis[i] <- TRUE
      
      if (debug) {
        
        cat(
          "[AXIS FAIL] poly:", i,
          " max axis length:",
          max(seg_len[axis_mask], na.rm = TRUE),
          "\n"
        )
      }
      
      next
    }
    
    # ---------------------------------------------------
    # OVERLAP FILTER
    # ---------------------------------------------------
    
    nbrs <- knn$nn.index[i,]
    
    overlap_found <- FALSE
    max_overlap <- 0
    max_neighbor <- NA
    
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
      
      if (length(a) == 0)
        next
      
      overlap_val <- max(a, na.rm = TRUE)
      
      if (is.na(overlap_val))
        next
      
      # store max overlap
      if (overlap_val > max_overlap) {
        
        max_overlap <- overlap_val
        max_neighbor <- j
      }
      
      # reject
      if (overlap_val > overlap_tol) {
        
        overlap_found <- TRUE
        
        debug_df$failed_overlap[i] <- TRUE
        
        if (debug) {
          
          cat(
            "[OVERLAP FAIL] poly:", i,
            " neighbor:", j,
            " overlap:", overlap_val,
            "\n"
          )
        }
        
        break
      }
    }
    
    debug_df$max_overlap[i] <- max_overlap
    debug_df$overlap_neighbor[i] <- max_neighbor
    
    # ---------------------------------------------------
    # ACCEPT
    # ---------------------------------------------------
    
    isolated[i] <- !overlap_found
    
    if (!overlap_found) {
      
      debug_df$accepted[i] <- TRUE
      
      if (debug)
        cat("[ACCEPTED] poly:", i, "\n")
    }
  }
  
  # =====================================================
  # OUTPUT
  # =====================================================
  
  isolated_sf <- sf_polys[isolated,]
  
  used_ids <- unique(
    unlist(
      lapply(
        isolated_sf$origin_ids,
        as.integer
      ),
      recursive = TRUE
    )
  )
  
  used_original <- original_sf[
    original_sf$poly_id %in% used_ids,
  ]
  
  remaining_original <- original_sf[
    !original_sf$poly_id %in% used_ids,
  ]
  
  # =====================================================
  # SUMMARY
  # =====================================================
  
  if (debug) {
    
    cat("\n============================\n")
    cat("FILTER SUMMARY\n")
    cat("============================\n")
    
    cat(
      "Accepted:",
      sum(debug_df$accepted),
      "\n"
    )
    
    cat(
      "Area fails:",
      sum(debug_df$failed_area),
      "\n"
    )
    
    cat(
      "Axis fails:",
      sum(debug_df$failed_axis),
      "\n"
    )
    
    cat(
      "Overlap fails:",
      sum(debug_df$failed_overlap),
      "\n"
    )
  }
  
  list(
    isolated_polygons = isolated_sf,
    remaining_original_polygons = remaining_original,
    used_original_polygons = used_original,
    debug_table = debug_df
  )
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
axis_merge_vectorised <- function(
    polygons,
    k = 10,
    distance_tol = 5,
    min_shared_length = 10
){
  suppressPackageStartupMessages({
    library(sf)
    library(FNN)
  })
  
  # =====================================================
  # CLEAN INPUT
  # =====================================================
  
  sf_obj <- NULL
  
  if (inherits(polygons, "sf")) {
    sf_obj <- polygons
    
  } else if (inherits(polygons, "sfc")) {
    sf_obj <- st_sf(geometry = polygons)
    
  } else if (is.list(polygons)) {
    
    sfg <- lapply(polygons, function(p) {
      
      if (inherits(p, "sfg")) return(p)
      
      p <- as.matrix(p)
      
      if (nrow(p) < 3) return(NULL)
      
      if (sqrt(sum((p[1,] - p[nrow(p),])^2)) > 1e-6) {
        p <- rbind(p, p[1,])
      }
      
      st_polygon(list(p))
    })
    
    sf_obj <- st_sf(geometry = st_sfc(sfg))
    
  } else {
    stop("Unsupported input")
  }
  
  sf_obj <- st_make_valid(sf_obj)
  
  n <- nrow(sf_obj)
  if (n < 2) return(NULL)
  
  # =====================================================
  # CENTROIDS FOR KNN
  # =====================================================
  
  cent <- st_coordinates(st_centroid(sf_obj))
  
  k_use <- min(k, n - 1)
  nn <- FNN::get.knn(cent, k = k_use)
  
  # =====================================================
  # EDGE BUFFER (PREALLOCATED)
  # =====================================================
  
  max_edges <- n * k_use
  edges <- matrix(0L, nrow = max_edges, ncol = 2)
  e <- 1L
  
  # =====================================================
  # HELPER: EXTRACT AXIS SEGMENTS (ON DEMAND)
  # =====================================================
  
  get_axis_features <- function(g) {
    
    coords <- st_coordinates(g)[,1:2, drop = FALSE]
    
    out <- list()
    k <- 1L
    
    for (i in 1:(nrow(coords) - 1)) {
      
      x1 <- coords[i,1]; y1 <- coords[i,2]
      x2 <- coords[i+1,1]; y2 <- coords[i+1,2]
      
      # vertical
      if (abs(x1 - x2) < 1e-6) {
        out[[k]] <- list(
          orient = 1L,
          coord = x1,
          start = min(y1, y2),
          end = max(y1, y2)
        )
        k <- k + 1L
        
        # horizontal
      } else if (abs(y1 - y2) < 1e-6) {
        out[[k]] <- list(
          orient = 0L,
          coord = y1,
          start = min(x1, x2),
          end = max(x1, x2)
        )
        k <- k + 1L
      }
    }
    
    out[seq_len(k - 1L)]
  }
  
  axis_cache <- vector("list", n)
  for (i in seq_len(n)) {
    axis_cache[[i]] <- get_axis_features(sf_obj$geometry[[i]])
  }
  
  # =====================================================
  # MAIN KNN LOOP
  # =====================================================
  
  for (i in seq_len(n)) {
    
    gi <- sf_obj$geometry[[i]]
    ai <- axis_cache[[i]]
    
    if (length(ai) == 0) next
    
    for (j in nn$nn.index[i, ]) {
      
      if (i >= j) next
      
      gj <- sf_obj$geometry[[j]]
      aj <- axis_cache[[j]]
      
      if (length(aj) == 0) next
      
      # =================================================
      # AXIS COMPATIBILITY CHECK
      # =================================================
      
      match_found <- FALSE
      
      for (a in ai) {
        for (b in aj) {
          
          if (a$orient != b$orient) next
          
          if (abs(a$coord - b$coord) > distance_tol) next
          
          overlap <- max(
            0,
            min(a$end, b$end) - max(a$start, b$start)
          )
          
          if (overlap >= min_shared_length) {
            match_found <- TRUE
            break
          }
        }
        if (match_found) break
      }
      
      if (!match_found) next
      
      # =================================================
      # STORE EDGE
      # =================================================
      
      edges[e, ] <- c(i, j)
      e <- e + 1L
      
      if (e > max_edges) {
        stop("Edge buffer overflow: increase k or max_edges")
      }
    }
  }
  
  # =====================================================
  # FINAL OUTPUT
  # =====================================================
  
  if (e == 1L) return(NULL)
  
  edges <- edges[1:(e - 1), , drop = FALSE]
  
  unique(as.data.table(edges)[, .(id1 = V1, id2 = V2)])
}