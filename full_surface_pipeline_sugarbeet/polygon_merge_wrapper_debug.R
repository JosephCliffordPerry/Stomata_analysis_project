run_iterative_polygon_pipeline <- function(
    polygons,
    parameter_sets,
    img = NULL,
    debug = TRUE
){
  
  suppressPackageStartupMessages({
    library(sf)
  })
  
  # =====================================================
  # INPUT NORMALISATION
  # =====================================================
  
  if (inherits(polygons, "sf")) {
    
    remaining_polygons <- polygons
    
  } else if (inherits(polygons, "sfc")) {
    
    remaining_polygons <- st_sf(
      node_id = seq_along(polygons),
      geometry = polygons
    )
    
  } else if (is.list(polygons)) {
    
    sfg <- lapply(seq_along(polygons), function(i){
      
      p <- as.matrix(polygons[[i]])
      
      if (any(p[1,] != p[nrow(p),])) {
        p <- rbind(p, p[1,])
      }
      
      st_polygon(list(p))
    })
    
    remaining_polygons <- st_sf(
      node_id = seq_along(sfg),
      geometry = st_sfc(sfg)
    )
    
  } else {
    stop("Unsupported polygon format")
  }
  
  remaining_polygons <- st_make_valid(remaining_polygons)
  
  # =====================================================
  # STORAGE
  # =====================================================
  
  accumulated_isolated <- list()
  debug_output <- list()
  
  # =====================================================
  # ITERATIVE PIPELINE
  # =====================================================
  
  for (iter in seq_along(parameter_sets)) {
    
    params <- parameter_sets[[iter]]
    
    cat("\n=============================\n")
    cat("ITERATION:", iter, "\n")
    cat("=============================\n")
    
    if (nrow(remaining_polygons) == 0) {
      cat("No polygons remaining\n")
      break
    }
    
    # =================================================
    # DEBUG START
    # =================================================
    
    if (debug && !is.null(img)) {
      
      plot_sf_overlay(
        img = img,
        sf_polys = remaining_polygons,
        alpha = 0.3,
        line_col = "red"
      )
    }
    
    # =================================================
    # AXIS MERGE
    # =================================================
    
    axis_edges <- axis_merge_vectorised(
      remaining_polygons$geometry,
      distance_tol = params$axis_distance_tol,
      min_shared_length = params$axis_min_shared_length
    )
    
    stage1_sf <- merge_polygons_convex_hull(
      polygons = remaining_polygons,
      axis_edges = axis_edges,
      max_group_size = params$max_group_size,
      max_bbox_width = params$max_bbox_width,
      max_bbox_height = params$max_bbox_height,
      max_area = params$max_area
    )
    
    if (debug && !is.null(img)) {
      
      plot_sf_overlay(
        img = img,
        sf_polys = stage1_sf,
        alpha = 0.4,
        line_col = "yellow"
      )
    }
    
    # =================================================
    # BBOX / IOU MERGE
    # =================================================
    
    bbox_edges <- build_bbox_iou_edges(
      stage1_sf,
      centroid_dist_tol = params$bbox_centroid_dist_tol,
      bbox_iou_threshold = params$bbox_iou_threshold
    )
    
    merged_sf <- merge_polygons_convex_hull(
      polygons = stage1_sf,
      axis_edges = bbox_edges,
      max_group_size = params$max_group_size,
      max_bbox_width = params$max_bbox_width,
      max_bbox_height = params$max_bbox_height,
      max_area = params$max_area
    )
    
    if (debug && !is.null(img)) {
      
      plot_sf_overlay(
        img = img,
        sf_polys = merged_sf,
        alpha = 0.4,
        line_col = "cyan"
      )
    }
    
    # =================================================
    # SPLIT / REMOVE
    # =================================================
    
    split_result <- split_isolated_polygons(
      polygons = merged_sf,
      original_polygons = remaining_polygons,
      k = params$split_k,
      overlap_tol = params$split_overlap_tol,
      area_min = params$split_area_min,
      axis_length_threshold = params$split_axis_length_threshold,
      axis_tol = params$split_axis_tol
    )
    
    isolated_n <- nrow(split_result$isolated_polygons)
    remaining_n <- nrow(split_result$remaining_original_polygons)
    
    cat("Isolated polygons :", isolated_n, "\n")
    cat("Remaining polygons:", remaining_n, "\n")
    
    # =================================================
    # DEBUG PLOTS
    # =================================================
    
    if (debug && !is.null(img)) {
      
      plot_sf_overlay(
        img = img,
        sf_polys = split_result$isolated_polygons,
        alpha = 0.5,
        line_col = "green"
      )
      
      plot_sf_overlay(
        img = img,
        sf_polys = split_result$remaining_original_polygons,
        alpha = 0.5,
        line_col = "magenta"
      )
      
      plot_isolation_debug(
        img = img,
        original_polygons = remaining_polygons,
        used_original_polygons = split_result$used_original_polygons,
        remaining_original_polygons = split_result$remaining_original_polygons,
        isolated_polygons = split_result$isolated_polygons
      )
    }
    
    # =================================================
    # SAVE DEBUG OBJECT
    # =================================================
    
    dbg <- list(
      iteration = iter,
      params = params,
      axis_edges = axis_edges,
      bbox_edges = bbox_edges,
      stage1_sf = stage1_sf,
      merged_sf = merged_sf,
      split_result = split_result
    )
    
    debug_output[[length(debug_output) + 1]] <- dbg
    
    # =================================================
    # STOP CONDITION
    # =================================================
    
    if (isolated_n == 0) {
      cat("Stopping: no isolated polygons found\n")
      break
    }
    
    # =================================================
    # ACCUMULATE RESULTS
    # =================================================
    
    accumulated_isolated[[length(accumulated_isolated) + 1]] <-
      split_result$isolated_polygons
    
    # =================================================
    # UPDATE REMAINING
    # =================================================
    
    remaining_polygons <- split_result$remaining_original_polygons
    
    if (nrow(remaining_polygons) == 0) {
      cat("Stopping: all polygons consumed\n")
      break
    }
  }
  
  # =====================================================
  # FINAL COMBINE
  # =====================================================
  
  if (length(accumulated_isolated) > 0) {
    
    final_isolated <- do.call(
      rbind,
      accumulated_isolated
    )
    
  } else {
    
    final_isolated <- remaining_polygons[0,]
  }
  
  # =====================================================
  # FINAL DEBUG
  # =====================================================
  
  if (debug && !is.null(img)) {
    
    plot_sf_overlay(
      img = img,
      sf_polys = final_isolated,
      alpha = 0.4,
      line_col = "green"
    )
  }
  
  # =====================================================
  # RETURN
  # =====================================================
  
  list(
    isolated_polygons = final_isolated,
    remaining_polygons = remaining_polygons,
    debug = debug_output
  )
}



# =========================================================
# PARAMETER SCHEDULE
# =========================================================

parameter_sets <- list(
  
  # =====================================================
  # PASS 1
  # =====================================================
  
  list(
    
    # axis merge
    axis_distance_tol = 2,
    axis_min_shared_length = 5,
    
    # bbox merge
    bbox_centroid_dist_tol = 64,
    bbox_iou_threshold = 0.8,
    
    # split
    split_k = 10,
    split_overlap_tol = 20,
    split_area_min = 500,
    split_axis_length_threshold = 50,
    split_axis_tol = 1e-6,
    
    # merge guards
    max_group_size = 50,
    max_bbox_width = Inf,
    max_bbox_height = Inf,
    max_area = Inf
  ),
  
  # =====================================================
  # PASS 2
  # =====================================================
  
  list(
    
    # axis merge
    axis_distance_tol = 5,
    axis_min_shared_length = 10,
    
    # bbox merge
    bbox_centroid_dist_tol = 64,
    bbox_iou_threshold = 0.5,
    
    # split
    split_k = 10,
    split_overlap_tol = 300,
    split_area_min = 500,
    split_axis_length_threshold = 50,
    split_axis_tol = 1e-6,
    
    # merge guards
    max_group_size = 50,
    max_bbox_width = Inf,
    max_bbox_height = Inf,
    max_area = Inf
  ),
  # =====================================================
  # PASS 3
  # =====================================================
  
  list(
    
    # axis merge
    axis_distance_tol = 5,
    axis_min_shared_length = 5,
    
    # bbox merge
    bbox_centroid_dist_tol = 128,
    bbox_iou_threshold = 0.3,
    
    # split
    split_k = 10,
    split_overlap_tol = 300,
    split_area_min = 500,
    split_axis_length_threshold = 50,
    split_axis_tol = 1e-6,
    
    # merge guards
    max_group_size = 50,
    max_bbox_width = Inf,
    max_bbox_height = Inf,
    max_area = 10000
  )
)



# =========================================================
# RUN PIPELINE
# =========================================================

pipeline_result <- run_iterative_polygon_pipeline(
  polygons = polygons,
  parameter_sets = parameter_sets,
  img = img,
  debug = TRUE
)



# =========================================================
# FINAL OUTPUT PLOTS
# =========================================================

plot_sf_overlay(
  img = img,
  sf_polys = pipeline_result$isolated_polygons,
  alpha = 0.4,
  line_col = "green"
)

plot_sf_overlay(
  img = img,
  sf_polys = pipeline_result$remaining_polygons,
  alpha = 0.4,
  line_col = "red"
)


plot_sf_overlay(
  img = img,
  sf_polys = pipeline_result[["debug"]][[3]][["merged_sf"]],
  alpha = 0.4,
  line_col = "yellow"
)
plot_sf_overlay(
  img = img,
  sf_polys = pipeline_result[["debug"]][[3]][["split_result"]][["isolated_polygons"]],
  alpha = 0.4,
  line_col = "yellow"
)
