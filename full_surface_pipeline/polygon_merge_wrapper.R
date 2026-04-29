#polygon merge wrapper

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
source("full_surface_pipeline/axis_aligned_graph_edge.R")


#######main_function
merge_output<-function(output){
filtered_output<-filter_polygons(polygons = ,df = )
axis_edges<-axis_merge_vectorised(filtered_output$polygons)
}