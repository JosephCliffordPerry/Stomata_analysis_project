source("Stomata_only_pipeline/yolo_polygon_interpolate.R")
# -----------------------------------------------------------
# POLYGON AREA (signed shoelace)
# -----------------------------------------------------------
polygon_area <- function(poly_mat) {
  x <- poly_mat[, 1]
  y <- poly_mat[, 2]
  # close polygon
  x2 <- c(x, x[1])
  y2 <- c(y, y[1])
  abs(sum(x2[-1] * y2[-length(y2)] - x2[-length(x2)] * y2[-1])) / 2
}

# -----------------------------------------------------------
# MAIN FUNCTION: flatten all stomata polygons → dataframe
# -----------------------------------------------------------
interpolate_stomata_list <- function(stomata_list, n_points = 100) {
  
  results <- list()
  row_id <- 1
  
  for (image_name in names(stomata_list)) {
    
    objs <- stomata_list[[image_name]]
    
    for (i in seq_along(objs)) {
      
      obj <- objs[[i]]
      
      poly_mat <- obj$segmentation_polygon
      area_val <- polygon_area(poly_mat)
      
      interp <- interpolate_polygon(poly_mat, n_points = n_points)
      
      # assemble output row
      df_row <- data.frame(
        image_id     = image_name,
        stomata_id   = i,
        feature_type = obj$class_id,  # 0 = pore, 1 = guard
        area         = area_val,
        t(interp[,1]),
        t(interp[,2])
      )
      
      colnames(df_row) <- c(
        "image_id", "stomata_id", "feature_type", "area",
        paste0("x", 1:n_points),
        paste0("y", 1:n_points)
      )
      
      results[[row_id]] <- df_row
      row_id <- row_id + 1
    }
  }
  
  bind_rows(results)
}

# -----------------------------------------------------------
# Example Usage
# # -----------------------------------------------------------
# df <- interpolate_stomata_list(stomata_list, n_points = 100)
# head(df)