library(dplyr)
####################################################
# Interpolate a polygon to evenly spaced points
# Input: 
#   poly_mat - n x 2 matrix of x,y coordinates
#   n_points - number of points to interpolate
# Output:
#   interpolated polygon as n_points x 2 matrix
####################################################
interpolate_polygon <- function(poly_mat, n_points = 100) {
  if (!is.matrix(poly_mat) || ncol(poly_mat) != 2) {
    stop("poly_mat must be an n x 2 matrix of x,y coordinates")
  }
  
  # -------------------
  # 1. Find longest axis
  # -------------------
  dists <- as.matrix(dist(poly_mat))
  max_idx <- which(dists == max(dists), arr.ind = TRUE)[1,]
  p1_idx <- max_idx[1]
  
  # rotate points so p1 is first
  if (p1_idx > 1) {
    poly_mat <- rbind(poly_mat[p1_idx:nrow(poly_mat), , drop = FALSE],
                      poly_mat[1:(p1_idx-1), , drop = FALSE])
  }
  
  # -------------------
  # 2. cumulative distances along edges
  # -------------------
  diffs <- diff(rbind(poly_mat, poly_mat[1,]))  # close polygon
  seg_lengths <- sqrt(rowSums(diffs^2))
  cum_lengths <- c(0, cumsum(seg_lengths))
  
  # -------------------
  # 3. interpolate x and y at evenly spaced distances
  # -------------------
  total_length <- cum_lengths[length(cum_lengths)]
  new_lengths <- seq(0, total_length, length.out = n_points + 1)[-1]  # remove duplicate
  interp_x <- approx(cum_lengths, c(poly_mat[,1], poly_mat[1,1]), xout = new_lengths)$y
  interp_y <- approx(cum_lengths, c(poly_mat[,2], poly_mat[1,2]), xout = new_lengths)$y
  
  return(cbind(interp_x, interp_y))
}

####################################################
# Plot original and interpolated polygons
# Input:
#   poly_orig - n x 2 matrix
#   poly_interp - m x 2 matrix
####################################################
library(ggplot2)

plot_polygons_compare <- function(poly_orig, poly_interp) {
  if (!is.matrix(poly_orig) || ncol(poly_orig) != 2) stop("poly_orig must be n x 2")
  if (!is.matrix(poly_interp) || ncol(poly_interp) != 2) stop("poly_interp must be n x 2")
  
  orig_df <- data.frame(
    x = poly_orig[,1], 
    y = poly_orig[,2], 
    type = "Original", 
    order = seq_len(nrow(poly_orig))
  )
  interp_df <- data.frame(
    x = poly_interp[,1],
    y = poly_interp[,2],
    type = "Interpolated",
    order = seq_len(nrow(poly_interp))
  )
  
  poly_df <- rbind(orig_df, interp_df)
  
  ggplot(poly_df, aes(x = x, y = y, color = type)) +
    geom_path(aes(group = type), size = 1.2) +
    geom_point(size = 1) +
    coord_equal() +
    theme_minimal() +
    labs(title = "Polygon Comparison: Original vs Interpolated")
}

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
# MAIN FUNCTION: flatten all cell polygons → dataframe
# -----------------------------------------------------------
interpolate_cell_list <- function(cell_list,metrics_list, n_points = 100) {
  
  results <- list()
  row_id <- 1
  

    
    for (i in seq_along(cell_list)) {
      
      obj <- cell_list[[i]]
      
      area_val <- polygon_area(obj)
      
      interp <- interpolate_polygon(obj, n_points = n_points)
      
      # assemble output row
      df_row <- data.frame(
        image_id     = names(cell_list[i]),
        cell_id   = i,
        area         = metrics_list$area[[i]],
        circularity= metrics_list$circularity[[i]],
        t(interp[,1]),
        t(interp[,2])
      )
      
      colnames(df_row) <- c(
        "image_id", "cell_id", "area","circularity",
        paste0("x", 1:n_points),
        paste0("y", 1:n_points)
      )
      
      results[[row_id]] <- df_row
      row_id <- row_id + 1
    
  }
  
  bind_rows(results)
}
#stomata_stats_helper
#
library(dplyr)
library(tidyr)

add_cell_geometry_columns <- function(df) {

  # ---- Step 1: wide → long ----
  long <- df %>%
    mutate(row_id = row_number()) %>%
    pivot_longer(
      cols = matches("^[xy]\\d+$"),
      names_to = c("coord", "idx"),
      names_pattern = "([xy])(\\d+)",
      values_to = "value"
    ) %>%
    pivot_wider(
      names_from = coord,
      values_from = value
    ) %>%
    mutate(idx = as.integer(idx))
  
  # ---- Step 2: ZR geometry ----
  long <- long %>%
    group_by(row_id) %>%
    arrange(idx, .by_group = TRUE) %>%
    mutate(
      # ---- centroid ----
      cx = mean(x, na.rm = TRUE),
      cy = mean(y, na.rm = TRUE),
      
      # ---- Radius (ZR) ----
      Radius_profile_ = sqrt((x - cx)^2 + (y - cy)^2),
      
      # ---- Previous + Next (circular) ----
      x_prev = lag(x, default = last(x)),
      y_prev = lag(y, default = last(y)),
      x_next = lead(x, default = first(x)),
      y_next = lead(y, default = first(y)),
      
      # ---- Tangent vectors ----
      vx_prev = x - x_prev,
      vy_prev = y - y_prev,
      vx_curr = x_next - x,
      vy_curr = y_next - y,
      
      # ---- Dot and cross for ZR turning Angle_profile_ ----
      dotp  = vx_prev * vx_curr + vy_prev * vy_curr,
      cross = vx_prev * vy_curr - vy_prev * vx_curr,
      
      # ---- ZR turning Angle_profile_ (0–360 external turning Angle_profile_) ----
      theta = atan2(cross, dotp) * 180 / pi,
      Angle_profile_ = ifelse(theta < 0, theta + 360, theta)
    ) %>%
    ungroup()
  
  
  # ---- Step 3: Opposite-point diameter (ZR width function) ----
  long <- long %>%
    group_by(row_id) %>%
    mutate(
      idx_op = ifelse(idx <= 50, idx + 50, idx - 50)
    ) %>%
    left_join(
      long %>% select(row_id, idx, x_op = x, y_op = y),
      by = c("row_id", "idx_op" = "idx")
    ) %>%
    mutate(
      Diameter_profile_ = sqrt((x - x_op)^2 + (y - y_op)^2)
    ) %>%
    ungroup()
  
  
  # ---- Step 4: widen to Angle_profile_1–100, radius1–100, opp_diam1–100 ----
  geom_wide <- long %>%
    select(row_id, idx, Angle_profile_, Radius_profile_, Diameter_profile_) %>%
    pivot_wider(
      names_from = idx,
      values_from = c(Angle_profile_, Radius_profile_, Diameter_profile_),
      names_glue = "{.value}{idx}"
    )
  
  # ---- Step 5: join back to df ----
  df %>%
    mutate(row_id = row_number()) %>%
    left_join(geom_wide, by = "row_id") %>%
    select(-row_id)
}





