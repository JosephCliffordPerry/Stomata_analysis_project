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

####################################################
# Example usage
####################################################
# Example from your segmentation
original_poly <- stomata_list[["A-T2R3_Ab_1_frame_0000_obb_1.png"]][[1]][["segmentation_polygon"]]

# Interpolate to 100 points
interpolated_poly <- interpolate_polygon(original_poly, n_points = 100)

# Plot comparison
plot_polygons_compare(original_poly, interpolated_poly)
