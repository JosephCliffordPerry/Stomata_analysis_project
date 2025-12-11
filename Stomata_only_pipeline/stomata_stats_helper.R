#stomata_stats_helper
#
library(dplyr)


add_stomata_geometry_columns <- function(df) {
  df <- stomata_df %>%
    # ensure feature_type is numeric (if it's a string)
    mutate(feature_type = as.numeric(feature_type)) %>%
    group_by(image_id) %>%
    mutate(
      has0 = any(feature_type == 0, na.rm = TRUE),
      has1 = any(feature_type == 1, na.rm = TRUE),
      status = case_when(
        has0 & has1            ~ "open",
        has0 & !has1           ~ "potential error",
        !has0 & has1           ~ "closed",
        TRUE                   ~ NA_character_
      )
    ) %>%
    ungroup() %>%
    select(-has0, -has1)
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
      radius = sqrt((x - cx)^2 + (y - cy)^2),
      
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
      
      # ---- Dot and cross for ZR turning angle ----
      dotp  = vx_prev * vx_curr + vy_prev * vy_curr,
      cross = vx_prev * vy_curr - vy_prev * vx_curr,
      
      # ---- ZR turning angle (0–360 external turning angle) ----
      theta = atan2(cross, dotp) * 180 / pi,
      angle = ifelse(theta < 0, theta + 360, theta)
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
      opposite_point_diameter = sqrt((x - x_op)^2 + (y - y_op)^2)
    ) %>%
    ungroup()
  
  
  # ---- Step 4: widen to angle1–100, radius1–100, opp_diam1–100 ----
  geom_wide <- long %>%
    select(row_id, idx, angle, radius, opposite_point_diameter) %>%
    pivot_wider(
      names_from = idx,
      values_from = c(angle, radius, opposite_point_diameter),
      names_glue = "{.value}{idx}"
    )
  
  # ---- Step 5: join back to df ----
  df %>%
    mutate(row_id = row_number()) %>%
    left_join(geom_wide, by = "row_id") %>%
    select(-row_id)
}

# 
# # Consensuses -------------------------------------------------------------
# 
# 
# 
# MakeStomataConsensus <- function(data, groups = NULL, highlight = NULL) {
#   
#   # ---- checks ----
#   if (!any(grepl("^x\\d+$", names(data)))) {
#     stop("Missing x1–x100 columns")
#   }
#   if (!any(grepl("^y\\d+$", names(data)))) {
#     stop("Missing y1–y100 columns")
#   }
#   
#   # ---- extract outlines (points × specimens) ----
#   Outlinex <- t(data %>% dplyr::select(matches("^x\\d+$")))
#   Outliney <- t(data %>% dplyr::select(matches("^y\\d+$")))
#   
#   n_points <- nrow(Outlinex)
#   
#   # ---- grouped consensus ----
#   if (!is.null(groups)) {
#     
#     if (length(groups) != ncol(Outlinex)) {
#       stop("Length of groups must match number of outlines")
#     }
#     
#     plot_data <- purrr::map_dfr(unique(groups), function(g) {
#       
#       idx <- which(groups == g)
#       
#       data.frame(
#         x = rowMeans(Outlinex[, idx, drop = FALSE]),
#         y = rowMeans(Outliney[, idx, drop = FALSE]),
#         group = as.factor(g),
#         point = 1:n_points
#       )
#     })
#     
#     p <- ggplot(plot_data, aes(x = x, y = y, group = group, fill = group)) +
#       geom_polygon(alpha = 0.5, color = "black") +
#       facet_wrap(~group) +
#       coord_equal() +
#       theme_minimal() +
#       labs(title = "Consensus Stomata Outlines")
#     
#   } else {
#     
#     # ---- global consensus ----
#     plot_data <- data.frame(
#       x = rowMeans(Outlinex),
#       y = rowMeans(Outliney),
#       point = 1:n_points
#     )
#     
#     p <- ggplot(plot_data, aes(x = x, y = y)) +
#       geom_polygon(alpha = 0.7, fill = "grey70", color = "black") +
#       coord_equal() +
#       theme_minimal() +
#       labs(title = "Global Consensus Stomata Outline")
#   }
#   
#   # ---- optional highlighted points ----
#   if (!is.null(highlight)) {
#     p <- p +
#       geom_point(
#         data = subset(plot_data, point %in% highlight),
#         aes(x = x, y = y),
#         color = "red",
#         size = 2
#       )
#   }
#   
#   return(p)
# }
# 
# make_Stomata_profile_graphs <- function(data,
#                                         groups = NULL,
#                                         positions = 100,
#                                         profile_type = NULL) {
#   
#   # Map user-facing profile names to column prefixes in the dataset
#   profile_map <- list(
#     Angle = "angle",
#     Radius = "radius",
#     OppositeDiameter = "opposite_point_diameter"
#   )
#   
#   # If a single profile type requested, keep only that
#   if (!is.null(profile_type)) {
#     if (!profile_type %in% names(profile_map)) {
#       stop("profile_type must be one of: ", paste(names(profile_map), collapse = ", "))
#     }
#     profile_map <- profile_map[profile_type]
#   }
#   
#   # Build profiles for each requested profile type
#   all_profiles <- lapply(names(profile_map), function(pt) {
#     prefix <- profile_map[[pt]]
#     
#     # select columns like angle1..angle100
#     prof_data <- data %>% dplyr::select(matches(paste0("^", prefix, "\\d+$")))
#     
#     if (ncol(prof_data) < positions) {
#       stop("Not enough profile columns for ", pt)
#     }
#     
#     # With groups (clusters)
#     if (!is.null(groups)) {
#       if (length(groups) != nrow(prof_data)) {
#         stop("Length of groups must match number of rows")
#       }
#       
#       combined <- cbind(prof_data, cluster = groups)
#       
#       purrr::map_dfr(unique(groups), function(g) {
#         subset <- combined[combined$cluster == g, 1:positions, drop = FALSE]
#         
#         data.frame(
#           x = seq_len(positions),
#           y = apply(subset, 2, median, na.rm = TRUE),
#           ymin = apply(subset, 2, quantile, probs = 0.25, na.rm = TRUE),
#           ymax = apply(subset, 2, quantile, probs = 0.75, na.rm = TRUE),
#           group = as.factor(g),
#           type = pt,
#           stringsAsFactors = FALSE
#         )
#       })
#       
#     } else {
#       # No groups: pooled profile
#       subset <- prof_data[, 1:positions, drop = FALSE]
#       
#       data.frame(
#         x = seq_len(positions),
#         y = apply(subset, 2, median, na.rm = TRUE),
#         ymin = apply(subset, 2, quantile, probs = 0.25, na.rm = TRUE),
#         ymax = apply(subset, 2, quantile, probs = 0.75, na.rm = TRUE),
#         group = "All",
#         type = pt,
#         stringsAsFactors = FALSE
#       )
#     }
#   })
#   
#   profile_df <- dplyr::bind_rows(all_profiles)
#   
#   # Plot
#   p <- ggplot(profile_df, aes(x = x, y = y, color = group, fill = group)) +
#     geom_ribbon(aes(ymin = ymin, ymax = ymax), alpha = 0.2, color = NA) +
#     geom_line(size = 1.1) +
#     labs(
#       x = "Contour Position",
#       y = "Value",
#       color = if (is.null(groups)) "" else "Group",
#       fill  = if (is.null(groups)) "" else "Group"
#     ) +
#     theme_minimal()
#   
#   if (length(profile_map) > 1) {
#     p <- p + facet_wrap(~type, scales = "free_y")
#   }
#   
#   return(p)
# }
# 
# 
# 
# 
# # consensus outline
# MakeStomataConsensus(df)
# 
# # profiles
# make_Stomata_profile_graphs(df_with_geometry)
# 
# 


