plot_processed_tile <- function(tile_processed) {
  
  # --- read image
  img <- tile_processed$image
  if (is.null(img)) stop("Tile image missing")
  
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  g <- rasterGrob(
    img,
    width = unit(1, "npc"),
    height = unit(1, "npc"),
    interpolate = FALSE
  )
  
  # --- polygons
  polys <- tile_processed$combined_polygons
  if (nrow(polys) == 0) {
    warning("No polygons to plot")
  }
  
  # convert geometry to plotting coordinates
  poly_df <- do.call(rbind, lapply(seq_len(nrow(polys)), function(i) {
    coords <- st_coordinates(polys[i, ])
    data.frame(
      x = coords[,1],
      y = h - coords[,2],        # invert Y for image coordinates
      type = polys$type[i],
      poly_id = i
    )
  }))
  
  # color scheme
  fill_cols <- c(
    object = "#1f77b4",  # blue
    gap    = "#ff7f0e"   # orange
  )
  
  # --- plot
  ggplot() +
    annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
    geom_polygon(
      data = poly_df,
      aes(x = x, y = y, group = poly_id, fill = type),
      color = "black",
      linewidth = 0.25,
      alpha = 0.4
    ) +
    scale_fill_manual(values = fill_cols) +
    coord_equal() +
    theme_void()
}

plot_processed_tile(tiles_processed[[2]])




library(ggplot2)
library(grid)
library(sf)

plot_cropped_tile <- function(tile_cropped) {
  
  img <- tile_cropped$image
  if (is.null(img)) stop("Tile image missing")
  
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  g <- rasterGrob(
    img,
    width = unit(1, "npc"),
    height = unit(1, "npc"),
    interpolate = FALSE
  )
  
  # ---- polygons (tile-local, bottom-left origin)
  poly_df <- do.call(rbind, lapply(seq_along(tile_cropped$polygons), function(i) {
    p <- tile_cropped$polygons[[i]]
    data.frame(
      x = p$x,
      y = p$y,
      polygon_id = i
    )
  }))
  
  if (is.null(poly_df) || nrow(poly_df) == 0) {
    warning("No polygons in tile")
    return(
      ggplot() +
        annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
        coord_equal(expand = FALSE) +
        theme_void()
    )
  }
  
  # ---- color per polygon
  poly_df$col <- factor(poly_df$polygon_id)
  
  ggplot(poly_df, aes(x = x, y = y, group = polygon_id, fill = col)) +
    annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
    geom_polygon(color = "black", linewidth = 0.25, alpha = 0.4) +
    scale_fill_viridis_d(guide = "none") +
    coord_equal(expand = FALSE) +
    theme_void()
}

plot_cropped_tile(tiles_cropped[[1]])
