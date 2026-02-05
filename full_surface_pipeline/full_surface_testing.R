library(tidyverse)
library(ggplot2)
library(grid)
library(tiff)
library(raster)
# graph test bit  ---------------------------------------------------------




# # INPUT PATHS
# 
# image_path <- "D:/stomata/20Xstomata_test_8bit/2tile_x002_y002.tif"
# label_path <- "D:/stomata/20Xstomata_test_8bitbatch_outputs/2tile_x002_y002_all_cells.txt"


# HELPER: COMPUTE POLYGON GEOMETRY

recompute_geometry <- function(df) {
  area <- 0.5 * abs(sum(df$x * c(tail(df$y,-1), df$y[1])) - sum(df$y * c(tail(df$x,-1), df$x[1])))
  perimeter <- sum(sqrt(diff(c(df$x, df$x[1]))^2 + diff(c(df$y, df$y[1]))^2))
  circularity <- if(perimeter>0) 4*pi*area / perimeter^2 else 0
  list(area = area, perimeter = perimeter, circularity = circularity)
}


# READ YOLO FILE + FILTER + CONVEX HULL

read_yolo_file <- function(file_path, image_width, image_height,
                           min_area, max_area, min_circ, max_circ) {
  
  lines <- readLines(file_path, warn = FALSE)
  parsed <- strsplit(lines, "\\s+")
  
  polys <- lapply(parsed, function(vals){
    vals <- as.numeric(vals)
    if(length(vals) < 3) return(NULL)
    
    class_id <- vals[1]
    coords <- vals[-1]
    
    df <- data.frame(
      x = coords[seq(1, length(coords), 2)] * image_width,
      y = coords[seq(2, length(coords), 2)] * image_height
    )
    
    # Filter polygons by area and circularity
    geom <- recompute_geometry(df)
    if(geom$area < min_area || geom$area > max_area) return(NULL)
    if(geom$circularity < min_circ || geom$circularity > max_circ) return(NULL)
    
    # Apply convex hull immediately after filtering
    if(nrow(df) >= 3){
      idx <- chull(df$x, df$y)
      df <- df[c(idx, idx[1]), ]  # close hull
    }
    
    list(
      class_id = class_id,
      segmentation = df
    )
  })
  
  Filter(Negate(is.null), polys)
}


# CONVERT LIST TO DATAFRAME

poly_list_to_df <- function(poly_list) {
  df_list <- lapply(seq_along(poly_list), function(i) {
    p <- poly_list[[i]]
    data.frame(
      polygon_id = i,
      class_id = p$class_id,
      x_scaled = p$segmentation$x,
      y_scaled = p$segmentation$y
    )
  })
  do.call(rbind, df_list)
}


# PLOT POLYGONS
# 
# plot_polys <- function(image_path, poly_df) {
#   
#   img <- tryCatch({
#     suppressWarnings(tiff::readTIFF(image_path))
#   }, error = function(e) NULL)
#   
#   if (is.null(img)) stop("Could not read image!")
#   
#   h <- dim(img)[1]
#   w <- dim(img)[2]
#   
#   poly_df <- poly_df %>%
#     mutate(
#       y_scaled = h - y_scaled  # invert Y for image coordinates
#     )
#   
#   g <- rasterGrob(img, width = unit(1,"npc"), height = unit(1,"npc"), interpolate = FALSE)
#   
#   # assign random colors per polygon
#   unique_ids <- unique(poly_df$polygon_id)
#   color_map <- setNames(sample(colors(), length(unique_ids), replace = TRUE), unique_ids)
#   poly_df$col <- color_map[as.character(poly_df$polygon_id)]
#   
#   p <- ggplot(poly_df, aes(x = x_scaled, y = y_scaled, group = polygon_id, fill = col)) +
#     annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
#     geom_polygon(color = "black", linewidth = 0.2, alpha = 0.4) +
#     scale_fill_identity() +
#     coord_equal() +
#     theme_void()
#   
#   print(p)
# }
# 

# RUN ON SINGLE IMAGE


# poly_df <- poly_list_to_df(poly_list)
# plot_polys(image_path, poly_df)



# image section tessalation  ----------------------------------------------
library(sf)
library(raster)
library(dplyr)
library(ggplot2)
library(grid)
library(tiff)


# BINARIZE POLYGONS

polygons_to_raster <- function(poly_list, image_width, image_height) {
  r <- raster(ncol = image_width, nrow = image_height, xmn = 0, xmx = image_width, ymn = 0, ymx = image_height)
  values(r) <- 0
  
  polys_sf <- st_sfc(lapply(poly_list, function(p) {
    coords <- p$segmentation
    if(nrow(coords) < 3) return(NULL)
    if(!all(coords[1,] == coords[nrow(coords),])) coords <- rbind(coords, coords[1,])
    st_polygon(list(as.matrix(coords)))
  }))
  polys_sf <- st_sf(geometry = polys_sf)
  
  r <- rasterize(polys_sf, r, field = 1, background = 0, fun = "max")
  return(r)
}




# TESSELLATE 128x128 TILES-------------------------------------------------


library(raster)
# 
# tessellate_tiles <- function(binary_raster, tile_size = 128, min_coverage = 0.1) {
#   tiles <- list()
#   tile_id <- 1
#   
#   ncols <- ncol(binary_raster)
#   nrows <- nrow(binary_raster)
#   
#   # Generate top-left corner positions for tiles
#   x_starts <- seq(1, ncols - tile_size + 1, by = tile_size)
#   y_starts <- seq(1, nrows - tile_size + 1, by = tile_size)
#   
#   for (x_start in x_starts) {
#     for (y_start in y_starts) {
#       
#       # Define tile extent in raster coordinates
#       e <- extent(binary_raster,
#                   r1 = y_start,
#                   r2 = y_start + tile_size - 1,
#                   c1 = x_start,
#                   c2 = x_start + tile_size - 1)
#       
#       # Crop the raster
#       tile_r <- crop(binary_raster, e)
#       if (is.null(tile_r)) next
#       
#       # Compute coverage (fraction of pixels inside polygons)
#       coverage <- sum(values(tile_r), na.rm = TRUE) / ncell(tile_r)
#       
#       if (coverage >= min_coverage) {
#         tiles[[tile_id]] <- list(
#           x_start = x_start, x_end = x_start + tile_size - 1,
#           y_start = y_start, y_end = y_start + tile_size - 1,
#           coverage = coverage
#         )
#         tile_id <- tile_id + 1
#       }
#     }
#   }
#   
#   return(tiles)
# }

# Box packing -------------------------------------------------------------
build_integral_image <- function(mat) {
  S <- mat
  for (i in 2:nrow(S)) S[i, ] <- S[i, ] + S[i - 1, ]  # cumulative sum along rows
  for (j in 2:ncol(S)) S[, j] <- S[, j] + S[, j - 1]  # cumulative sum along columns
  S
}


pack_boxes <- function(binary_raster,
                       tile_size = 128,
                       min_coverage = 0.1) {
  
  # 🔑 Flip raster so matrix row 1 = bottom
  binary_raster <- raster::flip(binary_raster, direction = "y")
  
  mat <- as.matrix(binary_raster)
  h <- nrow(mat)
  w <- ncol(mat)
  
  # Build integral image
  S <- mat
  S<-build_integral_image(S)
  
  rect_sum <- function(x, y) {
    x2 <- x + tile_size - 1
    y2 <- y + tile_size - 1
    
    if (x2 > w || y2 > h) return(0)
    
    A <- S[y2, x2]
    B <- if (x > 1) S[y2, x - 1] else 0
    C <- if (y > 1) S[y - 1, x2] else 0
    D <- if (x > 1 && y > 1) S[y - 1, x - 1] else 0
    
    A - B - C + D
  }
  
  candidates <- list()
  id <- 1
  
  x_max <- w - tile_size
  y_max <- h - tile_size 
  
  for (y in seq_len(y_max)) {
    for (x in seq_len(x_max)) {
      
      cov <- rect_sum(x, y) / (tile_size^2)
      
      if (cov >= min_coverage) {
        candidates[[id]] <- list(
          x_start = x,
          x_end   = x + tile_size - 1,
          y_start = y,
          y_end   = y + tile_size - 1,
          coverage = cov
        )
        id <- id + 1
      }
    }
  }
  
  if (length(candidates) == 0) return(list())
  
  # Sort by coverage
  candidates <- candidates[
    order(sapply(candidates, function(z) -z$coverage))
  ]
  
  # Greedy non-overlapping selection
  selected <- list()
  
  overlaps <- function(a, b) {
    !(a$x_end < b$x_start || a$x_start > b$x_end ||
        a$y_end < b$y_start || a$y_start > b$y_end)
  }
  
  for (cand in candidates) {
    if (!any(sapply(selected, overlaps, b = cand))) {
      selected[[length(selected) + 1]] <- cand
    }
  }
  
  selected
}



# Tile plotter ------------------------------------------------------------

# CONVERT POLYGON LIST TO DF FOR PLOTTING

poly_list_to_df <- function(poly_list) {
  df_list <- lapply(seq_along(poly_list), function(i) {
    p <- poly_list[[i]]
    data.frame(
      polygon_id = i,
      class_id = p$class_id,
      x_scaled = p$segmentation$x,
      y_scaled = p$segmentation$y
    )
  })
  do.call(rbind, df_list)
}


# PLOT IMAGE + POLYGONS + BEST TILES

plot_tiles_with_polygons <- function(image_path, tiles, poly_list) {
  
  img <- tiff::readTIFF(image_path)
  h <- dim(img)[1]
  w <- dim(img)[2]
  g <- rasterGrob(img, width = unit(1,"npc"), height = unit(1,"npc"), interpolate = FALSE)
  
  # Polygons for plotting
  poly_df <- poly_list_to_df(poly_list) %>%
    mutate(y_scaled = h - y_scaled)  # invert Y for image coords
  
  # Tiles as polygons
  tile_polys <- lapply(tiles, function(t) {
    data.frame(
      x = c(t$x_start, t$x_end, t$x_end, t$x_start, t$x_start),
      y = c(t$y_start, t$y_start, t$y_end, t$y_end, t$y_start)
    )
  })
  tile_df <- bind_rows(tile_polys, .id = "tile_id") %>%
    mutate(y = h - y)  # invert Y for plotting
  
  # Plot
  ggplot() +
    annotation_custom(g, xmin = 0, xmax = w, ymin = 0, ymax = h) +
    geom_polygon(data = tile_df, aes(x = x, y = y, group = tile_id),
                 color = "red", fill = NA, linewidth = 0.8) +
    geom_polygon(data = poly_df, aes(x = x_scaled, y = y_scaled, group = polygon_id),
                 color = "black", fill = "blue", alpha = 0.3) +
    coord_equal() +
    theme_void()
}


# USAGE EXAMPLE
# 
# # poly_list: filtered polygons (after convex hull)
# image_width <- 1004
# image_height <- 1002
# 
#  raster_bin <- polygons_to_raster(poly_list, image_width, image_height)
#  tiles2 <- pack_boxes(raster_bin, tile_size = 256, min_coverage = 0.7)
#  #tiles <- tessellate_tiles(raster_bin, tile_size = 128, min_coverage = 0.7)
#plot<-plot_tiles_with_polygons(image_path, tiles, poly_list)
#  plot2<-plot_tiles_with_polygons(image_path, tiles2, poly_list)
# # library(patchwork)
#   plot2
#plot
 
 