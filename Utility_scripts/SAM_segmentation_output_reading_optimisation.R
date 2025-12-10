
# filter only system ------------------------------------------------------


library(dplyr)
library(purrr)
library(ggplot2)
library(concaveman)
library(png)
library(grid)


# BASIC GEOMETRIC FUNCTIONS


polygon_area <- function(x, y) {
  0.5 * abs(sum(x * c(tail(y, -1), y[1])) -
              sum(y * c(tail(x, -1), x[1])))
}

polygon_perimeter <- function(x, y) {
  sum(sqrt(diff(c(x, x[1]))^2 + diff(c(y, y[1]))^2))
}


# READ YOLO POLYGONS


read_yolo_segmentation_file <- function(file_path,
                                        image_width = NULL,
                                        image_height = NULL,
                                        min_area = 0,
                                        max_area = Inf,
                                        min_circularity = 0,
                                        max_circularity = 1) {
  
  lines <- readLines(file_path)
  parsed <- strsplit(lines, "\\s+")
  
  polygons <- lapply(seq_along(parsed), function(i) {
    vals <- as.numeric(parsed[[i]])
    class_id <- vals[1]
    coords <- vals[-1]
    
    seg_df <- data.frame(
      x = coords[seq(1, length(coords), by = 2)],
      y = coords[seq(2, length(coords), by = 2)]
    )
    
    if (!is.null(image_width) && !is.null(image_height)) {
      seg_df$x <- seg_df$x * image_width
      seg_df$y <- seg_df$y * image_height
    }
    
    area <- polygon_area(seg_df$x, seg_df$y)
    perimeter <- polygon_perimeter(seg_df$x, seg_df$y)
    circularity <- if (perimeter > 0) (4 * pi * area) / (perimeter^2) else 0
    
    list(
      id = i,
      class_id = class_id,
      segmentation = seg_df,
      area = area,
      perimeter = perimeter,
      circularity = circularity
    )
  })
  
  # filtering
  Filter(function(p) {
    p$area >= min_area &&
      p$area <= max_area &&
      p$circularity >= min_circularity &&
      p$circularity <= max_circularity
  }, polygons)
}


# MERGING RULES (CORRECTED ROUND-SHAPE LOGIC)


should_merge <- function(p1, p2,
                         circ_round = 0.8,
                         circ_irregular = 0.4,
                         cutout_ratio = 0.5) {
  
  a1 <- p1$area ; a2 <- p2$area
  c1 <- p1$circularity ; c2 <- p2$circularity
  
  is_round1 <- c1 > circ_round
  is_round2 <- c2 > circ_round
  
  # ------------------------------------------------------------
  # RULE 1: small round shapes NEVER merge (cutout preservation)
  # ------------------------------------------------------------
  if (is_round1 && a1 < a2 * cutout_ratio)
    return(FALSE)
  
  if (is_round2 && a2 < a1 * cutout_ratio)
    return(FALSE)
  
  # ------------------------------------------------------------
  # RULE 2: round-round never merge
  # ------------------------------------------------------------
  if (is_round1 && is_round2)
    return(FALSE)
  
  # ------------------------------------------------------------
  # RULE 3: irregular–irregular always merge if bbox overlaps
  # ------------------------------------------------------------
  if (c1 < circ_irregular && c2 < circ_irregular)
    return(TRUE)
  
  # ------------------------------------------------------------
  # RULE 4: larger round shape can merge smaller irregular
  # ------------------------------------------------------------
  if (is_round1 && !is_round2 && a1 > a2)
    return(TRUE)
  
  if (is_round2 && !is_round1 && a2 > a1)
    return(TRUE)
  
  return(FALSE)
}


# FAST MERGING (NO GEOMETRY OPS)


merge_polygons_fast <- function(polys) {
  
  n <- length(polys)
  used <- rep(FALSE, n)
  out <- vector("list", n)
  idx <- 1
  
  # Precompute bounding boxes
  bboxes <- lapply(polys, function(p) {
    x <- p$segmentation$x
    y <- p$segmentation$y
    list(
      xmin = min(x), xmax = max(x),
      ymin = min(y), ymax = max(y)
    )
  })
  
  message("Merging polygons…")
  next_progress <- 0.1
  
  for (i in seq_len(n)) {
    if (used[i]) next
    
    curr <- polys[[i]]
    bb1 <- bboxes[[i]]
    
    for (j in seq_len(n)) {
      if (i == j || used[j]) next
      
      bb2 <- bboxes[[j]]
      
      # FAST bbox reject
      if (bb2$xmin > bb1$xmax || bb2$xmax < bb1$xmin ||
          bb2$ymin > bb1$ymax || bb2$ymax < bb1$ymin)
        next
      
      # shape-based merge rule
      if (!should_merge(curr, polys[[j]]))
        next
      
      # merge: concatenate points
      curr$segmentation <- rbind(curr$segmentation,
                                 polys[[j]]$segmentation)
      
      # recompute stats
      x <- curr$segmentation$x
      y <- curr$segmentation$y
      curr$area <- polygon_area(x,y)
      curr$perimeter <- polygon_perimeter(x,y)
      curr$circularity <- (4*pi*curr$area)/(curr$perimeter^2)
      
      # update bbox
      bb1 <- list(
        xmin = min(x), xmax = max(x),
        ymin = min(y), ymax = max(y)
      )
      
      used[j] <- TRUE
    }
    
    out[[idx]] <- curr
    idx <- idx + 1
    
    # progress updates
    p <- i/n
    if (p >= next_progress) {
      message(sprintf("Progress: %d%% complete",
                      round(100*next_progress)))
      next_progress <- next_progress + 0.1
    }
  }
  
  out <- out[!sapply(out, is.null)]
  return(out)
}


# APPLY CONCAVE HULL


apply_concave_hull <- function(polys, concavity = 2) {
  lapply(polys, function(p) {
    df <- p$segmentation
    if (nrow(df) < 4) return(p)
    
    pts <- as.matrix(df[, c("x", "y")])
    
    hull <- concaveman(pts, concavity = concavity)
    hull <- as.data.frame(hull)
    colnames(hull) <- c("x", "y")
    
    p$segmentation <- hull
    return(p)
  })
}




# LOAD YOLO POLYGONS


yolo_polygons <- read_yolo_segmentation_file(
  "grid_of_4_cells_all.txt",
  image_width = 1002,
  image_height = 1004,
  min_area     = 300,
  max_area     = 10000,
  min_circularity = 0.20,
  max_circularity = 1.00
)


# MERGE


merged <- merge_polygons_fast(yolo_polygons)


# REMOVE TINY POST-MERGE ARTIFACTS


merged <- Filter(function(p) p$area > 300 & p$area < 10000, merged)


# CONCAVE HULL SHAPE CLEANUP


merged <- apply_concave_hull(merged, concavity = 2)
merged[]

# CONVERT TO DATAFRAME FOR PLOTTING


yolo_df <- map_df(merged, function(poly) {
  seg <- poly$segmentation
  seg$polygon_id <- poly$id
  seg$class_id <- poly$class_id
  seg$circularity <- poly$circularity
  seg
})


# SCALE TO IMAGE COORDS


img <- readPNG("D:/stomata/sam_overlays/overlay_0715.png")
img_width <- dim(img)[2]
img_height <- dim(img)[1]

yolo_df <- yolo_df %>%
  mutate(
    x_scaled = x / 1002 * img_width,
    y_scaled = img_height - (y / 1004 * img_height)
  )


# PLOT FINAL RESULT


g <- rasterGrob(img, width = unit(1,"npc"), height = unit(1,"npc"))

ggplot(yolo_df, aes(x = x_scaled, y = y_scaled,
                    group = polygon_id,
                    fill = factor(class_id))) +
  annotation_custom(g, xmin = 0, xmax = img_width,
                    ymin = 0, ymax = img_height) +
  geom_polygon(alpha = 0.4, color = "black", linewidth = 0.2) +
  coord_equal() +
  theme_minimal() +
  labs(title = "Filtered SAM Segmentations (Merged + Concave Hull)")



library(dplyr)
library(purrr)
library(ggplot2)
library(png)
library(grid)
library(raster)
library(sp)
library(patchwork)

# ---------------------- GEOMETRIC FUNCTIONS ----------------------

polygon_area <- function(x, y) {
  0.5 * abs(sum(x * c(tail(y, -1), y[1])) -
              sum(y * c(tail(x, -1), x[1])))
}

polygon_perimeter <- function(x, y) {
  sum(sqrt(diff(c(x, x[1]))^2 + diff(c(y, y[1]))^2))
}

# ---------------------- READ YOLO POLYGONS ----------------------

read_yolo_segmentation_file <- function(file_path,
                                        image_width = NULL,
                                        image_height = NULL,
                                        min_area = 0,
                                        max_area = Inf,
                                        min_circularity = 0,
                                        max_circularity = 1) {
  
  lines <- readLines(file_path)
  parsed <- strsplit(lines, "\\s+")
  
  polygons <- lapply(seq_along(parsed), function(i) {
    vals <- as.numeric(parsed[[i]])
    class_id <- vals[1]
    coords <- vals[-1]
    
    seg_df <- data.frame(
      x = coords[seq(1, length(coords), by = 2)],
      y = coords[seq(2, length(coords), by = 2)]
    )
    
    if (!is.null(image_width) && !is.null(image_height)) {
      seg_df$x <- seg_df$x * image_width
      seg_df$y <- seg_df$y * image_height
    }
    
    area <- polygon_area(seg_df$x, seg_df$y)
    perimeter <- polygon_perimeter(seg_df$x, seg_df$y)
    circularity <- if (perimeter > 0) (4 * pi * area) / (perimeter^2) else 0
    
    list(
      id = i,
      class_id = class_id,
      segmentation = seg_df,
      area = area,
      perimeter = perimeter,
      circularity = circularity
    )
  })
  
  Filter(function(p) {
    p$area >= min_area &&
      p$area <= max_area &&
      p$circularity >= min_circularity &&
      p$circularity <= max_circularity
  }, polygons)
}

# ---------------------- CONVEX HULL ----------------------

apply_convex_hull <- function(polys) {
  lapply(polys, function(p) {
    df <- p$segmentation
    if (nrow(df) < 3) return(NULL)
    hull_idx <- chull(df$x, df$y)
    p$segmentation <- df[hull_idx, , drop = FALSE]
    return(p)
  })
}

filter_valid_polys <- function(polys) {
  Filter(function(p) {
    !is.null(p) &&
      nrow(p$segmentation) >= 3 &&
      all(!is.na(p$segmentation$x)) &&
      all(!is.na(p$segmentation$y)) &&
      polygon_area(p$segmentation$x, p$segmentation$y) > 0
  }, polys)
}

# ---------------------- MERGE WITHIN LIST (IoU) ----------------------

merge_within_list <- function(polys, iou_threshold = 0.1) {
  if (length(polys) <= 1) return(polys)
  
  bbox <- function(df) list(
    xmin = min(df$x), xmax = max(df$x),
    ymin = min(df$y), ymax = max(df$y)
  )
  
  polygon_iou <- function(p1, p2) {
    all_x <- c(p1$segmentation$x, p2$segmentation$x)
    all_y <- c(p1$segmentation$y, p2$segmentation$y)
    res <- max(diff(range(all_x))/100, diff(range(all_y))/100)
    x_seq <- seq(min(all_x), max(all_x), by = res)
    y_seq <- seq(min(all_y), max(all_y), by = res)
    grid_pts <- expand.grid(x = x_seq, y = y_seq)
    inside_poly <- function(poly, pts) {
      sp::point.in.polygon(pts$x, pts$y, poly$segmentation$x, poly$segmentation$y) > 0
    }
    inside1 <- inside_poly(p1, grid_pts)
    inside2 <- inside_poly(p2, grid_pts)
    inter <- sum(inside1 & inside2)
    union <- sum(inside1 | inside2)
    if (union == 0) return(0)
    inter / union
  }
  
  merge_two <- function(p1, p2) {
    seg <- rbind(p1$segmentation, p2$segmentation)
    area <- polygon_area(seg$x, seg$y)
    perim <- polygon_perimeter(seg$x, seg$y)
    circ <- if(perim>0) 4*pi*area/perim^2 else 0
    list(id = p1$id, class_id = p1$class_id,
         segmentation = seg, area = area, perimeter = perim, circularity = circ)
  }
  
  merged <- list()
  used <- rep(FALSE, length(polys))
  
  for (i in seq_along(polys)) {
    if (used[i]) next
    curr <- polys[[i]]
    for (j in seq_along(polys)) {
      if (i==j || used[j]) next
      bb1 <- bbox(curr$segmentation)
      bb2 <- bbox(polys[[j]]$segmentation)
      if (bb2$xmin > bb1$xmax || bb2$xmax < bb1$xmin ||
          bb2$ymin > bb1$ymax || bb2$ymax < bb1$ymin)
        next
      iou <- polygon_iou(curr, polys[[j]])
      if (iou >= iou_threshold) {
        curr <- merge_two(curr, polys[[j]])
        used[j] <- TRUE
      }
    }
    merged <- append(merged, list(curr))
    used[i] <- TRUE
  }
  merged
}

# ---------------------- STEPWISE MERGE ----------------------

stepwise_merge <- function(polys, circ_round = 0.7, max_small_area = 500) {
  round_shapes <- list()
  large_shapes <- list()
  small_shapes <- list()
  
  for (p in polys) {
    if (p$circularity > circ_round) round_shapes <- append(round_shapes, list(p))
    else if (p$area > max_small_area) large_shapes <- append(large_shapes, list(p))
    else small_shapes <- append(small_shapes, list(p))
  }
  
  list(round_shapes = merge_within_list(round_shapes, 0.1),
       large_shapes = merge_within_list(large_shapes, 0.3),
       small_shapes = small_shapes)
}

# ---------------------- PIXEL-WISE DEDUPLICATION ----------------------

polygon_to_sp <- function(poly) {
  Polygons(list(Polygon(poly$segmentation)), ID = as.character(poly$id))
}

raster_deduplicate <- function(polys, img_width, img_height, priority = "area") {
  r <- raster(ncol = img_width, nrow = img_height, xmn = 0, xmx = img_width, ymn = 0, ymx = img_height)
  values(r) <- NA
  
  if (priority == "area") {
    areas <- sapply(polys, function(p) p$area)
    polys <- polys[order(-areas)] # largest first
  }
  
  for (i in seq_along(polys)) {
    sp_poly <- SpatialPolygons(list(polygon_to_sp(polys[[i]])))
    mask <- rasterize(sp_poly, r, field = i, background = NA)
    r[is.na(values(r))] <- mask[is.na(values(r))]
  }
  return(r)
}

# ---------------------- CONVERT MASK TO POLYGONS ----------------------

raster_to_polygons <- function(r) {
  sp_polys <- rasterToPolygons(r, dissolve = TRUE)
  lapply(seq_along(sp_polys), function(i) {
    coords <- sp_polys@polygons[[i]]@Polygons[[1]]@coords
    data.frame(x = coords[,1], y = coords[,2], polygon_id = i)
  })
}

# ---------------------- LOAD YOLO POLYGONS ----------------------

yolo_polygons <- read_yolo_segmentation_file(
  "grid_of_4_cells_all.txt",
  image_width = 1002,
  image_height = 1004,
  min_area = 300,
  max_area = 10000,
  min_circularity = 0.2,
  max_circularity = 1.0
)

# ---------------------- MERGE ----------------------

merged_lists <- stepwise_merge(yolo_polygons)

# ---------------------- CONVEX HULL ----------------------

round_shapes <- filter_valid_polys(apply_convex_hull(merged_lists$round_shapes))
large_shapes <- filter_valid_polys(apply_convex_hull(merged_lists$large_shapes))

# ---------------------- PIXEL-WISE DEDUPLICATION ----------------------

img <- readPNG("D:/stomata/sam_overlays/overlay_0715.png")
img_width <- dim(img)[2]
img_height <- dim(img)[1]

mask_large <- raster_deduplicate(large_shapes, img_width, img_height, priority = "area")
mask_round <- raster_deduplicate(round_shapes, img_width, img_height, priority = "area")

# Convert raster back to polygons for plotting
df_large <- bind_rows(raster_to_polygons(mask_large))
df_round <- bind_rows(raster_to_polygons(mask_round))
df_all <- bind_rows(df_large, df_round)

# ---------------------- SCALE ----------------------

scale_coords <- function(df) {
  df %>% mutate(
    x_scaled = x / 1002 * img_width,
    y_scaled = img_height - (y / 1004 * img_height)
  )
}

df_large <- scale_coords(df_large)
df_round <- scale_coords(df_round)
df_all <- scale_coords(df_all)

# ---------------------- MULTIPANEL PLOT ----------------------

g <- rasterGrob(img, width = unit(1,"npc"), height = unit(1,"npc"))

plot_polys <- function(df, title_text) {
  ggplot(df, aes(x = x_scaled, y = y_scaled, group = polygon_id)) +
    annotation_custom(g, xmin = 0, xmax = img_width, ymin = 0, ymax = img_height) +
    geom_polygon(alpha = 0.4, fill = "blue", color = "black", linewidth = 0.2) +
    coord_equal() + theme_minimal() + labs(title = title_text)
}

p_main <- plot_polys(df_all, "All Shapes")
p_large <- plot_polys(df_large, "Large Shapes")
p_round <- plot_polys(df_round, "Round Shapes")

(p_main | p_large) /
  (p_round | ggplot() + theme_void())

p_large
p_round
p_main

