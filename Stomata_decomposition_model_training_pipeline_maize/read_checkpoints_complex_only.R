library(sf)
library(dplyr)
library(terra)

# -------------------------------
# SETTINGS
# -------------------------------

rds_dir <- "E:/Stomata_maize/checkpoints/checkpoints5"

image_dir <- "E:/Stomata_maize/all_images/all_images/crops"

border_buffer <- 10

# -------------------------------
# LOAD CHECKPOINT
# -------------------------------

load_checkpoint <- function(f) {
  
  x <- readRDS(f)
  
  # checkpoint files
  if (is.list(x) && "data" %in% names(x)) {
    return(bind_rows(x$data))
  }
  
  # final result files
  if (is.data.frame(x)) {
    return(x)
  }
  
  stop(paste("Unknown file structure:", f))
}

# -------------------------------
# PIXELS -> POLYGON
# -------------------------------

pixels_to_polygon <- function(df_sub) {
  
  rows <- df_sub$row
  cols <- df_sub$col
  
  min_r <- min(rows)
  min_c <- min(cols)
  
  rr <- rows - min_r + 1
  cc <- cols - min_c + 1
  
  mat <- matrix(
    0,
    nrow = max(rr),
    ncol = max(cc)
  )
  
  mat[cbind(rr, cc)] <- 1
  
  r <- rast(mat)
  
  p <- as.polygons(r, dissolve = TRUE)
  p <- p[p$lyr.1 == 1, ]
  
  if (nrow(p) == 0) return(NULL)
  
  p_sf <- st_as_sf(p)
  
  # shift back into original coordinates
  geom <- st_geometry(p_sf)
  geom <- geom + c(min_c - 1, min_r - 1)
  
  st_sf(geometry = geom)
}

# -------------------------------
# FILTER FUNCTION
# -------------------------------

filter_polygons <- function(
    polygons_sf,
    
    complex_min_area = 5000,
    complex_max_area = 6000,
    complex_min_circularity = 0.50,
    complex_max_circularity = 0.80
){
  
  polygons_sf %>%
    
    mutate(
      
      area_fail =
        area < complex_min_area |
        area > complex_max_area,
      
      circularity_fail =
        circularity < complex_min_circularity |
        circularity > complex_max_circularity
    ) %>%
    
    filter(
      !area_fail &
        !circularity_fail
    )
}

# -------------------------------
# PROCESS ONE CHECKPOINT
# -------------------------------

process_checkpoint <- function(f) {
  
  cat("\n====================================\n")
  cat("PROCESSING:", basename(f), "\n")
  cat("====================================\n")
  
  all_df <- load_checkpoint(f)
  
  if (nrow(all_df) == 0) {
    cat("Empty checkpoint\n")
    return(NULL)
  }
  
  # -------------------------------
  # KEEP ONLY COMPLEX
  # -------------------------------
  
  all_df <- all_df %>%
    filter(object == "Complex")
  
  if (nrow(all_df) == 0) {
    cat("No Complex objects\n")
    return(NULL)
  }
  
  # force character columns
  all_df$image <- as.character(all_df$image)
  all_df$object <- as.character(all_df$object)
  
  groups <- all_df %>%
    group_by(image, object) %>%
    group_split()
  
  poly_list <- vector("list", length(groups))
  
  counter <- 1
  
  for (g in groups) {
    
    poly <- tryCatch(
      pixels_to_polygon(g),
      error = function(e) NULL
    )
    
    if (is.null(poly)) next
    
    poly$image <- as.character(g$image[1])
    poly$object <- as.character(g$object[1])
    
    poly$name <- paste0(
      g$image[1],
      "_",
      g$object[1]
    )
    
    poly_list[[counter]] <- poly
    
    counter <- counter + 1
  }
  
  poly_list <- poly_list[
    !sapply(poly_list, is.null)
  ]
  
  if (length(poly_list) == 0) {
    cat("No polygons generated\n")
    return(NULL)
  }
  
  polygons_sf <- bind_rows(poly_list)
  
  # -------------------------------
  # IMAGE DIMENSIONS
  # -------------------------------
  
  image_name <- unique(
    as.character(polygons_sf$image)
  )[1]
  
  # -------------------------------
  # FIND IMAGE FILE
  # -------------------------------
  
  image_name <- trimws(image_name)
  
  image_matches <- list.files(
    image_dir,
    pattern = paste0(
      "^",
      gsub(
        "([.|()\\^{}+$*?]|\\[|\\])",
        "\\\\\\1",
        image_name
      ),
      "(\\.png)?$"
    ),
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (length(image_matches) == 0) {
    
    cat(
      "Image not found:",
      image_name,
      "\n"
    )
    
    return(NULL)
  }
  
  image_path <- image_matches[1]
  
  img_rast <- rast(image_path)
  
  img_width <- ncol(img_rast)
  img_height <- nrow(img_rast)
  
  polygons_sf$image_width <- img_width
  polygons_sf$image_height <- img_height
  
  # -------------------------------
  # METRICS
  # -------------------------------
  
  polygons_sf$area <- as.numeric(
    st_area(polygons_sf)
  )
  
  perim <- as.numeric(
    st_length(
      st_boundary(polygons_sf)
    )
  )
  
  polygons_sf$circularity <-
    4 * pi * polygons_sf$area / (perim^2)
  
  # -------------------------------
  # BOUNDING BOX FILTER
  # -------------------------------
  
  bbox_df <- do.call(
    rbind,
    lapply(
      st_geometry(polygons_sf),
      st_bbox
    )
  )
  
  bbox_df <- as.data.frame(bbox_df)
  
  polygons_sf <- bind_cols(
    polygons_sf,
    bbox_df
  )
  
  polygons_sf <- polygons_sf %>%
    
    filter(
      xmin > border_buffer,
      ymin > border_buffer,
      xmax < (image_width - border_buffer),
      ymax < (image_height - border_buffer)
    )
  
  if (nrow(polygons_sf) == 0) {
    
    cat(
      "All polygons removed by border filter\n"
    )
    
    return(NULL)
  }
  
  # -------------------------------
  # GEOMETRIC FILTER
  # -------------------------------
  
  filtered_sf <- filter_polygons(
    polygons_sf,
    
    complex_min_area = 4500,
    complex_max_area = 7000,
    complex_min_circularity = 0.5,
    complex_max_circularity = 0.8
  )
  
  cat(
    "Remaining polygons:",
    nrow(filtered_sf),
    "\n"
  )
  
  return(filtered_sf)
}

# -------------------------------
# RUN ALL CHECKPOINTS
# -------------------------------

files <- list.files(
  rds_dir,
  pattern = "\\.rds$",
  full.names = TRUE
)

filtered_list <- vector(
  "list",
  length(files)
)

for (i in seq_along(files)) {
  
  filtered_list[[i]] <- tryCatch(
    
    process_checkpoint(files[i]),
    
    error = function(e) {
      
      cat(
        "FAILED:",
        basename(files[i]),
        "\n"
      )
      
      cat(
        "ERROR:",
        conditionMessage(e),
        "\n"
      )
      
      NULL
    }
  )
}

filtered_list <- filtered_list[
  !sapply(filtered_list, is.null)
]

if (length(filtered_list) == 0) {
  
  stop("No valid outputs generated")
}

filtered_sf <- bind_rows(filtered_list)

# -------------------------------
# SAVE
# -------------------------------

save(
  filtered_sf,
  file = "filtered_maize_complexes.RDA"
)

cat("\n====================================\n")
cat("FINAL OUTPUT\n")
cat("====================================\n")

print(filtered_sf)