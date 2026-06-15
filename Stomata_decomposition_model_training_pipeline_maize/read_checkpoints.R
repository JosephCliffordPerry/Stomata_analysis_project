library(sf)
library(dplyr)
library(terra)

# -------------------------------
# SETTINGS
# -------------------------------

rds_dir <- "E:/Stomata_maize/checkpoints/checkpoints4"

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
    
    complex_min_area = 1000,
    complex_max_area = 10000,
    complex_min_circularity = 0.3,
    complex_max_circularity = 1,
    
    companion_min_area = 200,
    companion_max_area = 5000,
    companion_min_circularity = 0.1,
    companion_max_circularity = 1
){
  
  df <- polygons_sf
  
  # metric filters
  df <- df %>%
    
    mutate(
      
      area_fail =
        case_when(
          
          object == "Complex" ~
            area < complex_min_area |
            area > complex_max_area,
          
          object %in% c("Companion1", "Companion2") ~
            area < companion_min_area |
            area > companion_max_area,
          
          TRUE ~ TRUE
        ),
      
      circularity_fail =
        case_when(
          
          object == "Complex" ~
            circularity < complex_min_circularity |
            circularity > complex_max_circularity,
          
          object %in% c("Companion1", "Companion2") ~
            circularity < companion_min_circularity |
            circularity > companion_max_circularity,
          
          TRUE ~ TRUE
        )
    )
  
  filtered <- df %>%
    filter(
      !area_fail &
        !circularity_fail
    )
  
  # image completeness check
  image_check <- filtered %>%
    st_drop_geometry() %>%
    group_by(image) %>%
    summarise(
      has_complex = any(object == "Complex"),
      has_c1 = any(object == "Companion1"),
      has_c2 = any(object == "Companion2"),
      .groups = "drop"
    )
  
  valid_images <- image_check %>%
    filter(
      has_complex &
        has_c1 &
        has_c2
    ) %>%
    pull(image)
  
  filtered %>%
    filter(image %in% valid_images)
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
    
    poly$image <- g$image[1]
    poly$object <- g$object[1]
    poly$name <- paste0(g$image[1], "_", g$object[1])
    
    poly_list[[counter]] <- poly
    counter <- counter + 1
  }
  
  poly_list <- poly_list[!sapply(poly_list, is.null)]
  
  if (length(poly_list) == 0) {
    cat("No polygons generated\n")
    return(NULL)
  }
  
  polygons_sf <- bind_rows(poly_list)
  
  # -------------------------------
  # METRICS
  # -------------------------------
  
  polygons_sf$area <- as.numeric(
    st_area(polygons_sf)
  )
  
  perim <- as.numeric(
    st_length(st_boundary(polygons_sf))
  )
  
  polygons_sf$circularity <-
    4 * pi * polygons_sf$area / (perim^2)
  
  # -------------------------------
  # FILTER
  # -------------------------------
  
  filtered_sf <- filter_polygons(
    polygons_sf,
    
    complex_min_area = 2000,
    complex_max_area = 10000,
    complex_min_circularity = 0.2,
    complex_max_circularity = 0.9,
    
    companion_min_area = 500,
    companion_max_area = 3000,
    companion_min_circularity = 0.2,
    companion_max_circularity = 0.9
  )
  
  cat("Remaining polygons:", nrow(filtered_sf), "\n")
  
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

filtered_list <- vector("list", length(files))

for (i in seq_along(files)) {
  
  filtered_list[[i]] <- tryCatch(
    process_checkpoint(files[i]),
    error = function(e) {
      cat("FAILED:", basename(files[i]), "\n")
      cat(e$message, "\n")
      NULL
    }
  )
}

filtered_list <- filtered_list[
  !sapply(filtered_list, is.null)
]

filtered_sf <- bind_rows(filtered_list)

# -------------------------------
# SAVE
# -------------------------------

save(
  filtered_sf,
  file = "filtered_maize_polys.RDA"
)

cat("\n====================================\n")
cat("FINAL OUTPUT\n")
cat("====================================\n")

print(filtered_sf)
