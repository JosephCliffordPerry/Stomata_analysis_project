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
    
    companion_min_area = 700,
    companion_max_area = 1500,
    companion_min_circularity = 0.3,
    companion_max_circularity = 0.55
){
  
  polygons_sf %>%
    
    filter(object == "Companion") %>%
    
    mutate(
      
      area_fail =
        area < companion_min_area |
        area > companion_max_area,
      
      circularity_fail =
        circularity < companion_min_circularity |
        circularity > companion_max_circularity
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
  # KEEP ONLY COMPANION CELLS
  # -------------------------------
  
  all_df <- all_df %>%
    filter(object %in% c("Companion1", "Companion2")) %>%
    mutate(object = "Companion")
  
  if (nrow(all_df) == 0) {
    cat("No companion cells\n")
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
    poly$object <- "Companion"
    poly$name <- paste0(g$image[1], "_Companion")
    
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
    
    companion_min_area = 850,
    companion_max_area = 1500,
    companion_min_circularity = 0.4,
    companion_max_circularity = 0.6
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
  file = "filtered_companion_polys.RDA"
)

cat("\n====================================\n")
cat("FINAL OUTPUT\n")
cat("====================================\n")

print(filtered_sf)