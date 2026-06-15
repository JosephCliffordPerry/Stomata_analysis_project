library(sf)
library(dplyr)
library(terra)

# -----------------------------------
# SETTINGS
# -----------------------------------

root_dir <- "E:/Stomata_maize/checkpoints/inference_outputs"

# -----------------------------------
# MATRIX -> POLYGON
# -----------------------------------

matrix_to_polygon <- function(mat) {
  
  mat <- as.data.frame(mat, stringsAsFactors = FALSE)
  
  # expected: image, object, row, col
  if (ncol(mat) < 4) return(NULL)
  
  colnames(mat)[1:4] <- c("image", "object", "row", "col")
  
  rows <- as.integer(mat$row)
  cols <- as.integer(mat$col)
  
  if (length(rows) == 0 || all(is.na(rows))) return(NULL)
  
  min_r <- min(rows)
  min_c <- min(cols)
  
  rr <- rows - min_r + 1
  cc <- cols - min_c + 1
  
  rmat <- matrix(
    0,
    nrow = max(rr),
    ncol = max(cc)
  )
  
  rmat[cbind(rr, cc)] <- 1
  
  r <- rast(rmat)
  
  p <- tryCatch(
    as.polygons(r, dissolve = TRUE),
    error = function(e) NULL
  )
  
  if (is.null(p)) return(NULL)
  
  p <- p[p$lyr.1 == 1, ]
  
  if (nrow(p) == 0) return(NULL)
  
  sf_poly <- st_as_sf(p)
  
  # shift back to original coordinates
  geom <- st_geometry(sf_poly)
  geom <- geom + c(min_c - 1, min_r - 1)
  
  sf_poly <- st_sf(geometry = geom)
  
  sf_poly$image <- mat$image[1]
  sf_poly$object <- mat$object[1]
  
  sf_poly
}

# -----------------------------------
# FILTER
# -----------------------------------

filter_polygons <- function(
    sf,
    min_area = 4500,
    max_area = 7000,
    min_circ = 0.4,
    max_circ = 0.8
){
  
  sf$area <- as.numeric(st_area(sf))
  
  perim <- as.numeric(st_length(st_boundary(sf)))
  
  sf$circularity <- 4 * pi * sf$area / (perim^2)
  
  sf %>%
    filter(
      area >= min_area,
      area <= max_area,
      circularity >= min_circ,
      circularity <= max_circ
    )
}

# -----------------------------------
# PROCESS SINGLE RDA FILE
# -----------------------------------

process_file <- function(f){
  
  cat("\nProcessing:", basename(f), "\n")
  
  env <- new.env()
  load(f, envir = env)
  
  if (!exists("out_list", envir = env)) {
    cat("Missing out_list\n")
    return(NULL)
  }
  
  out_list <- env$out_list
  
  poly_list <- vector("list", length(out_list))
  
  for (i in seq_along(out_list)) {
    
    poly_list[[i]] <- tryCatch(
      matrix_to_polygon(out_list[[i]]),
      error = function(e) NULL
    )
  }
  
  poly_list <- poly_list[!sapply(poly_list, is.null)]
  
  if (length(poly_list) == 0) return(NULL)
  
  sf_all <- bind_rows(poly_list)
  
  # keep only Complex masks
  sf_all <- sf_all %>%
    filter(object == "Complex")
  
  sf_all <- filter_polygons(sf_all)
  
  cat("Remaining polygons:", nrow(sf_all), "\n")
  
  sf_all
}

# -----------------------------------
# RECURSIVE FILE COLLECTION
# -----------------------------------

files <- list.files(
  root_dir,
  pattern = "\\.RDA$",
  full.names = TRUE,
  recursive = TRUE
)

cat("Found files:", length(files), "\n")

# -----------------------------------
# RUN ALL TASKS
# -----------------------------------

results <- lapply(files, function(f) {
  tryCatch(
    process_file(f),
    error = function(e) {
      cat("FAILED:", basename(f), "\n")
      NULL
    }
  )
})

results <- results[!sapply(results, is.null)]

filtered_sf <- bind_rows(results)

# -----------------------------------
# SAVE
# -----------------------------------

save(
  filtered_sf,
  file = "filtered_maize_complexes.RDA"
)

cat("\nDONE\n")
print(filtered_sf)