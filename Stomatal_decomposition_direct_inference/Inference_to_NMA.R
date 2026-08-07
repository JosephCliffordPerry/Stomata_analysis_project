#Inference_to_NMA

# ---- polygon validation ----
validate_polygon <- function(coords) {
  
  coords <- as.matrix(coords)
  
  if (nrow(coords) < 3) return(NULL)
  
  # remove sequential duplicates
  keep <- c(TRUE, rowSums(abs(diff(coords))) != 0)
  coords <- coords[keep, , drop = FALSE]
  
  if (nrow(coords) < 3) return(NULL)
  
  # ensure polygon closed
  if (!all(coords[1, ] == coords[nrow(coords), ])) {
    coords <- rbind(coords, coords[1, ])
  }
  
  # signed area
  x <- coords[, 1]
  y <- coords[, 2]
  
  area <- 0.5 * sum(
    x[-1] * y[-length(y)] -
      x[-length(x)] * y[-1]
  )
  
  # reject degenerate polygons
  if (abs(area) < 1e-6) return(NULL)
  
  # enforce clockwise orientation
  if (area > 0) {
    coords <- coords[nrow(coords):1, , drop = FALSE]
  }
  
  coords
}

# ---- extract all outlines ----
all_outlines <- vector("list", length(Stomata))

library(sf)
library(magick)
idx <- 1
for (i in seq_along(Stomata)) {
  
  # Extract sf coordinates
  coords <- st_coordinates(
    Stomata[[i]][[2]]
  )[, 1:2]
  
  if (is.null(coords) || nrow(coords) < 3)
    next
  image_path<-paste0("E:/Stomata_maize/all_images/all_images/crops/",Stomata[[i]][[1]],".png")
  # Get image dimensions
  img_info <- image_info(
    image_read(image_path)
  )
  
  img_height <- img_info$height
  
  # Convert Cartesian -> image coordinates
  coords[,2] <- img_height - coords[,2]
  
  coords <- validate_polygon(coords)
  
  if (is.null(coords)) {
    cat("Invalid polygon skipped:", i, "\n")
    next
  }
  
  outline_df <- data.frame(
    Image = image_path,
    Object = i - 1,
    x = coords[,1],
    y = coords[,2]
  )
  
  all_outlines[[idx]] <- outline_df
  idx <- idx + 1
}


# Combine outlines as before
combined_df <- do.call(
  rbind,
  Filter(Negate(is.null), all_outlines)
)

write.table(

  combined_df,

  file = "all_outlines2.txt",

  sep = "\t",

  row.names = FALSE,

  col.names = TRUE,

  quote = FALSE
)

cat(
  "\nSaved",
  length(unique(combined_df$Image)),
  "images to:",
  normalizePath(output_dir),
  "\n"
)