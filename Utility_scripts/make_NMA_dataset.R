library(reticulate)
library(EBImage)

Sys.setenv(RETICULATE_PYTHON = "managed")
np <- import("numpy")

overlay_dir <- "E:/Stomata_maize/all_images/all_images/crops/overlays"
img_dir     <- "E:/Stomata_maize/all_images/all_images/crops"

mask_files <- list.files(overlay_dir, pattern = "_mask.npy$", full.names = TRUE)

# ---- polygon validation ----
validate_polygon <- function(coords) {
  
  if (nrow(coords) < 3) return(NULL)
  
  # remove sequential duplicates
  keep <- c(TRUE, rowSums(abs(diff(coords))) != 0)
  coords <- coords[keep, , drop = FALSE]
  
  if (nrow(coords) < 3) return(NULL)
  
  # ensure polygon closed
  if (!all(coords[1, ] == coords[nrow(coords), ])) {
    coords <- rbind(coords, coords[1, ])
  }
  
  # compute signed area (shoelace)
  x <- coords[,1]
  y <- coords[,2]
  
  area <- 0.5 * sum(x[-1] * y[-length(y)] - x[-length(x)] * y[-1])
  
  # reject degenerate polygons
  if (abs(area) < 1e-6) return(NULL)
  
  # enforce consistent orientation (clockwise)
  if (area > 0) coords <- coords[nrow(coords):1, ]
  
  coords
}

# container for outlines
all_outlines <- list()
idx <- 1

for (mask_file in mask_files) {
  
  cat("Processing", mask_file, "\n")
  
  mask <- py_to_r(np$load(mask_file))
  mask <- mask > 0
  
  mask_img <- EBImage::Image(mask, colormode = "Grayscale")
  
  contours <- EBImage::ocontour(mask_img)
  if (length(contours) == 0) next
  
  contour <- contours[[which.max(sapply(contours, nrow))]]
  
  coords <- validate_polygon(contour)
  if (is.null(coords)) {
    cat("Invalid polygon skipped:", mask_file, "\n")
    next
  }
  
  img_name <- gsub("_mask.npy$", ".png", basename(mask_file))
  img_path <- file.path(img_dir, img_name)
  
  outline_df <- data.frame(
    ImageFile = img_path,
    ObjectID  = 0,
    x         = coords[,1],
    y         = coords[,2]
  )
  
  all_outlines[[idx]] <- outline_df
  idx <- idx + 1
}

combined_df <- do.call(rbind, all_outlines)

out_file <- file.path(overlay_dir, "all_outlines.txt")

write.table(
  combined_df,
  file = out_file,
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

cat("Saved combined file:", out_file, "\n")