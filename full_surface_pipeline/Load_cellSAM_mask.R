library(reticulate)
library(terra)
library(sf)

Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy"),
  python_version = "3.12.4"
)


np <- import("numpy")

# load NumPy array
arr <- np$load("C:/Users/jp19193/Downloads/input_segmentation-20260129T122442Z-3-001/input_segmentation/output_cellSAM/August 21 - 25 samples_1012-1-2_obb_3_mask.npy")



mask_to_instance_polygons_terra <- function(arr, show_progress = TRUE) {
  
  r <- terra::rast(arr)
  labels <- setdiff(unique(as.vector(arr)), 0)
  
  if (length(labels) == 0) {
    return(st_sf(geometry = st_sfc()))
  }
  
  polys <- vector("list", length(labels))
  k <- 1L
  
  if (show_progress) {
    pb <- utils::txtProgressBar(
      min = 0,
      max = length(labels),
      style = 3
    )
    on.exit(close(pb), add = TRUE)
  }
  
  for (i in seq_along(labels)) {
    
    id <- labels[i]
    
    m <- r
    m[m != id] <- NA
    m[m == id] <- 1
    
    p <- terra::as.polygons(
      m,
      dissolve = TRUE,
      na.rm = TRUE
    )
    
    if (!is.null(p) && nrow(p) > 0) {
      sfp <- sf::st_as_sf(p)
      if (nrow(sfp) > 0) {
        sfp$instance_id <- id
        polys[[k]] <- sfp
        k <- k + 1L
      }
    }
    
    if (show_progress) {
      utils::setTxtProgressBar(pb, i)
    }
  }
  
  polys <- polys[seq_len(k - 1L)]
  
  if (length(polys) == 0) {
    return(st_sf(geometry = st_sfc()))
  }
  
  do.call(rbind, polys)
}


polys<-mask_to_instance_polygons_terra(arr)

library(ggplot2)
plot_polys_sf <- function(polys_sf) {
  
  if (nrow(polys_sf) == 0) {
    warning("No polygons to plot")
    return(ggplot() + theme_void())
  }
  
  ggplot(polys_sf) +
    geom_sf(
      aes(fill = factor(instance_id)),
      color = "black",
      linewidth = 0.25,
      alpha = 0.4
    ) +
    coord_sf(expand = FALSE) +
    theme_void() +
    guides(fill = "none")
}


plot_polys_sf(polys)


library(ggplot2)
library(jpeg)
library(grid)

plot_polys_on_image <- function(polys_sf, image_path) {
  
  if (nrow(polys_sf) == 0) {
    warning("No polygons to plot")
    return(ggplot() + theme_void())
  }
  
  # read image
  img <- jpeg::readJPEG(image_path)
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  # raster grob
  g <- rasterGrob(
    img,
    width = unit(1, "npc"),
    height = unit(1, "npc"),
    interpolate = FALSE
  )
  
  ggplot() +
    annotation_custom(
      g,
      xmin = 0, xmax = w,
      ymin = 0, ymax = h
    ) +
    geom_sf(
      data = polys_sf,
      aes(fill = factor(instance_id)),
      color = "black",
      linewidth = 0.25,
      alpha = 0.4,
      inherit.aes = FALSE
    ) +
    coord_sf(
      xlim = c(0, w),
      ylim = c(0, h),
      expand = FALSE
    ) +
    theme_void() +
    guides(fill = "none")
}

plot_polys_on_image(
  polys,
  "C:/Users/jp19193/Downloads/input_segmentation-20260129T122442Z-3-001/input_segmentation/August 21 - 25 samples_1012-1-2_obb_3.jpg"
)
