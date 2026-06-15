# =========================================================
# Batch wrapper for YOLO + stitching pipeline
# - Runs pipeline for all images in a folder
# - Saves graph outputs per image
# - Returns list of stitched polygons for every image
# =========================================================

library(fs)
library(ggplot2)
library(sf)
#This has the issue of relying on a unoptimised polygon stitching function that makes it too slow 
#sourece scripts 
source("D:/Stomatal_analysis_project/full_surface_pipeline/Tile_based_yolo_inference.R")

source("D:/Stomatal_analysis_project/full_surface_pipeline/over_tile_merge.R")
# ---------------------------------------------------------
# Helper: plot stitched polygons over image
# (assumes load_image() already exists in your pipeline)
# ---------------------------------------------------------
plot_stitched_overlay <- function(img, stitched_sf, alpha = 0.4, title = NULL){
  
  H <- dim(img)[1]
  W <- dim(img)[2]
  
  # -------------------------------------------------
  # Normalise image to valid raster format
  # -------------------------------------------------
  if(length(dim(img)) == 2){
    # pure grayscale
    img_rgb <- array(rep(img,3), dim = c(H,W,3))
    
  } else {
    
    C <- dim(img)[3]
    
    if(C == 1){
      img_rgb <- array(rep(img[,,1],3), dim = c(H,W,3))
      
    } else if(C == 3){
      img_rgb <- img
      
    } else if(C == 4){
      img_rgb <- img[,,1:3]   # drop alpha safely
      
    } else {
      # microscopy multi-channel → take first channel
      img_rgb <- array(rep(img[,,1],3), dim = c(H,W,3))
    }
  }
  
  # -------------------------------------------------
  # Ensure numeric range 0-1
  # -------------------------------------------------
  if(max(img_rgb, na.rm=TRUE) > 1)
    img_rgb <- img_rgb / max(img_rgb, na.rm=TRUE)
  
  grob <- grid::rasterGrob(
    img_rgb,
    width  = grid::unit(1,"npc"),
    height = grid::unit(1,"npc")
  )
  
  p <- ggplot() +
    annotation_custom(grob, xmin = 0, xmax = W, ymin = 0, ymax = H)
  
  if(!is.null(stitched_sf) && nrow(stitched_sf) > 0){
    p <- p +
      geom_sf(
        data = stitched_sf,
        fill = "red",
        colour = "yellow",
        alpha = alpha,
        inherit.aes = FALSE
      )
  }
  
  p +
    coord_sf(expand = FALSE) +
    scale_y_reverse() +
    theme_void() +
    ggtitle(title)
}

# ---------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------
batch_yolo_stitch <- function(
    image_dir,
    model_path,
    out_plot_dir = "stitched_plots",
    tile_size  = 128,
    overlap    = 96,
    iou_thresh = 0.5,
    min_area   = 500,
    max_area   = 3000,
    min_circ   = 0.4,
    max_circ   = 1.0,
    alpha      = 0.4,
    shrink_percent = 0.40,
    dilate_percent = 0.20,
    stitch_min_area = 300,
    stitch_max_area = 7500,
    remove_edge = TRUE,
    recursive = FALSE
){
  
  dir_create(out_plot_dir)
  
  image_files <- dir_ls(
    image_dir,
    recurse = recursive,
    regexp = "\\.(tif|tiff|png|jpg|jpeg)$",
    type = "file"
  )
  
  all_polygons <- list()
  
  for(i in seq_along(image_files)){
    
    img_path <- image_files[i]
    message("Processing: ", img_path)
    
    params <- list(
      image_path = img_path,
      model_path = model_path,
      tile_size  = tile_size,
      overlap    = overlap,
      iou_thresh = iou_thresh,
      min_area   = min_area,
      max_area   = max_area,
      min_circ   = min_circ,
      max_circ   = max_circ,
      alpha      = alpha
    )
 
    # ---------------------------------------------
    # Run YOLO pipeline
    # ---------------------------------------------
    img <- load_image(params$image_path)
    output <- run_yolo_pipeline(params)
    
    polys   <- output[[1]]
    img_dim <- dim(img)
    
    # ---------------------------------------------
    # Stitch polygons
    # ---------------------------------------------
    stitched <- tryCatch({
      curve_stitch_sf(
        polys,
        img_dim,
        shrink_percent = shrink_percent,
        dilate_percent = dilate_percent,
        min_area = stitch_min_area,
        max_area = stitch_max_area,
        remove_edge = remove_edge
      )
    }, error = function(e){
      message("Stitch failed: ", basename(img_path))
      return(NULL)
    })
    
    # ---------------------------------------------
    # Save plot
    # ---------------------------------------------
    if(!is.null(stitched)){
      
      plot_obj <- plot_stitched_overlay(
        img,
        stitched$stitched_sf,
        alpha = alpha
      )
      
      out_file <- file.path(
        out_plot_dir,
        paste0(path_ext_remove(basename(img_path)), "_stitched.png")
      )
      
      ggsave(
        out_file,
        plot_obj,
        width = 8,
        height = 8,
        dpi = 300 
      )
    }
    
    # ---------------------------------------------
    # Store polygons
    # ---------------------------------------------
    all_polygons[[basename(img_path)]] <- stitched
  }
  
  return(all_polygons)
}

# =========================================================
# Example usage
# =========================================================
# polys_all <- batch_yolo_stitch(
#   image_dir = "E:/Stomata/Sugarbeet_stomata_imaging/mips2",
#   model_path = "E:/Stomata/Sugarbeet_stomata_imaging/beetmip_model2/beetmip_model2.pt"
# )
