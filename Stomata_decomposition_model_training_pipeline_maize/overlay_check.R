library(sf)
library(dplyr)
library(ggplot2)
library(png)
library(jpeg)
library(grid)

# =========================================================
# OVERLAY POLYGONS ON IMAGES
# FIXED COORDINATE ALIGNMENT
# =========================================================

overlay_polygons <- function(
    polygons_sf,
    image_dir,
    output_dir = file.path(image_dir, "polygon_overlays6")
){
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # -------------------------------------------------
  # Find images
  # -------------------------------------------------
  
  image_files <- list.files(
    image_dir,
    pattern = "\\.(png|jpg|jpeg)$",
    full.names = TRUE,
    recursive = TRUE,
    ignore.case = TRUE
  )
  
  image_lookup <- data.frame(
    image_name = tools::file_path_sans_ext(basename(image_files)),
    image_path = image_files,
    stringsAsFactors = FALSE
  )
 
  # match against polygons_sf$image
  image_lookup <- image_lookup[
    image_lookup$image_name %in% unlist(polygons_sf$image),
  ]
  
  image_names <- unique(polygons_sf$image)
  
  # -------------------------------------------------
  # Loop
  # -------------------------------------------------
  
  for(img_name in image_names){
    
    cat("\nProcessing:", img_name, "\n")
    
    # -------------------------------------------------
    # Match image
    # -------------------------------------------------
    
    idx <- which(image_lookup$image_name == img_name)
    
    if(length(idx) == 0){
      cat("No image match\n")
      next
    }
    
    img_path <- image_lookup$image_path[idx[1]]
    
    # -------------------------------------------------
    # Read image
    # -------------------------------------------------
    
    if(grepl("\\.png$", img_path, ignore.case = TRUE)){
      img <- png::readPNG(img_path)
    } else {
      img <- jpeg::readJPEG(img_path)
    }
    
    h <- dim(img)[1]
    w <- dim(img)[2]
    
    # -------------------------------------------------
    # Polygon subset
    # -------------------------------------------------
    
    sub_poly <- polygons_sf %>%
      filter(image == img_name)
    
    sub_poly$object <- factor(
      sub_poly$object,
      levels = c("Complex", "Companion1", "Companion2","Companion")
    )
    
    # -------------------------------------------------
    # FLIP Y AXIS SAFELY
    # Handles POLYGON + MULTIPOLYGON
    # -------------------------------------------------
    
    flip_geometry_y <- function(sf_obj, image_height){
      
      geom_new <- lapply(
        st_geometry(sf_obj),
        function(g){
          
          # -----------------------------
          # POLYGON
          # -----------------------------
          
          if(inherits(g, "POLYGON")){
            
            rings <- lapply(
              g,
              function(ring){
                
                ring[,2] <- image_height - ring[,2]
                
                # close polygon
                if(!all(ring[1,] == ring[nrow(ring),])){
                  ring <- rbind(ring, ring[1,])
                }
                
                ring
              }
            )
            
            return(st_polygon(rings))
          }
          
          # -----------------------------
          # MULTIPOLYGON
          # -----------------------------
          
          if(inherits(g, "MULTIPOLYGON")){
            
            polys <- lapply(
              g,
              function(poly){
                
                lapply(
                  poly,
                  function(ring){
                    
                    ring[,2] <- image_height - ring[,2]
                    
                    # close polygon
                    if(!all(ring[1,] == ring[nrow(ring),])){
                      ring <- rbind(ring, ring[1,])
                    }
                    
                    ring
                  }
                )
              }
            )
            
            return(st_multipolygon(polys))
          }
          
          return(g)
        }
      )
      
      st_set_geometry(
        sf_obj,
        st_sfc(geom_new, crs = st_crs(sf_obj))
      )
    }
    
    # -------------------------------------------------
    # Convert image to grob
    # -------------------------------------------------
    
    img_grob <- rasterGrob(
      img,
      width = unit(1, "npc"),
      height = unit(1, "npc"),
      interpolate = FALSE
    )
    
    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    
    p <- ggplot() +
      
      annotation_custom(
        img_grob,
        xmin = 0,
        xmax = w,
        ymin = 0,
        ymax = h
      ) +
      
      geom_sf(
        data = sub_poly,
        aes(color = object),
        fill = NA,
        linewidth = 0.8
      ) +
      
      scale_color_manual(
        values = c(
          "Complex" = "red",
          "Companion1" = "cyan",
          "Companion2" = "yellow",
          "Companion" = "red"
        ),
        drop = FALSE
      ) +
      
      coord_sf(
        xlim = c(0, w),
        ylim = c(0, h),
        expand = FALSE
      ) +
      
      theme_void() +
      ggtitle(img_name)
    
    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    
    out_path <- file.path(
      output_dir,
      paste0(
        gsub("[^A-Za-z0-9_-]", "_", img_name),
        "_overlay.png"
      )
    )
    
    ggsave(
      out_path,
      p,
      limitsize = FALSE
    )
    
    cat("Saved:", out_path, "\n")
  }
}

overlay_polygons(polygons_sf = filtered_sf,image_dir = "E:/Stomata_maize/all_images/all_images/crops")
