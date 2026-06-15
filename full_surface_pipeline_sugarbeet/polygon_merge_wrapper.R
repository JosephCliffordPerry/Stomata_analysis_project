#polygon merge wrapper

source("full_surface_pipeline/polygon_merge_helpers.R")
source("full_surface_pipeline/axis_aligned_graph_edge.R")

#merge_output <- function(output){
  
  filtered_output <- filter_polygons(
    polygons = output$polygons,
    df = output$metrics,
    area_min = 300,
    circ_min = 0.4
  )
  
  polys <- filtered_output$polygons
 
  
  # --------------------------------
  # Stage 1: axis merge
  # --------------------------------
  axis_edges <- axis_merge_vectorised(polys, distance_tol = 5,
                                      min_shared_length = 5)
  
  stage1_sf <- merge_polygons_convex_hull(
    polygons = polys,
    axis_edges = axis_edges
  )
  
  polys_1<-split_isolated_polygons(
    stage1_sf,
    k = 10,
    overlap_tol = 10,
    area_min = 500,
    axis_length_threshold = 10)
  # --------------------------------
  # Stage 2: bbox/iou merge
  # --------------------------------
  bbox_edges <- build_bbox_iou_edges(
    stage1_sf,
    centroid_dist_tol = 64,
    bbox_iou_threshold = 0.8
  )

    
  result <- merge_polygons_convex_hull(
    polygons = stage1_sf$geometry,
    axis_edges = bbox_edges)
  
    polys_2<-split_isolated_polygons(
      result,
      k = 10,
      overlap_tol = 20,
      axis_length_threshold = 30 )
  
  
  #return(result)
#}

  img <- tiff::readTIFF("E:/Stomata/Sugarbeet_stomata_imaging/sugarbeet_all_mips/V1T1R1_Ab_nd2_-_V1T1R1_Ab_nd2_(series_1)_MIP.tif")
 
  
  plot_sf_overlay(
    img = img,
    sf_polys = polys,
    alpha = 0.4,
    line_col = "yellow"
  )
  plot_sf_overlay(
    img = img,
    sf_polys = polys_1$isolated_polygons,
    alpha = 0.4,
    line_col = "yellow"
  )
  
  plot_sf_overlay(
    img,
    stage1_sf,
    line_col = "yellow"
  )
    plot_sf_overlay(
    img = img,
    sf_polys = polys_2$isolated_polygons,
    alpha = 0.4,
    line_col = "yellow"
  )
    plot_sf_overlay(
      img = img,
      sf_polys = polys_2$remaining_polygons,
      alpha = 0.4,
      line_col = "yellow"
    )
  
  plot_sf_overlay <- function(
    img,
    sf_polys,
    alpha = 0.4,
    fill_col = "red",
    line_col = NA
  ){
    
    suppressPackageStartupMessages({
      library(sf)
      library(ggplot2)
      library(grid)
    })
    
    # ---------------------------------
    # grayscale -> rgb
    # ---------------------------------
    if (length(dim(img)) == 2) {
      
      img <- array(
        rep(img, 3),
        dim = c(dim(img), 3)
      )
    }
    
    H <- dim(img)[1]
    W <- dim(img)[2]
    
    # ---------------------------------
    # raster background
    # ---------------------------------
    grob <- rasterGrob(
      img,
      width = unit(1, "npc"),
      height = unit(1, "npc"),
      interpolate = FALSE
    )
    
    # ---------------------------------
    # convert polygon lists -> sf
    # ---------------------------------
    if (is.list(sf_polys) && !inherits(sf_polys, "sf")) {
      
      sf_polys <- lapply(sf_polys, function(p){
        
        if (inherits(p, "sfg"))
          return(p)
        
        p <- as.matrix(p)
        
        # close polygon
        if (any(p[1,] != p[nrow(p),])) {
          p <- rbind(p, p[1,])
        }
        
        st_polygon(list(p))
      })
      
      sf_polys <- st_sf(
        geometry = st_sfc(sf_polys)
      )
    }
    
    sf_polys <- st_as_sf(sf_polys)
    
    # ---------------------------------
    # flip geometry vertically
    # ---------------------------------
    geom_fixed <- lapply(
      st_geometry(sf_polys),
      function(g){
        
        coords <- st_coordinates(g)[,1:2,drop=FALSE]
        
        coords[,2] <- H - coords[,2]
        
        # close polygon
        if (any(coords[1,] != coords[nrow(coords),])) {
          coords <- rbind(coords, coords[1,])
        }
        
        st_polygon(list(coords))
      }
    )
    
    sf_fixed <- st_sf(
      geometry = st_sfc(geom_fixed)
    )
    
    # ---------------------------------
    # plot
    # ---------------------------------
    ggplot() +
      
      annotation_custom(
        grob,
        xmin = 0,
        xmax = W,
        ymin = 0,
        ymax = H
      ) +
      
      geom_sf(
        data = sf_fixed,
        fill = fill_col,
        colour = line_col,
        alpha = alpha,
        inherit.aes = FALSE
      ) +
      
      coord_sf(
        xlim = c(0, W),
        ylim = c(0, H),
        expand = FALSE
      ) +
      
      theme_void()
  }
  