library(sf)
library(dplyr)
library(purrr)
library(tidyr)

rda_dir<-"E:/Stomata_maize/all_images/consensus_and_inference_rda2"

load_polygons <- function(rda_dir){
  
  files <- list.files(
    rda_dir,
    pattern="\\.RDA$",
    recursive=TRUE,
    full.names=TRUE
  )
  
  polys <- list()
  
  for(i in seq_along(files)){
    
    load(files[i])
    
    image_sf$file <- basename(files[i])
    
    polys[[i]] <- image_sf
  }
  
  dplyr::bind_rows(polys)
  
}

align_polygons <- function(polygons){
  
  out <- list()
  
  files <- unique(polygons$file)
  
  for(f in files){
    
    message(f)
    
    img <- polygons |>
      dplyr::filter(file==f)
    
    consensus <- img |>
      dplyr::filter(object=="Consensus") |>
      dplyr::slice(1)
    
    xy <- sf::st_coordinates(consensus)[,1:2, drop=FALSE]
    
    xy <- xy[complete.cases(xy), , drop=FALSE]
    
    if(nrow(xy) < 3){
      message("Skipping (degenerate polygon): ", f)
      next
    }
    
    centre <- colMeans(xy)
    
    X <- sweep(xy, 2, centre)
    
    if(nrow(X) < 2){
      message("Skipping (not enough variance): ", f)
      next
    }
    
    pca <- prcomp(X)
    
    theta <- atan2(
      pca$rotation[2,1],
      pca$rotation[1,1]
    )
    
    aligned <- lapply(img$geometry,function(g){
      
      crd <- sf::st_coordinates(g)
      
      pts <- sweep(crd[,1:2],2,centre)
      
      R <- matrix(c(
        cos(-theta),-sin(-theta),
        sin(-theta), cos(-theta)
      ),2)
      
      pts <- pts %*% R
      
      ## reflect if necessary
      if(mean(pts[,1])>0)
        pts[,1] <- -pts[,1]
      
      pts <- as.matrix(pts)
      
      # Force exact closure
      pts[nrow(pts), ] <- pts[1, ]
      
      sf::st_sfc(
        sf::st_polygon(list(pts))
      )
      
    })
    
    img$geometry <- do.call(c,aligned)
    

    
    out[[f]] <- img
    
  }
  
  dplyr::bind_rows(out)
  
}

build_probability_map <- function(aligned, n = 200) {
  
  library(sf)
  
  rasterise_shape <- function(xy, n = 200) {
    
    # xy must already be:
    # - centred
    # - rotated
    # - scaled
    
    grid_x <- seq(-1.5, 1.5, length.out = n)
    grid_y <- seq(-1.5, 1.5, length.out = n)
    
    inside <- sp::point.in.polygon(
      rep(grid_x, each = n),
      rep(grid_y, times = n),
      xy[,1],
      xy[,2]
    )
    
    matrix(inside, n, n)
  }
    
  align_to_shape_space <- function(g) {
    
    xy <- sf::st_coordinates(g)[,1:2]
    
    # centre
    xy <- sweep(xy, 2, colMeans(xy))
    
    # rotate via PCA
    pca <- prcomp(xy)
    theta <- atan2(pca$rotation[2,1], pca$rotation[1,1])
    
    R <- matrix(c(
      cos(-theta), -sin(-theta),
      sin(-theta),  cos(-theta)
    ), nrow = 2, byrow = TRUE)
    
    xy <- xy %*% R
    
    # reflect to enforce orientation consistency
    if(mean(xy[,1]) > 0)
      xy[,1] <- -xy[,1]
    
    # scale (shape-only)
    s <- sqrt(mean(rowSums(xy^2)))
    if(s > 0) xy <- xy / s
    
    xy[nrow(xy), ] <- xy[1, ]
    
    xy
  }
  
  build_shape_matrices <- function(aligned_sf, n = 200) {
    
    geoms <- sf::st_geometry(aligned_sf)
    
    mats <- vector("list", length(geoms))
    
    for(i in seq_along(geoms)) {
      
      xy <- align_to_shape_space(geoms[[i]])
      
      mats[[i]] <- rasterise_shape(xy, n)
    }
    
    mats
  }
  build_probability_map <- function(aligned_sf, n = 200) {
    
    mats <- build_shape_matrices(aligned_sf, n)
    
    holder <- Reduce("+", mats)
    
    holder / length(mats)
  }
  }





polys<-load_polygons(rda_dir)

aligned<-align_polygons(polys)

probability_map<-build_probability_map(aligned)
image(probability_map)
