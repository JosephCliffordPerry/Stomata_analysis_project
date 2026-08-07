load_polygons <- function(rda_dir){

  library(EBImage)
  
  largest_connected_component <- function(mask){
    
   
    cc <- bwlabel(mask)
    
    if(max(cc) == 1){
      return(mask)}
    
    sizes <- table(cc[cc > 0])
    
    largest <- as.integer(names(sizes)[which.max(sizes)])
    
    cc == largest
  }
  
  files <- list.files(
    rda_dir,
    pattern="\\.RDS$",
    recursive=TRUE,
    full.names=TRUE
  )
  
  polys <- list()
  
  for(i in seq_along(files)){
  
    
    polys[[i]] <- readRDS(files[[i]])
  }


  build_consensus_masks <- function(samples){
    
    for(i in seq_along(samples)){
      
      message(samples[[i]]$image)
      
      stack <- simplify2array(samples[[i]]$masks)
      
      votes <- apply(
        stack,
        c(1,2),
        sum
      )
      
      density <- votes / length(samples[[i]]$masks)
      
      samples[[i]]$density <- density
      
      samples[[i]]$consensus_mask <- largest_connected_component(density >= 0.5)
    }
    
    samples
  }

library(sf)
library(terra)


mask_to_polygon <- function(mask){
  
  # convert logical to numeric
  mask <- matrix(
    as.integer(mask),
    nrow = nrow(mask),
    ncol = ncol(mask)
  )
  
  r <- terra::rast(mask)
  
  p <- terra::as.polygons(
    r,
    dissolve = TRUE,
    values = TRUE
  )
  
  # keep only foreground
  p <- p[p$lyr.1 == 1, ]
  
  if(nrow(p) == 0)
    return(NULL)
  
  sf::st_as_sf(p)
}

build_consensus_polygons <- function(samples){
  
  for(i in seq_along(samples)){
    
    poly <- mask_to_polygon(
      samples[[i]]$consensus_mask
    )
    
    samples[[i]]$consensus_polygon <- poly
    
  }
  
  samples
}

extract_landmarks <- function(samples,
                              n_points=200){
  
  for(i in seq_along(samples)){
    
    poly <- samples[[i]]$consensus_polygon
    
    xy <- sf::st_coordinates(poly)[,1:2]
    
    samples[[i]]$landmarks <-
      resample_polygon(
        xy,
        n_points
      )
  }
  
  samples
}

filter_masks <- function(
    samples,
    min_score = 0.85,
    min_ratio = 0.4,
    max_ratio = 0.7
){
  
  out <- list()
  
  k <- 1
  
  for(i in seq_along(samples)){
    
    sample <- samples[[i]]
    
    message(sample$image)
    
    h <- sample$shape[[1]]
    w <- sample$shape[[2]]
    total_area <- h * w
    
    keep <- logical(length(sample$masks))
    
    for(j in seq_along(sample$masks)){
      
      area_ratio <- sum(sample$masks[[j]]) / total_area
      
      keep[j] <-
        sample$scores[j] >= min_score &&
        area_ratio >= min_ratio &&
        area_ratio <= max_ratio
      
    }
    
    if(!any(keep)){
      message("  removed (no valid masks)")
      next
    }
    
    sample$masks  <- sample$masks[keep]
    sample$scores <- sample$scores[keep]
    
    out[[k]] <- sample
    k <- k + 1
  }
  
  out
}

samples <- polys
#remove obvious detection errors
samples <- filter_masks(
  samples
)

# Mask branch
samples <- build_consensus_masks(samples)



# Shape branch
samples <- build_consensus_polygons(samples)
return(samples)

}


load_polygons_only <- function(rda_dir) {
  
  library(EBImage)
  library(sf)
  library(terra)
  
  filter_masks <- function(
    samples,
    min_score = 0.85,
    min_ratio = 0.4,
    max_ratio = 0.7
  ){
    
    out <- list()
    
    k <- 1
    
    for(i in seq_along(samples)){
      
      sample <- samples[[i]]
      
      message(sample$image)
      
      h <- sample$shape[[1]]
      w <- sample$shape[[2]]
      total_area <- h * w
      
      keep <- logical(length(sample$masks))
      
      for(j in seq_along(sample$masks)){
        
        area_ratio <- sum(sample$masks[[j]]) / total_area
        
        keep[j] <-
          sample$scores[j] >= min_score &&
          area_ratio >= min_ratio &&
          area_ratio <= max_ratio
        
      }
      
      if(!any(keep)){
        message("  removed (no valid masks)")
        next
      }
      
      sample$masks  <- sample$masks[keep]
      sample$scores <- sample$scores[keep]
      
      out[[k]] <- sample
      k <- k + 1
    }
    
    out
  }
  largest_connected_component <- function(mask){
    
    
    cc <- bwlabel(mask)
    
    if(max(cc) == 1){
      return(mask)}
    
    sizes <- table(cc[cc > 0])
    
    largest <- as.integer(names(sizes)[which.max(sizes)])
    
    cc == largest
  }
  
  mask_to_polygon <- function(mask){
    
    # convert logical to numeric
    mask <- matrix(
      as.integer(mask),
      nrow = nrow(mask),
      ncol = ncol(mask)
    )
    
    r <- terra::rast(mask)
    
    p <- terra::as.polygons(
      r,
      dissolve = TRUE,
      values = TRUE
    )
    
    # keep only foreground
    p <- p[p$lyr.1 == 1, ]
    
    if(nrow(p) == 0)
      return(NULL)
    
    sf::st_as_sf(p)
  }
  
  
  files <- list.files(
    rda_dir,
    pattern = "\\.RDS$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  polygons <- vector("list", length(files))
  
  for (i in seq_along(files)) {
    
    sample <- readRDS(files[[i]])
    if (length(sample$masks) == 0)
      next
    filtered <- filter_masks(list(sample))
    
    if (length(filtered) <=1)
      next
    
    sample <- filtered[[1]]
    stack <- simplify2array(sample[["masks"]])
    
    votes <- apply(
      stack,
      c(1,2),
      sum
    )
    
    density <- votes / length(sample[["masks"]])
    
    
    consensus_mask <- largest_connected_component(
      density >= 0.5
    )
    
    poly<- mask_to_polygon(
      consensus_mask
    )
    
    polygons[[i]]<- list(
      image = sample$image_path,
      polygon = poly
    )
  }
  
  Filter(Negate(is.null), polygons)
}
 