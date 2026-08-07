library(sf)
library(png)
library(imager)   
# edge detection
compute_edge_map <- function(image_path){

  img <- imager::load.image(image_path)

  if(dim(img)[4] > 1)
    img <- imager::grayscale(img)

  g <- imager::imgradient(img, "xy")

  mag <- sqrt(g[[1]]^2 + g[[2]]^2)
  mag <- mag / max(mag)

  # cimg -> matrix
  edge_map <- as.array(mag)[,,1,1]

  # imager uses x,y; transpose so edge_map[row, col]
  edge_map <- t(edge_map)

  edge_map
}

library(imager)


compute_normals <- function(shape){
  
  pts <- as.matrix(shape)
  
  n <- nrow(pts)
  
  prev <- rbind(
    pts[n,],
    pts[-n, , drop = FALSE]
  )
  
  nxt <- rbind(
    pts[-1, , drop = FALSE],
    pts[1,]
  )
  
  tangent <- nxt - prev
  
  normals <- cbind(
    -tangent[,2],
    tangent[,1]
  )
  
  length <- sqrt(rowSums(normals^2))
  
  normals / pmax(length, 1e-8)
  
}


sample_edge_profile <- function(edge_map,
                                point,
                                normal,
                                width = 6){
  
  h <- dim(edge_map)[2]
  w <- dim(edge_map)[1]
  
  scores <- numeric(
    length = 2 * width + 1
  )
  
  for(i in seq_along(scores)){
    
    d <- i - width - 1
    
    pos <- point + d * normal
    
    x <- round(pos[1])
    y <- round(pos[2])
    
    if(
      x >= 1 &&
      x <= w &&
      y >= 1 &&
      y <= h
    ){
      
      scores[i] <- edge_map[y,x]
      
    }
    
  }
  
  which.max(scores) - width - 1
  
}


apply_edge_search <- function(shape,
                              edge_map,
                              width = 6){
  
  normals <- compute_normals(shape)
  
  updated <- shape
  
  for(i in seq_len(nrow(shape))){
    
    shift <- sample_edge_profile(
      edge_map,
      shape[i,],
      normals[i,],
      width
    )
    
    updated[i,] <-
      shape[i,] +
      shift * normals[i,]
    
  }
  
  updated
  
}



asm_iteration <- function(shape,
                          model,
                          edge_map,
                          probability_map = NULL,
                          search_width = 6){
  
  ## move landmarks to image edges
  
  candidate <- apply_edge_search(
    shape,
    edge_map,
    search_width
  )
  
  
  ## enforce ASM shape space
  
  weights <- project_shape(
    candidate,
    model
  )
  
  
  fitted <- reconstruct_shape(
    model,
    weights
  )
  
  
  fitted
  
}

