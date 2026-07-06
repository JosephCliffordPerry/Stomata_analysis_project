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

compute_normals <- function(pts){
  
  pts <- as_landmark_matrix(pts)
  
  n <- nrow(pts)
  
  prev_pts <- rbind(
    pts[n, , drop = FALSE],
    pts[1:(n-1), , drop = FALSE]
  )
  
  next_pts <- rbind(
    pts[2:n, , drop = FALSE],
    pts[1, , drop = FALSE]
  )
  
  tangent <- next_pts - prev_pts
  
  normals <- cbind(-tangent[,2], tangent[,1])
  
  lens <- sqrt(rowSums(normals^2))
  
  normals / pmax(lens, 1e-8)
}

sample_along_normal <- function(edge_map, pt, normal,
                                step = 1, width = 5){
  
  scores <- numeric(2 * width + 1)
  
  h <- nrow(edge_map)
  w <- ncol(edge_map)
  
  k <- 1
  
  for(d in seq(-width, width)){
    
    p <- pt + d * step * normal
    
    x <- round(p[1])
    y <- round(p[2])
    
    if(x < 1 || x > w || y < 1 || y > h){
      scores[k] <- 0
    } else {
      scores[k] <- edge_map[y, x]
    }
    
    k <- k + 1
  }
  
  which.max(scores) - (width + 1)
}


asm_iteration <- function(shape,
                          model,
                          edge_map,
                          step = 1,
                          width = 5){
  
  pts <- as_landmark_matrix(shape)
  if(is.null(pts) || nrow(pts) < 10 || ncol(pts) != 2)
    stop("Invalid shape entering ASM iteration")

  normals <- compute_normals(pts)
  
  updated <- pts
  
  h <- nrow(edge_map)
  w <- ncol(edge_map)
  
  for(i in seq_len(nrow(pts))){
    
    best_shift <- 0
    best_score <- -Inf
    
    for(d in seq(-width, width)){
      
      p <- pts[i,] + d * step * normals[i,]
      
      x <- round(p[1])
      y <- round(p[2])
      
      if(x >= 1 && x <= w && y >= 1 && y <= h){
        
        score <- edge_map[y, x]
        
        if(score > best_score){
          best_score <- score
          best_shift <- d
        }
      }
    }
    
    updated[i,] <- pts[i,] + best_shift * step * normals[i,]
  }
  
  updated
}


project_to_asm <- function(shape, model){
  
  pts <- as_landmark_matrix(shape)
  
  # HARD CHECK
  if(nrow(pts) != model$n_landmarks)
    stop(paste0(
      "Landmark mismatch: got ",
      nrow(pts),
      " expected ",
      model$n_landmarks
    ))
  
  x <- as.vector(t(pts))
  
  if(length(x) != length(model$pca$center))
    stop("Vector dimension mismatch in PCA projection")
  
  x0 <- x - model$pca$center
  
  b <- drop(t(model$modes) %*% x0)
  
  limit <- 3 * sqrt(model$eigenvalues)
  
  b <- pmax(pmin(b, limit), -limit)
  
  reconstruct_shape(model, b)
}


fit_asm <- function(initial_shape,
                    model,
                    edge_map,
                    n_iter = 10,
                    step = 1,
                    width = 5){
  
  shape <- initial_shape
  
  for(i in 1:n_iter){
    
    shape_new <- asm_iteration(
      shape,
      model,
      edge_map,
      step,
      width
    )
    
    shape <- project_to_asm(shape_new, model)
  }
  
  shape
}

as_landmark_matrix <- function(shape){
  
  if(inherits(shape, "sfc"))
    shape <- sf::st_coordinates(shape)[,1:2, drop = FALSE]
  
  if(is.list(shape))
    shape <- do.call(rbind, shape)
  
  shape <- as.matrix(shape)
  
  shape <- shape[complete.cases(shape), , drop = FALSE]
  
  # remove duplicate closure if present
  if(nrow(shape) > 2){
    if(all(shape[1,] == shape[nrow(shape),]))
      shape <- shape[-nrow(shape), , drop = FALSE]
  }
  
  shape
}
aligned_poly <- aligned$landmarks[[13]]

initial <- fit_initial_shape(aligned_poly, model)

initial <- as_landmark_matrix(initial)


if (initial[1] == initial[nrow(initial)])
  initial <- initial[-nrow(initial), , drop = FALSE]
image_path <- paste0(
  "E:/Stomata_maize/all_images/all_images/crops/",
  aligned$image[13],
  ".png"
)

edge_map <- compute_edge_map(image_path)

refined <- fit_asm(
  initial_shape = initial,
  model = model,
  edge_map = edge_map,
  n_iter = 15,
  step = 1,
  width = 6
)
