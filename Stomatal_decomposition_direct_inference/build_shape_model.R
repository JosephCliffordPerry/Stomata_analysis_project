source("Stomatal_decomposition_direct_inference/align_polygons.R")
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



sample_polygon_landmarks <- function(aligned, n_points = 200){
  
  resample_polygon <- function(g){
    
    xy <- sf::st_coordinates(g)[,1:2, drop = FALSE]
    
    # remove duplicated closing point
    if(all(xy[1,] == xy[nrow(xy),]))
      xy <- xy[-nrow(xy), , drop = FALSE]
    
    # edge vectors
    nxt <- rbind(xy[-1,,drop=FALSE], xy[1,,drop=FALSE])
    
    seg_len <- sqrt(rowSums((nxt - xy)^2))
    
    cum_len <- c(0, cumsum(seg_len))
    total_len <- tail(cum_len,1)
    
    target <- seq(0, total_len, length.out = n_points + 1)
    target <- target[-length(target)]      # don't duplicate first point
    
    pts <- matrix(NA_real_, n_points, 2)
    
    j <- 1
    
    for(i in seq_along(target)){
      
      while(cum_len[j+1] < target[i])
        j <- j + 1
      
      d <- target[i] - cum_len[j]
      
      if(seg_len[j] == 0){
        
        pts[i,] <- xy[j,]
        
      } else {
        
        alpha <- d / seg_len[j]
        
        pts[i,] <-
          (1-alpha)*xy[j,] +
          alpha*nxt[j,]
        
      }
      
    }
    
    ## ---------------------------------------------------
    ## choose a consistent starting landmark
    ## right-most point after alignment
    ## ---------------------------------------------------
    
    start <- which.max(pts[,1])
    
    pts <- rbind(
      pts[start:n_points,,drop=FALSE],
      pts[1:(start-1),,drop=FALSE]
    )
    
    pts
    
  }
  
  aligned$landmarks <-
    lapply(sf::st_geometry(aligned), resample_polygon)
  
  aligned
  
}


# ============================================================
# Build Active Shape Model (ASM)
# ============================================================

build_shape_model <- function(aligned, variance = 0.98){
  
  stopifnot("landmarks" %in% names(aligned))
  
  ## ----------------------------------------
  ## Convert landmarks to vectors
  ## ----------------------------------------
  
  X <- do.call(
    rbind,
    lapply(aligned$landmarks, function(x)
      as.vector(t(x)))
  )
  
  ## ----------------------------------------
  ## PCA
  ## ----------------------------------------
  
  pca <- prcomp(
    X,
    center = TRUE,
    scale. = FALSE
  )
  
  ## Number of modes to keep
  
  var_exp <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
  
  n_modes <- which(var_exp >= variance)[1]
  
  ## Mean shape
  
  mean_shape <- matrix(
    pca$center,
    ncol = 2,
    byrow = TRUE
  )
  
  list(
    
    mean_shape = mean_shape,
    
    modes = pca$rotation[,1:n_modes],
    
    eigenvalues = pca$sdev[1:n_modes]^2,
    
    pca = pca,
    
    variance_explained = var_exp,
    
    n_modes = n_modes,
    
    n_landmarks = nrow(mean_shape)
    
  )
  
}

reconstruct_shape <- function(model, weights = NULL){
  
  if(is.null(weights))
    weights <- rep(0, model$n_modes)
  
  x <- model$pca$center +
    model$modes %*% weights
  
  matrix(
    x,
    ncol = 2,
    byrow = TRUE
  )
  
}

plot_mode <- function(model,
                      mode = 1,
                      sd_mult = 2){
  
  sd <- sqrt(model$eigenvalues[mode])
  
  mean <- model$mean_shape
  
  minus <- reconstruct_shape(
    model,
    replace(
      rep(0, model$n_modes),
      mode,
      -sd_mult * sd
    )
  )
  
  plus <- reconstruct_shape(
    model,
    replace(
      rep(0, model$n_modes),
      mode,
      sd_mult * sd
    )
  )
  
  plot_shape(minus, col = "blue")
  
  plot_shape(mean,
             add = TRUE,
             col = "black")
  
  plot_shape(plus,
             add = TRUE,
             col = "red")
  
}

plot_shape <- function(shape,
                       add = FALSE,
                       col = "red",
                       lwd = 2){
  
  shape <- rbind(shape, shape[1,])
  
  if(!add){
    
    plot(
      shape,
      asp = 1,
      type = "l",
      col = col,
      lwd = lwd,
      xlab = "",
      ylab = ""
    )
    
  }else{
    
    lines(
      shape,
      col = col,
      lwd = lwd
    )
    
  }
  
}


project_shape <- function(shape, model){
  
  ## shape is a 200x2 landmark matrix
  
  x <- as.vector(t(shape))
  
  ## subtract mean
  
  x0 <- x - model$pca$center
  
  ## PCA coefficients
  
  b <- drop(
    t(model$modes) %*% x0
  )
  
  ## constrain to plausible shapes
  
  limit <- 3 * sqrt(model$eigenvalues)
  
  b <- pmax(
    pmin(b, limit),
    -limit
  )
  
  b
  
}
polys<-load_polygons(rda_dir)

aligned<-align_polygons(polys)

aligned <- sample_polygon_landmarks(aligned, 200)

model <- build_shape_model(aligned)

plot_shape(model$mean_shape)


plot_mode(model, mode = 1)

fit_initial_shape <- function(shape, model){
  
  b <- project_shape(shape, model)
  
  reconstruct_shape(
    model,
    b
  )
  
}


consensus <- aligned$landmarks[[1]]


initial <- fit_initial_shape(
  consensus,
  model
)


plot_shape(consensus,col="red")

plot_shape(
  initial,
  add=TRUE,
  col="blue"
)
d