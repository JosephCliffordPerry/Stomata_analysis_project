library(sf)
library(dplyr)
library(purrr)
library(tidyr)


# ============================================================
# Build Active Shape Model (ASM)
# ============================================================

build_shape_model <- function(samples, variance = 0.98){
  
  X <- do.call(
    rbind,
    lapply(samples, function(s){
      
      if(is.null(s$landmarks))
        return(NULL)
      
      as.vector(t(s$landmarks))
      
    })
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

# 
# model <- build_shape_model(aligned_Stomata)
# 
# plot_shape(model$mean_shape)
# 
# 
# plot_mode(model, mode = 1)

fit_initial_shape <- function(shape, model){
  
  b <- project_shape(shape, model)
  
  reconstruct_shape(
    model,
    b
  )
  
}

# 
# consensus <- aligned$landmarks[[1]]
# 
# 
# initial <- fit_initial_shape(
#   consensus,
#   model
# )
# 
# 
# plot_shape(consensus,col="red")
# 
# plot_shape(
#   initial,
#   add=TRUE,
#   col="blue"
# )

