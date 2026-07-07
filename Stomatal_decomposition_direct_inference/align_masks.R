align_polygons <- function(polygons){
  
  out <- list()
  
  files <- unique(polygons$file)
  
  for(f in files){
    
    message(f)
    
    img <- polygons |>
      dplyr::filter(file == f)
    
    consensus <- img |>
      dplyr::filter(object == "Consensus") |>
      dplyr::slice(1)
    
    xy <- sf::st_coordinates(consensus)[,1:2, drop = FALSE]
    xy <- xy[complete.cases(xy), , drop = FALSE]
    
    if(nrow(xy) < 3){
      message("Skipping (degenerate polygon): ", f)
      next
    }
    
    # ---------------------------
    # CENTRE
    # ---------------------------
    centre <- colMeans(xy)
    X <- sweep(xy, 2, centre)
    
    if(nrow(X) < 2){
      message("Skipping (not enough variance): ", f)
      next
    }
    
    # ---------------------------
    # PCA ROTATION
    # ---------------------------
    pca <- prcomp(X)
    
    theta <- atan2(
      pca$rotation[2,1],
      pca$rotation[1,1]
    )
    
    R <- matrix(c(
      cos(-theta), -sin(-theta),
      sin(-theta),  cos(-theta)
    ), 2, 2)
    
    # ---------------------------
    # SCALE (RMS radius)
    # ---------------------------
    X_rot <- X %*% R
    scale <- sqrt(mean(rowSums(X_rot^2)))
    
    if(scale <= 0 || is.na(scale))
      scale <- 1
    
    # ---------------------------
    # REFLECTION RULE
    # ---------------------------
    reflected <- FALSE
    
    if(mean(X_rot[,1]) > 0){
      X_rot[,1] <- -X_rot[,1]
      reflected <- TRUE
    }
    
    # ---------------------------
    # FULL TRANSFORM MATRIX
    # (clean inverse handling later)
    # ---------------------------
    A <- R / scale
    
    transform <- list(
      centre = centre,
      A = A,
      A_inv = solve(A),
      reflected = reflected
    )
    
    img$transform <- rep(list(transform), nrow(img))
    
    # ---------------------------
    # APPLY TRANSFORM TO ALL GEOMS
    # ---------------------------
    aligned <- lapply(img$geometry, function(g){
      
      crd <- sf::st_coordinates(g)[,1:2]
      
      pts <- sweep(crd, 2, centre) %*% A
      
      if(reflected)
        pts[,1] <- -pts[,1]
      
      pts[nrow(pts), ] <- pts[1, ]
      
      sf::st_sfc(sf::st_polygon(list(pts)))
      
    })
    
    img$geometry <- do.call(c, aligned)
    
    out[[f]] <- img
  }
  
  dplyr::bind_rows(out)
}


######################
warp_mask <- function(mask, centre, A, reflected){
  
  coords <- which(mask, arr.ind = TRUE)
  coords <- coords[, c(2,1)]
  
  coords <- sweep(coords, 2, centre) %*% A
  
  if(reflected)
    coords[,1] <- -coords[,1]
  
  new <- matrix(0, nrow(mask), ncol(mask))
  
  coords <- round(coords)
  
  ok <- coords[,1] >= 1 & coords[,1] <= ncol(mask) &
    coords[,2] >= 1 & coords[,2] <= nrow(mask)
  
  coords <- coords[ok, , drop=FALSE]
  
  new[cbind(coords[,2], coords[,1])] <- 1
  
  new
}
align_masks <- function(masks){
  
  out <- list()
  
  for(i in seq_along(masks)){
    
    consensus <- masks[[i]]$masks[[1]]  # or chosen consensus mask
    
    xy <- which(consensus, arr.ind = TRUE)
    xy <- xy[, c(2,1)]
    
    centre <- colMeans(xy)
    X <- sweep(xy, 2, centre)
    
    pca <- prcomp(X)
    
    theta <- atan2(pca$rotation[2,1], pca$rotation[1,1])
    
    R <- matrix(c(
      cos(-theta), -sin(-theta),
      sin(-theta),  cos(-theta)
    ), 2)
    
    Xr <- X %*% R
    
    scale <- sqrt(mean(rowSums(Xr^2)))
    A <- R / scale
    
    reflected <- FALSE
    if(mean(Xr[,1]) > 0){
      reflected <- TRUE
    }
    
    transform <- list(
      centre = centre,
      A = A,
      reflected = reflected
    )
    
    aligned_masks <- lapply(masks[[i]]$masks, function(m){
      
      warp_mask(m, centre, A, reflected)
      
    })
    
    out[[i]] <- list(
      masks = aligned_masks,
      transform = transform
    )
  }
  
  out
}

