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

