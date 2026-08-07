compute_alignment <- function(samples){
  
  for(i in seq_along(samples)){
    
    message(samples[[i]]$image)
    
    poly <- samples[[i]]$consensus_polygon
    
    if(is.null(poly))
      next
    
    xy <- sf::st_coordinates(poly)[,1:2,drop=FALSE]
    xy <- xy[complete.cases(xy),]
    
    if(nrow(xy) < 5){
      samples[[i]]$transform <- NULL
      next
    }
    
    if(all(xy[1,] == xy[nrow(xy),])){
      xy <- xy[-nrow(xy),]
    }
    
    #---------------------------------
    # polygon centre
    #---------------------------------
    
    polygon_centre <- colMeans(xy)
    
    if(any(!is.finite(polygon_centre))){
      samples[[i]]$transform <- NULL
      next
    }
    
    angle <- samples[[i]]$chain_angle
    
    if(is.null(angle) || is.na(angle)){
      
      message("missing chain angle, skipping")
      
      samples[[i]]$transform <- NULL
      next
    }
    
    angle <- angle * pi / 180
    
    theta <- -angle
    
    R <- matrix(
      c(
        cos(theta),
        -sin(theta),
        sin(theta),
        cos(theta)
      ),
      nrow = 2,
      byrow = TRUE
    )
    
    #---------------------------------
    # polygon scale
    #---------------------------------
    
    X_poly <- sweep(
      xy,
      2,
      polygon_centre
    )
    
    X_poly_rot <- X_poly %*% t(R)
    
    polygon_scale <- sqrt(
      mean(
        rowSums(X_poly_rot^2)
      )
    )
    
    if(!is.finite(polygon_scale) || polygon_scale == 0)
      polygon_scale <- 1
    
    #---------------------------------
    # mask centre and scale
    #---------------------------------
    
    mask_centre <- polygon_centre
    mask_scale <- polygon_scale
    
    density <- samples[[i]]$density
    
    if(!is.null(density)){
      
      coords <- which(
        density > 0,
        arr.ind = TRUE
      )
      
      if(length(coords) > 0 && nrow(coords) > 0){
        
        mask_xy <- coords[,c(2,1),drop=FALSE]
        
        mask_centre <- colMeans(mask_xy)
        
        X_mask <- sweep(
          mask_xy,
          2,
          mask_centre
        )
        
        X_mask_rot <- X_mask %*% t(R)
        
        mask_scale <- sqrt(
          mean(
            rowSums(X_mask_rot^2)
          )
        )
        
        if(!is.finite(mask_scale) || mask_scale == 0)
          mask_scale <- 1
      }
    }
    
    message(
      "polygon_scale = ",
      round(polygon_scale,3),
      " ; mask_scale = ",
      round(mask_scale,3)
    )
    
    samples[[i]]$transform <- list(
      
      polygon_centre = polygon_centre,
      
      mask_centre = mask_centre,
      
      R = t(R),
      
      R_inv = R,
      
      polygon_scale = polygon_scale,
      
      mask_scale = mask_scale,
      
      reflected = FALSE
      
    )
  }
  
  samples
}
remove_holes <- function(poly) {
  geom <- sf::st_geometry(poly)[[1]]
  sf::st_sfc(
    sf::st_polygon(list(geom[[1]])),
    crs = sf::st_crs(poly)
  )
}
align_polygon <- function(poly, transform){
  
  poly <- remove_holes(poly)

  xy <- sf::st_coordinates(poly)[,1:2]
  
  pts <- sweep(
    xy,
    2,
    transform$polygon_centre
  )
  
  pts <- pts %*% transform$R
  
  pts <- pts / transform$polygon_scale
  
  if(transform$reflected)
    pts[,1] <- -pts[,1]
  
  pts[nrow(pts),] <- pts[1,]
  
  sf::st_sfc(
    sf::st_polygon(list(pts))
  )
}
align_polygons <- function(samples){
  
  for(i in seq_along(samples)){
    
    tr <- samples[[i]]$transform
    
    if(is.null(tr))
      next
    
    samples[[i]]$aligned_polygon <-
      align_polygon(
        samples[[i]]$consensus_polygon,
        tr
      )
  }
  
  samples
}

build_aligned_landmarks <- function(samples,
                                    n_points=200){
  
  for(i in seq_along(samples)){
    
    poly <- samples[[i]]$aligned_polygon
    
    if(is.null(poly))
      next
    
    samples[[i]]$landmarks <-
      resample_polygon(poly,n_points)
  }
  
  samples
}

align_density <- function(samples,
                          grid_size = 200){
  
  for(i in seq_along(samples)){
    
    message(samples[[i]]$image)
    
    tr <- samples[[i]]$transform
    
    if(is.null(tr))
      next
    
    samples[[i]]$aligned_density <-
      warp_density_to_grid(
        density = samples[[i]]$density,
        transform = tr,
        grid_size = grid_size
      )
  }
  
  samples
}  
  #--------------------------------------------------
  # Helper: resample polygon
  #--------------------------------------------------
resample_polygon <- function(g,
                             n_points = 200){
  g <- remove_holes(g)
  xy <- sf::st_coordinates(g)[,1:2,drop=FALSE]
  
  if(all(xy[1,] == xy[nrow(xy),]))
    xy <- xy[-nrow(xy),,drop=FALSE]
  
  nxt <- rbind(
    xy[-1,,drop=FALSE],
    xy[1,,drop=FALSE]
  )
  
  seg_len <- sqrt(
    rowSums(
      (nxt - xy)^2
    )
  )
  
  cum_len <- c(0, cumsum(seg_len))
  total <- tail(cum_len,1)
  
  target <- seq(
    0,
    total,
    length.out = n_points + 1
  )[-(n_points + 1)]
  
  pts <- matrix(
    NA_real_,
    n_points,
    2
  )
  
  j <- 1
  
  for(i in seq_along(target)){
    
    while(cum_len[j+1] < target[i])
      j <- j + 1
    
    d <- target[i] - cum_len[j]
    
    if(seg_len[j] == 0){
      
      pts[i,] <- xy[j,]
      
    } else {
      
      a <- d / seg_len[j]
      
      pts[i,] <-
        (1-a) * xy[j,] +
        a * nxt[j,]
    }
  }
  
  start <- which.max(pts[,1])
  
  pts <- pts[
    c(start:n_points,
      seq_len(start-1)),
    ,
    drop=FALSE
  ]
  
  pts
}
  
  #--------------------------------------------------
  # Helper: warp mask
  #--------------------------------------------------
warp_density_to_grid <- function(
    density,
    transform,
    grid_size = 200,
    extent = 1.5
){
  
  coords <- which(
    density > 0,
    arr.ind = TRUE
  )
  
  if(nrow(coords) == 0)
    return(matrix(0, grid_size, grid_size))
  
  xy <- coords[,c(2,1),drop=FALSE]
  
  values <- density[coords]
  
  #-----------------------------
  # centre
  #-----------------------------
  
  xy <- sweep(
    xy,
    2,
    transform$mask_centre
  )
  
  #-----------------------------
  # rotate
  #-----------------------------
  
  xy_rot <- xy %*% transform$R
  
  message(
    "rotated x: ",
    paste(round(range(xy_rot[,1]),3), collapse=" "),
    " | y: ",
    paste(round(range(xy_rot[,2]),3), collapse=" ")
  )
  
  #-----------------------------
  # scale
  #-----------------------------
  
  xy <- xy_rot / transform$mask_scale
  
  message(
    "scaled x: ",
    paste(round(range(xy[,1]),3), collapse=" "),
    " | y: ",
    paste(round(range(xy[,2]),3), collapse=" ")
  )
  
  if(transform$reflected)
    xy[,1] <- -xy[,1]
  
  gx <- ((xy[,1] + extent)/(2*extent)) *
    (grid_size - 1) + 1
  
  gy <- ((xy[,2] + extent)/(2*extent)) *
    (grid_size - 1) + 1
  
  gx <- round(gx)
  gy <- round(gy)
  
  keep <-
    gx >= 1 &
    gx <= grid_size &
    gy >= 1 &
    gy <= grid_size
  
  out <- matrix(
    0,
    grid_size,
    grid_size
  )
  
  counts <- matrix(
    0,
    grid_size,
    grid_size
  )
  
  for(k in which(keep)){
    
    out[gy[k],gx[k]] <-
      out[gy[k],gx[k]] + values[k]
    
    counts[gy[k],gx[k]] <-
      counts[gy[k],gx[k]] + 1
  }
  
  keep_cells <- counts > 0
  
  out[keep_cells] <-
    out[keep_cells] /
    counts[keep_cells]
  
  out
}


prepare_aligned_shapes <- function(
    samples,
    n_landmarks = 200,
    grid_size = 200
){
  
  message("Preparing aligned shapes...")
  
  samples <- compute_alignment(samples)
  
  samples <- align_polygons(samples)
  
  samples <- build_aligned_landmarks(
    samples,
    n_points = n_landmarks
  )
  
  samples <- align_density(
    samples,
    grid_size = grid_size
  )
  
  samples
}