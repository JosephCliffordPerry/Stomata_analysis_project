
#theory notes 
# due to the instability of the PCA putting the shapes into it are unlikely to help as it 
# likely contains many points that are errors however it does give a very good way to check
# the likely hood of a point being in a place without just using a general shape and may be more 
# interpretable than using the matrices I should then be able to calculate a normal displacement rule set 
# that is informed by both the likelyhood of the original polygon points being that shape, the image and 
# the aligned landmarks 



library(ggplot2)
library(reshape2)
library(patchwork)

warp_probability_to_image <- function(
    probability_map,
    transform,
    image_dims,
    extent = 1.5){
  
  H <- image_dims[1]
  W <- image_dims[2]
  
  atlas_size <- nrow(probability_map)
  
  out <- matrix(
    0,
    H,
    W
  )
  
  for(y in seq_len(H)){
    
    for(x in seq_len(W)){
      
      p <- matrix(
        c(x, y),
        nrow = 1
      )
      
      aligned <- sweep(
        p,
        2,
        transform$polygon_centre
      )
      
      aligned <- aligned %*%
        transform$R_inv
      
      aligned <- aligned /
        transform$polygon_scale
      
      ax <- aligned[1]
      ay <- aligned[2]
      
      gx <-
        ((ax + extent) /
           (2 * extent)) *
        (atlas_size - 1) + 1
      
      gy <-
        ((ay + extent) /
           (2 * extent)) *
        (atlas_size - 1) + 1
      
      gx <- round(gx)
      gy <- round(gy)
      
      if(
        gx >= 1 &&
        gx <= atlas_size &&
        gy >= 1 &&
        gy <= atlas_size
      ){
        
        out[y, x] <-
          probability_map[
            gy,
            gx
          ]
      }
    }
  }
  
  out
}
update_transform <- function(shape, transform){
  
  centre <- colMeans(shape)
  
  X <- sweep(
    shape,
    2,
    centre
  )
  
  scale <- sqrt(
    mean(
      rowSums(X^2)
    )
  )
  
  transform$polygon_centre <- centre
  transform$polygon_scale <- scale
  
  transform
}
compute_normals <- function(shape){
  
  n <- nrow(shape)
  
  prev <- rbind(
    shape[n,],
    shape[-n,,drop=FALSE]
  )
  
  nxt <- rbind(
    shape[-1,,drop=FALSE],
    shape[1,]
  )
  
  tangent <- nxt - prev
  
  normals <- cbind(
    -tangent[,2],
    tangent[,1]
  )
  
  len <- sqrt(
    rowSums(normals^2)
  )
  
  normals / pmax(len,1e-8)
}

sample_profile <- function(
    edge_map,
    atlas_map,
    density_map,
    point,
    normal,
    width = 8){
  
  offsets <- -width:width
  
  edge_profile <- rep(NA_real_, length(offsets))
  density_profile <- rep(NA_real_, length(offsets))
  atlas_profile <- rep(NA_real_, length(offsets))
  
  valid <- rep(FALSE, length(offsets))
  
  for(i in seq_along(offsets)){
    
    d <- offsets[i]
    
    pos <- point + d * normal
    
    x <- round(pos[1])
    y <- round(pos[2])
    
    if(
      x < 1 ||
      y < 1 ||
      x > ncol(edge_map) ||
      y > nrow(edge_map)
    ){
      next
    }
    
    valid[i] <- TRUE
    
    edge_profile[i] <-
      edge_map[y,x]
    
    density_profile[i] <-
      density_map[y,x]
    
    atlas_profile[i] <-
      atlas_map[y,x]
  }
  
  #
  # not enough valid samples
  #
  if(sum(valid) < 3){
    return(   list(     shift = 0,     score = NULL,     grad_offsets = NULL   ) )
  }
  
  score <- rep(NA_real_, length(offsets) - 1)
  
  for(i in seq_len(length(score))){
    
    #
    # only use gradients between
    # two valid neighbouring samples
    #
    if(!(valid[i] && valid[i + 1])){
      next
    }
    
    edge_grad <-
      abs(
        edge_profile[i + 1] -
          edge_profile[i]
      )
    
    density_grad <-
      abs(
        density_profile[i + 1] -
          density_profile[i]
      )
    
    raw_score <- rep(NA_real_, length(offsets) - 1)
    
    for(i in seq_len(length(raw_score))){
      
      if(!(valid[i] && valid[i + 1])){
        next
      }
      
      edge_grad <-
        abs(
          edge_profile[i + 1] -
            edge_profile[i]
        )
      
      density_grad <-
        abs(
          density_profile[i + 1] -
            density_profile[i]
        )
      
      raw_score[i] <-
        1.5 * edge_grad +
        1.5 * density_grad +
        1.5 * atlas_profile[i + 1]
    }
    score <- raw_score
    
    for(i in seq_along(raw_score)){
      
      if(is.na(raw_score[i]))
        next
      
      lo <- max(1, i - 3)
      hi <- min(length(raw_score), i + 3)
      
      support <-
        sum(
          raw_score[lo:hi] >
            0.7 * raw_score[i],
          na.rm = TRUE
        )
      
      score[i] <-
        raw_score[i] *
        support
    }
    
  }
  
  #
  # no usable evidence
  #
  if(all(is.na(score))){
    return(   list(     shift = 0,     score = NULL,     grad_offsets = NULL   ) )
  }
  
  max_score <- max(
    score,
    na.rm = TRUE
  )
  
  #
  # flat profile -> don't move
  #
  if(
    !is.finite(max_score) ||
    max_score < 1e-6
  ){
    return(   list(     shift = 0,     score = NULL,     grad_offsets = NULL   ) )
  }
  
  grad_offsets <-
    (
      offsets[-length(offsets)] +
        offsets[-1]
    ) / 2
  
  centre_idx <-
    which.min(abs(grad_offsets))
  
  centre_score <-
    score[centre_idx]
  
  best_idx <-
    which.max(score)
  
  best_score <-
    score[best_idx]
  
  if(
    is.na(centre_score) ||
    !is.finite(centre_score) ||
    is.na(best_score) ||
    !is.finite(best_score)
  ){
    return(   list(     shift = 0,     score = NULL,     grad_offsets = NULL   ) )
  }
  
  if(
    centre_score >=
    0.95 * best_score
  ){
    return(   list(     shift = 0,     score = NULL,     grad_offsets = NULL   ) )
  }
  #
  # already near optimum
  #
  
  
  best_offset <-
    grad_offsets[best_idx]
  cat(
    "centre_offset:",
    round(grad_offsets[centre_idx],2),
    "best_offset:",
    round(grad_offsets[best_idx],2),
    "centre_score:",
    round(centre_score,3),
    "best_score:",
    round(best_score,3),
    "\n"
  )
  list(
    shift = best_offset,
    score = score,
    grad_offsets = grad_offsets
  )
  

}

to_asm_space <- function(shape,
                         transform){
  
  x <- sweep(
    shape,
    2,
    transform$polygon_centre
  )
  
  x <- x %*% transform$R
  
  x <- x / transform$polygon_scale
  
  x
}
from_asm_space <- function(shape, transform){
  
  x <- shape * transform$polygon_scale
  
  x <- x %*% transform$R_inv
  
  x <- sweep(
    x,
    2,
    transform$polygon_centre,
    "+"
  )
  
  x
}
smooth_shape <- function(
    shape,
    alpha = 0.15){
  
  prev <- rbind(
    shape[nrow(shape),],
    shape[-nrow(shape),]
  )
  
  nxt <- rbind(
    shape[-1,],
    shape[1,]
  )
  
  (1-alpha)*shape +
    alpha*(prev+nxt)/2
}
smooth_shifts <- function(
    shifts,
    n_iter = 5){
  
  n <- length(shifts)
  
  for(k in seq_len(n_iter)){
    
    prev <- c(
      shifts[n],
      shifts[-n]
    )
    
    nxt <- c(
      shifts[-1],
      shifts[1]
    )
    
    shifts <-
      (
        prev +
          2 * shifts +
          nxt
      ) / 4
  }
  
  shifts
}


compute_shifts <- function(
    shape,
    edge_map,
    atlas_map,
    density_map,
    width = 8){
  
  normals <- compute_normals(shape)
  
  shifts <- numeric(
    nrow(shape)
  )
  
  response_map <- matrix(
    0,
    nrow(edge_map),
    ncol(edge_map)
  )
  
  for(i in seq_len(nrow(shape))){
    
    result <-
      sample_profile(
        edge_map,
        atlas_map,
        density_map,
        shape[i,],
        normals[i,],
        width
      )
    
    shifts[i] <- result$shift
    
    if(
      is.null(result$score) ||
      is.null(result$grad_offsets)
    ){
      next
    }
    
    for(k in seq_along(result$score)){
      
      if(is.na(result$score[k]))
        next
      
      pos <-
        shape[i,] +
        result$grad_offsets[k] *
        normals[i,]
      
      x <- round(pos[1])
      y <- round(pos[2])
      
      if(
        x >= 1 &&
        y >= 1 &&
        x <= ncol(edge_map) &&
        y <= nrow(edge_map)
      ){
        
        response_map[y,x] <-
          response_map[y,x] +
          result$score[k]
      }
    }
  }
  
  list(
    shifts = shifts,
    normals = normals,
    response_map = response_map
  )
}


apply_combined_search <- function(
    shape,
    edge_map,
    atlas_map,
    density_map,
    width = 8){
  
  result <-
    compute_shifts(
      shape,
      edge_map,
      atlas_map,
      density_map,
      width
    )
  
  shifts <-
    smooth_shifts(
      result$shifts,
      n_iter = 5
    )
  # shifts <-result$shifts
  normals <- result$normals
  
  search_start <- shape - width * normals
  
  search_end <- shape + width * normals
  
  chosen <- shape + shifts * normals
  
  updated <- shape
  
  for(i in seq_len(nrow(shape))){
    
    updated[i,] <-
      shape[i,] +
      shifts[i] * normals[i,]
    
    updated[i,1] <- pmin(
      ncol(edge_map),
      pmax(1, updated[i,1])
    )
    
    updated[i,2] <- pmin(
      nrow(edge_map),
      pmax(1, updated[i,2])
    )
  }
  
  list(
    shape = updated,
    shifts = shifts,
    normals = normals,
    search_start = search_start,
    search_end = search_end,
    chosen = chosen,
    response_map = result$response_map
  )
}

asm_iteration <- function(
    shape,
    model,
    edge_map,
    density_map,
    probability_map,
    transform,
    search_width = 8){
  
  # transform <-
  #   update_transform(
  #     shape,
  #     transform
  #   )
  
  atlas_map <-
    warp_probability_to_image(
      probability_map,
      transform,
      image_dims =
        dim(edge_map)
    )
  
  search_result <-
    apply_combined_search(
      shape,
      edge_map,
      atlas_map,
      density_map,
      width = search_width
    )
  
  candidate <-
    search_result$shape
  
  edge_grad_map <-
    compute_gradient_map(edge_map)
  
  density_grad_map <-
    compute_gradient_map(density_map)
  
  combined_grad_map <-
    1.5 * edge_grad_map +
    1.5 * density_grad_map
  candidate <-
    smooth_shape(
      candidate,
      alpha = 0.15
    )
  
  candidate_aligned <-
    to_asm_space(
      candidate,
      transform
    )
  
  weights <-
    project_shape(
      candidate_aligned,
      model
    )
  
  shape_aligned <-
    reconstruct_shape(
      model,
      weights
    )
  
  # fitted <-
  #   from_asm_space(
  #     shape_aligned,
  #     transform
  #   )
  # 
  fitted<- candidate
  list(
    shape = fitted,
    
    atlas_map = atlas_map,
    
    transform = transform,
    
    debug = list(
      
      shape_before = shape,
      
      candidate = candidate,
      
      fitted = fitted,
      
      candidate_aligned =
        candidate_aligned,
      
      shape_aligned =
        shape_aligned,
      
      shifts =
        search_result$shifts,
      
      normals =
        search_result$normals,
      
      edge_grad_map =
        edge_grad_map,
      
      density_grad_map =
        density_grad_map,
      
      combined_grad_map =
        
        search_result$response_map,
      
      edge_map =
        edge_map,
      
      density_map =
        density_map,
      
      atlas_map =
        atlas_map,
      
      search_start =
        search_result$search_start,
      
      search_end =
        search_result$search_end,
      
      chosen =
        search_result$chosen
      
    )
  )
}
compute_gradient_map <- function(mat){
  
  gx <- matrix(
    0,
    nrow(mat),
    ncol(mat)
  )
  
  gy <- matrix(
    0,
    nrow(mat),
    ncol(mat)
  )
  
  gx[,2:(ncol(mat)-1)] <-
    mat[,3:ncol(mat)] -
    mat[,1:(ncol(mat)-2)]
  
  gy[2:(nrow(mat)-1),] <-
    mat[3:nrow(mat),] -
    mat[1:(nrow(mat)-2),]
  
  sqrt(
    gx^2 + gy^2
  )
}

matrix_to_df <- function(mat){
  
  df <- melt(mat)
  
  names(df) <- c(
    "y",
    "x",
    "value"
  )
  
  df
}
shape_to_df <- function(shape, image_height){
  
  shape <- rbind(
    shape,
    shape[1,,drop=FALSE]
  )
  
  data.frame(
    x = shape[,1],
    y = image_height - shape[,2]
  )
}
run_asm <- function(
    image_matrix,
    shape,
    model,
    edge_map,
    density_map,
    probability_map,
    transform,
    max_iter = 25){
  history <- list()
  for(iter in 1:max_iter){
    
    previous <- shape
    
    result <-
      asm_iteration(
        shape,
        model,
        edge_map,
        density_map,
        probability_map,
        transform,
        search_width = 8
      )
    history[[iter]] <- result$debug
    shape <- result$shape
   
    transform <- result$transform
    
    
    movement <- mean(
      sqrt(
        rowSums(
          (shape - previous)^2
        )
      )
    )
    
    cat(
      "iteration:",
      iter,
      "movement:",
      round(movement,4),
      "\n"
    )
    
    if(movement < 0.05){
      
      cat("Converged\n")
      
      break
    }
  }
  list(
    
    final_shape = shape,
    
    history = history
  )
}

get_image_matrix <- function(sample){
  

    
    img <- imager::load.image(
      sample$image_path
    )
    
 
  
  if(length(dim(img)) == 4){
    
    img <- imager::grayscale(img)
    
    img <- as.array(img)[,,1,1]
    
    img <- t(img)
    
  }
  
  img
}
sample<-aligned_Stomata[[1382]]

edge_map <-
  compute_edge_map(
    sample$image_path
  )

final_shape <-
  run_asm(
    image_matrix =
      get_image_matrix(sample),
    
    shape =
      resample_polygon(
        sample$consensus_polygon,
        n_points =
          model$n_landmarks
      ),
    
    model = model,
    
    edge_map =
      edge_map,
    
    density_map =
      sample$density,
    
    probability_map =
      probability_map,
    
    transform =
      sample$transform,
    
    max_iter = 20
  )


