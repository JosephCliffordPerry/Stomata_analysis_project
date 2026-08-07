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