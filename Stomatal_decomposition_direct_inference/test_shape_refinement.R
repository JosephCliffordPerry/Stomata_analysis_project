source("D:/Stomatal_analysis_project/Stomatal_decomposition_direct_inference/ASM_itteration.R")
test_shape_refinement <- function(
    sample,
    model,
    probability_map,
    n_landmarks = 200,
    edge_sigma = 1,
    n_iter = 15,
    search_width = 6,
    display = TRUE
){
  plot_shape <- function(shape,
                       add = FALSE,
                       col = "red",
                       lwd = 2,
                       xlim = NULL,
                       ylim = NULL){

  shape <- rbind(shape, shape[1,])

  if(is.null(xlim))
    xlim <- range(shape[,1])

  if(is.null(ylim))
    ylim <- range(shape[,2])

  if(!add){

    plot(
      shape,
      type = "l",
      asp = 1,
      xlim = xlim,
      ylim = ylim,
      col = col,
      lwd = lwd,
      xlab = "",
      ylab = ""
    )

  } else {

    lines(
      shape,
      col = col,
      lwd = lwd
    )

  }
}
  stopifnot(!is.null(sample$consensus_mask))
  stopifnot(!is.null(sample$consensus_polygon))
  overlay_refined_shape <- function(image_path,
                                    polygon,
                                    col = "red",
                                    lwd = 2){
    
    img <- png::readPNG(image_path)
    
    h <- dim(img)[1]
    
    plot(
      c(1, ncol(img)),
      c(h, 1),
      type = "n",
      asp = 1,
      xlab = "",
      ylab = "",
      axes = FALSE
    )
    
    rasterImage(
      img,
      1,
      h,
      ncol(img),
      1
    )
    
    polygon <- rbind(
      polygon,
      polygon[1,]
    )
    
    lines(
      polygon,
      col = col,
      lwd = lwd
    )
  }
  ## ------------------------------------------------------------
  ## Original image
  ## ------------------------------------------------------------
  
  img <- imager::load.image(sample$image_path)
  
  ## ------------------------------------------------------------
  ## Edge map
  ## ------------------------------------------------------------
  
  edge_map <- compute_edge_map(
    sample$image_path
  )
  
  ## ------------------------------------------------------------
  ## Align into model space
  ## ------------------------------------------------------------
  
  # aligned <- prepare_aligned_shape(
  #   sample,
  #   n_landmarks = n_landmarks
  # )
  
  aligned<- sample
  
  transform <- aligned$transform
  
  consensus <- aligned$landmarks
  
  ## ------------------------------------------------------------
  ## Initial ASM fit
  ## ------------------------------------------------------------
  
  initial <- fit_initial_shape(
    consensus,
    model
  )
  
  ## ------------------------------------------------------------
  ## Shape refinement
  ## ------------------------------------------------------------
  
  history <- vector(
    "list",
    n_iter + 1
  )
  
  history[[1]] <- initial
  
  shape <- initial
  
  for(i in seq_len(n_iter)){
    
    shape <- asm_iteration(
      shape = shape,
      model = model,
      edge_map = edge_map,
      probability_map = probability_map,
      search_width = search_width
    )
    
    history[[i + 1]] <- shape
  }
  
  ## ------------------------------------------------------------
  ## Restore image coordinates
  ## ------------------------------------------------------------
  
  restored <- reverse_alignment(
    shape,
    transform
  )
  
  ## ------------------------------------------------------------
  ## Display
  ## ------------------------------------------------------------
  
  if(display){
    
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar))
    
    par(mfrow = c(2,3))
    
    ## Original
    
    plot(img)
    
    ## Consensus mask
    
    image(sample$consensus_mask)
    
    ## Edge map
    
    image(as.matrix(edge_map))
    
    ## Mean + consensus
    
    all_pts <- rbind(
      model$mean_shape,
      consensus,
      initial,
      shape
    )
    
    xlim <- range(all_pts[,1])
    ylim <- range(all_pts[,2])
    
    plot_shape(
      model$mean_shape,
      col="blue",
      xlim=xlim,
      ylim=ylim
    )
    
    plot_shape(
      consensus,
      add=TRUE,
      col="red"
    )
    
    ## Initial fit
    
    plot_shape(
      consensus,
      col = "black"
    )
    
    plot_shape(
      initial,
      add = TRUE,
      col = "blue"
    )
    
    ## Final overlay
    
    overlay_refined_shape(
      sample$image_path,
      restored
    )
  }
  
  list(
    
    image = img,
    
    edge_map = edge_map,
    
    transform = transform,
    
    consensus = consensus,
    
    initial_shape = initial,
    
    refined_shape = shape,
    
    restored_polygon = restored,
    
    history = history
    
  )
  
}
test_shape_refinement(sample = aligned_Stomata[[sample(1:length(aligned_Stomata),1)]],model = model)

####################
library(ggplot2)
library(png)

img <- readPNG(sample$image_path)

if(length(dim(img)) == 2){
  img <- array(rep(img, 3), dim = c(nrow(img), ncol(img), 3))}

h <- dim(img)[1]
w <- dim(img)[2]
history_image <- lapply(history, reverse_alignment, transform = transform)
shape_df <- history_to_df(history_image)

# image coordinates -> plotting coordinates
shape_df$y <- h - shape_df$y

ggplot() +
  annotation_raster(
    raster = as.raster(img),
    xmin = 0,
    xmax = w,
    ymin = 0,
    ymax = h
  ) +
  geom_path(
    data = shape_df,
    aes(
      x = x,
      y = y,
      group = frame,
      alpha = frame
    ),
    linewidth = 1
  ) 
   +
  scale_alpha(range = c(0.05, 1), guide = "none")

