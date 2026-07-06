py_run_string("
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian

def run_active_contour(image, init_snake,
                       alpha=0.01,
                       beta=5,
                       gamma=0.001,
                       w_line=0,
                       w_edge=1,
                       sigma=2,
                       max_num_iter=500):

    image = image.astype(np.float64)

    image -= image.min()

    if image.max() > 0:
        image /= image.max()

    image = gaussian(image, sigma=sigma)

    snake = active_contour(
        image,
        init_snake,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        w_line=w_line,
        w_edge=w_edge,
        max_num_iter=max_num_iter,
        boundary_condition='periodic'
    )

    return snake
")

refine_with_active_contour <- function(
    consensus_sf,
    snake_img,
    n_points = 250,
    alpha = 0.01,
    beta = 5,
    gamma = 0.001,
    sigma = 2,
    max_iter = 500
){
  
  library(sf)
  
  coords <- st_coordinates(consensus_sf)[,1:2]
  
  coords <- coords[-nrow(coords),]
  
  centre <- colMeans(coords)
  
  eig <- eigen(cov(coords))
  
  major <- 2 * sqrt(eig$values[1])
  
  minor <- 2 * sqrt(eig$values[2])
  
  theta <- atan2(
    eig$vectors[2,1],
    eig$vectors[1,1]
  )
  
  t <- seq(0,2*pi,length.out=n_points)
  
  ellipse <- cbind(
    major*cos(t),
    minor*sin(t)
  )
  
  R <- matrix(
    c(
      cos(theta),-sin(theta),
      sin(theta), cos(theta)
    ),
    2,2,
    byrow=TRUE
  )
  
  ellipse <- ellipse %*% R
  
  ellipse[,1] <- ellipse[,1] + centre[1]
  ellipse[,2] <- ellipse[,2] + centre[2]
  
  snake <- py$run_active_contour(
    snake_img,
    ellipse,
    alpha,
    beta,
    gamma,
    0,
    1,
    sigma,
    as.integer(max_iter)
  )
  
  snake <- py_to_r(snake)
  
  snake <- rbind(
    snake,
    snake[1,,drop=FALSE]
  )
  
  snake_sf <- st_sf(
    geometry = st_sfc(
      st_polygon(list(snake))
    )
  )
  
  list(
    ellipse = st_sf(
      geometry = st_sfc(
        st_polygon(
          list(
            rbind(ellipse, ellipse[1,])
          )
        )
      )
    ),
    snake = snake_sf
  )
}