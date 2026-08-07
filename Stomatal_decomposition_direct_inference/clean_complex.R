
refine_with_active_contour <- function(consensus_sf,img_path,density){

py_run_string("
import numpy as np
import cv2

def build_snake_inputs(image_path, density):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    density = density.astype(np.float32)
    density = (density - density.min()) / (density.max() - density.min() + 1e-8)

    return {
        'image': img,
        'density': density
    }
")
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

    image = image - image.min()
    if image.max() > 0:
        image = image / image.max()

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

globals()['run_active_contour'] = run_active_contour
")
snake_inputs <- py$build_snake_inputs(img_path, density)
snake_img <- snake_inputs$image

fit_ellipse <- function(coords){
  
  ctr <- colMeans(coords)
  covmat <- cov(coords)
  eig <- eigen(covmat)
  
  list(
    centre = ctr,
    major = 2 * sqrt(eig$values[1]),
    minor = 2 * sqrt(eig$values[2]),
    rotation = atan2(eig$vectors[2,1], eig$vectors[1,1])
  )
}


ellipse_points <- function(fit, n=200){
  
  t <- seq(0, 2*pi, length.out=n)
  
  x <- fit$major * cos(t)
  y <- fit$minor * sin(t)
  
  R <- matrix(c(
    cos(fit$rotation), -sin(fit$rotation),
    sin(fit$rotation),  cos(fit$rotation)
  ), 2,2, byrow=TRUE)
  
  pts <- cbind(x,y) %*% R
  
  pts[,1] <- pts[,1] + fit$centre[1]
  pts[,2] <- pts[,2] + fit$centre[2]
  
  pts
}

  
  coords <- st_coordinates(consensus_sf)[,1:2]
  coords <- coords[-nrow(coords),]
  
  fit <- fit_ellipse(coords)
  init <- ellipse_points(fit, n = 200)
  
  snake <- py$run_active_contour(
    snake_img,
    init,
    0.01,   # alpha
    5,      # beta
    0.001,  # gamma
    0,
    1,
    2,      # sigma
    500
  )
  
  snake <- py_to_r(snake)
  
  snake <- rbind(snake, snake[1,])
  
  list(
    ellipse = st_sfc(st_polygon(list(rbind(init, init[1,])))),
    snake = st_sfc(st_polygon(list(snake)))
  )
}

