library(reticulate)
library(tiff)
library(polyclip)
library(ggplot2)
library(grid)
library(sf)

Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","ultralytics"),
  python_version = "3.12.4"
)

params <- list(
  image_path = "E:/Stomata/Sugarbeet_stomata_imaging/sugarbeet_all_mips/V1T1R1_Ab_nd2_-_V1T1R1_Ab_nd2_(series_1)_MIP.tif",
  model_path = "E:/Stomata/Sugarbeet_stomata_imaging/beetmip_model2/beetmip_model2.pt",
  tile_size  = 128,
  overlap    = 64,
  iou_thresh = 0.5,
  min_area   = 1000,
  max_area   = 3000,
  min_circ   = 0.5,
  max_circ   = 1.0,
  alpha      = 0.4
)

initialise_yolo_model <- function(model_path){
  
  py_run_string(sprintf("
import numpy as np
from ultralytics import YOLO

model = YOLO(r'''%s''')

def segment_tile(tile):
    tile = np.asarray(tile)

    if tile.ndim == 3 and tile.shape[0] == 1:
        tile = np.moveaxis(tile, 0, -1)

    if tile.ndim == 2:
        tile = np.stack([tile]*3, axis=-1)
    elif tile.shape[2] == 1:
        tile = np.repeat(tile, 3, axis=2)

    tile = (255 * tile / tile.max()).astype(np.uint8)

    res = model.predict(tile, task='seg', verbose=False)[0]

    polys = []
    if res.masks is not None:
        for p in res.masks.xy:
            polys.append(np.asarray(p))

    return polys
", model_path))
}

load_image <- function(path){
  img <- tiff::readTIFF(path)
  if (length(dim(img)) == 2)
    img <- array(img, dim = c(dim(img), 1))
  img
}

pad_image <- function(img, tile, stride){
  
  d <- dim(img)
  
  if(length(d) == 2){
    img <- array(img, dim = c(d[1], d[2], 1))
    d <- dim(img)
  }
  
  h <- d[1]
  w <- d[2]
  c <- d[3]
  
  H <- ceiling((h - tile) / stride) * stride + tile
  W <- ceiling((w - tile) / stride) * stride + tile
  
  out <- array(0, dim = c(H, W, c))
  out[1:h, 1:w, ] <- img
  
  out
}

generate_tiles <- function(dim, tile, stride){
  
  ys <- seq(1, dim[1] - tile + 1, by = stride)
  xs <- seq(1, dim[2] - tile + 1, by = stride)
  
  expand.grid(
    tile_y = seq_along(ys),
    tile_x = seq_along(xs)
  ) |>
    transform(
      y0 = ys[tile_y],
      x0 = xs[tile_x]
    )
}

clean_polygon <- function(p){
  if (nrow(p) < 3) return(NULL)
  if (!all(p[1,] == p[nrow(p),]))
    p <- rbind(p, p[1,])
  p
}

shift_polygon <- function(p, xoff, yoff){
  p[,1] <- p[,1] + xoff - 1
  p[,2] <- p[,2] + yoff - 1
  p
}

poly_area <- function(p){
  x <- p[,1]; y <- p[,2]
  0.5 * abs(sum(x[-1]*y[-length(y)] - x[-length(x)]*y[-1]))
}

poly_perimeter <- function(p){
  sum(sqrt(rowSums((p[-1,] - p[-nrow(p),])^2)))
}

polygon_metrics <- function(polys){
  
  do.call(rbind, lapply(seq_along(polys), function(i){
    
    p <- polys[[i]]
    
    a   <- poly_area(p)
    per <- poly_perimeter(p)
    circ <- 4*pi*a/(per^2)
    
    data.frame(
      id = i,
      area = a,
      circularity = circ
    )
  }))
}

plot_overlay <- function(img, polys, metrics, alpha){
  
  H <- dim(img)[1]
  W <- dim(img)[2]
  
  img_rgb <- array(rep(img,3), dim = c(H,W,3))
  grob <- rasterGrob(img_rgb, width=unit(1,"npc"), height=unit(1,"npc"))
  
  df <- do.call(rbind, lapply(seq_along(polys), function(i){
    
    p <- polys[[i]]
    tile_label <- paste(metrics$tile_x[i], metrics$tile_y[i], sep = "_")
    
    data.frame(
      x = p[,1],
      y = p[,2],
      id = i,
      fill_col = tile_label
    )
  }))
  
  ggplot(df, aes(x, y, group=id, fill=fill_col)) +
    annotation_custom(grob, 0, W, 0, H) +
    geom_polygon(alpha = alpha, colour = NA) +
    coord_equal() +
    theme_void() +
    scale_y_reverse()+
    theme(legend.position = "none")
}

run_yolo_pipeline <- function(p){
  
  p$stride <- p$tile_size - p$overlap
  
  initialise_yolo_model(p$model_path)
  
  img <- load_image(p$image_path)
  imgp <- pad_image(img, p$tile_size, p$stride)
  
  tiles <- generate_tiles(dim(imgp), p$tile_size, p$stride)
  
  pb <- txtProgressBar(0, nrow(tiles), style=3)
  
  polys <- list()
  
  poly_tile_map <- data.frame(
    id = integer(),
    tile_x = integer(),
    tile_y = integer()
  )
  
  for (i in seq_len(nrow(tiles))){
    
    setTxtProgressBar(pb, i)
    
    t <- tiles[i,]
    
    tile <- imgp[
      t$y0:(t$y0+p$tile_size-1),
      t$x0:(t$x0+p$tile_size-1),
      , drop=FALSE
    ]
    
    raw <- py$segment_tile(tile)
    
    for (p0 in raw){
      
      p1 <- clean_polygon(p0)
      if (is.null(p1)) next
      
      p2 <- shift_polygon(p1, t$x0, t$y0)
      
      new_id <- length(polys) + 1
      polys[[new_id]] <- p2
      
      poly_tile_map <- rbind(
        poly_tile_map,
        data.frame(
          id = new_id,
          tile_x = t$tile_x,
          tile_y = t$tile_y
        )
      )
    }
  }
  
  close(pb)
  
  if (length(polys) == 0) stop("No detections")
  
  metrics <- polygon_metrics(polys)
  metrics <- merge(metrics, poly_tile_map, by = "id")
  
  graph <- plot_overlay(img, polys, metrics, p$alpha)
  
  return(list(
    polygons = polys,
    overlay_plot = graph,
    metrics = metrics,
    tile_map = poly_tile_map
  ))
}

#---- Run ----
output <- run_yolo_pipeline(params)
print(output$overlay_plot)