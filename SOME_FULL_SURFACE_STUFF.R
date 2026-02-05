library(reticulate)
library(tiff)
library(polyclip)
library(ggplot2)
library(grid)
Sys.setenv(RETICULATE_PYTHON = "managed")
reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","ultralytics"), 
  python_version = "3.12.4"
)
params <- list(
  image_path = "E:/Stomata/Sugarbeet_stomata_imaging/mips2/V1T2R2-Ab-mip.nd2 - V1T2R2-Ab-mip.nd2 (series 1)_MIP.tif",
  model_path = "E:/Stomata/Sugarbeet_stomata_imaging/beetmip_model2/beetmip_model2.pt",
  tile_size  = 128,
  overlap    = 96,
  iou_thresh = 0.5,
  min_area   = 10,
  max_area   = 5000,
  min_circ   = 0.2,
  max_circ   = 1.0,
  alpha      = 0.4
)

params$stride <- params$tile_size - params$overlap
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
", params$model_path))


load_image <- function(path){
  img <- tiff::readTIFF(path)
  if (length(dim(img)) == 2)
    img <- array(img, dim = c(dim(img), 1))
  img
}

pad_image <- function(img, tile, stride){
  h <- dim(img)[1]; w <- dim(img)[2]; c <- dim(img)[3]
  H <- ceiling((h - tile) / stride) * stride + tile
  W <- ceiling((w - tile) / stride) * stride + tile
  out <- array(0, dim = c(H, W, c))
  out[1:h, 1:w, ] <- img
  out
}

generate_tiles <- function(dim, tile, stride){
  expand.grid(
    y0 = seq(1, dim[1] - tile + 1, by = stride),
    x0 = seq(1, dim[2] - tile + 1, by = stride)
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

polygon_iou <- function(a, b){
  pa <- as_polyclip(a)
  pb <- as_polyclip(b)
  
  inter <- polyclip(pa, pb, op = "intersection")
  if (length(inter) == 0) return(0)
  
  inter_area <- sum(vapply(inter, function(pp){
    poly_area(cbind(pp$x, pp$y))
  }, numeric(1)))
  
  union_area <- poly_area(a) + poly_area(b) - inter_area
  inter_area / union_area
}


bbox <- function(p){
  c(min(p[,1]), min(p[,2]), max(p[,1]), max(p[,2]))
}

overlap_bbox <- function(a, b){
  !(a[3] < b[1] || a[1] > b[3] || a[4] < b[2] || a[2] > b[4])
}
as_polyclip <- function(p){
  list(list(x = p[,1], y = p[,2]))
}

deduplicate_polygons <- function(polys, iou_thresh){
  n <- length(polys)
  if (n <= 1) return(polys)
  
  boxes <- lapply(polys, bbox)
  keep <- rep(TRUE, n)
  
  for (i in seq_len(n - 1)){
    if (!keep[i]) next
    for (j in (i+1):n){
      if (!keep[j]) next
      if (!overlap_bbox(boxes[[i]], boxes[[j]])) next
      if (polygon_iou(polys[[i]], polys[[j]]) >= iou_thresh){
        u <- polyclip(
          as_polyclip(polys[[i]]),
          as_polyclip(polys[[j]]),
          op = "union"
        )
        
        # keep largest resulting polygon
        areas <- vapply(u, function(pp){
          poly_area(cbind(pp$x, pp$y))
        }, numeric(1))
        
        polys[[i]] <- cbind(u[[which.max(areas)]]$x,
                            u[[which.max(areas)]]$y)
        
        keep[j] <- FALSE
      }
    }
  }
  
  polys[keep]
}

filter_polygons <- function(polys, min_area, max_area, min_circ, max_circ){
  Filter(function(p){
    a <- poly_area(p)
    per <- poly_perimeter(p)
    circ <- 4*pi*a/(per^2)
    
    a >= min_area && a <= max_area &&
      circ >= min_circ && circ <= max_circ
  }, polys)
}

plot_overlay <- function(img, polys, alpha){
  H <- dim(img)[1]; W <- dim(img)[2]
  img_rgb <- array(rep(img,3), dim = c(H,W,3))
  grob <- rasterGrob(img_rgb, width=unit(1,"npc"), height=unit(1,"npc"))
  
  df <- do.call(rbind, lapply(seq_along(polys), function(i){
    p <- polys[[i]]
    data.frame(x=p[,1], y=p[,2], id=i)
  }))
  
  ggplot(df, aes(x,y,group=id)) +
    annotation_custom(grob, 0,W,0,H) +
    geom_polygon(fill="red", color="black", alpha=alpha) +
    coord_equal() + 
    theme_void()+
    scale_y_reverse()
  
  
}
run_yolo_pipeline <- function(p){
  
  img <- load_image(p$image_path)
  imgp <- pad_image(img, p$tile_size, p$stride)
  tiles <- generate_tiles(dim(imgp), p$tile_size, p$stride)
  
  pb <- txtProgressBar(0, nrow(tiles), style=3)
  polys <- list()
   
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
      polys[[length(polys)+1]] <- shift_polygon(p1, t$x0, t$y0)
    }
  }
  close(pb)
  
  if (length(polys) == 0) stop("No detections")
  
  polys <- deduplicate_polygons(polys, p$iou_thresh)
  polys <- filter_polygons(
    polys, p$min_area, p$max_area, p$min_circ, p$max_circ
  )
  
  plot_overlay(img, polys, p$alpha)
}
run_yolo_pipeline(params)




