source("Stomatal_decomposition_direct_inference/align_polygons.R")
reverse_alignment <- function(shape, transform){
  
  pts <- shape
  
  # remove closure if present
  if(all(pts[1,] == pts[nrow(pts),]))
    pts <- pts[-nrow(pts),]
  
  # inverse transform
  pts <- pts %*% transform$A_inv
  
  # restore reflection
  if(transform$reflected)
    pts[,1] <- -pts[,1]
  
  # restore translation
  pts <- sweep(pts, 2, transform$centre, "+")
  
  pts <- rbind(pts, pts[1,])
  
  sf::st_sfc(sf::st_polygon(list(pts)))
}


overlay_refined_shape <- function(image_path, polygon){
  
  img <- png::readPNG(image_path)
  
  h <- dim(img)[1]
  w <- dim(img)[2]
  
  xy <- sf::st_coordinates(polygon)[,1:2]
  
  # flip y axis (image coordinates)
  xy[,2] <- h - xy[,2]
  
  plot(
    c(0, w),
    c(h, 0),
    type = "n",
    asp = 1,
    xlab = "",
    ylab = ""
  )
  
  rasterImage(img, 0, h, w, 0)
  
  lines(xy, col = "red", lwd = 2)
}

aligned<-align_polygons(polys)
aligned_poly <- aligned$geometry[[13]]
transform <- aligned$transform[[13]]


restored <- reverse_alignment(
  sf::st_coordinates(aligned_poly)[,1:2],
  transform
)


orig <- sf::st_coordinates(polys$geometry[[13]])[,1:2]

plot(orig, type = "l", col = "black", asp = 1)
lines(sf::st_coordinates(restored)[,1:2], col = "red")


max(abs(orig - sf::st_coordinates(restored)[,1:2]))
overlay_refined_shape(image_path = paste0("E:/Stomata_maize/all_images/all_images/crops/",aligned$image[13],".png"),polygon = restored)
