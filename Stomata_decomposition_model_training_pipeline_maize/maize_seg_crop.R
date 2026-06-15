# =========================================================
# MAIZE COMPLEX CROP PIPELINE
# =========================================================

library(reticulate)
library(tools)
library(imager)
library(sf)
library(dplyr)
library(ggplot2)
library(progress)

# =========================================================
# 1️⃣ YOLO SEGMENTATION INFERENCE
# =========================================================

yolo_seg_inference <- function(image_dir, model_path) {
  
  Sys.setenv(RETICULATE_PYTHON = "managed")
  
  reticulate::py_require(
    packages = c(
      "ultralytics",
      "opencv-python",
      "numpy"
    ),
    python_version = "3.12.4"
  )
  
  np <- import("numpy", convert = FALSE)
  cv2 <- import("cv2", convert = FALSE)
  ultralytics <- import("ultralytics")
  
  model <- ultralytics$YOLO(model_path)
  
  image_paths <- list.files(
    image_dir,
    pattern = "\\.(jpg|jpeg|png|tif|tiff)$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  all_results <- list()
  
  # =====================================================
  # PROGRESS BAR
  # =====================================================
  
  pb <- progress_bar$new(
    format = "YOLO inference [:bar] :percent eta: :eta",
    total = length(image_paths),
    clear = FALSE,
    width = 80
  )
  
  for (img_path in image_paths) {
    
    pb$tick()
    
    img <- cv2$imread(img_path, cv2$IMREAD_UNCHANGED)
    
    if (is.null(img)) {
      message("⚠️ Failed to read image: ", basename(img_path))
      next
    }
    
    # Ensure 3-channel image
    if (length(dim(img)) == 2L ||
        (length(dim(img)) == 3L && dim(img)[3] == 1L)) {
      
      img <- cv2$cvtColor(img, cv2$COLOR_GRAY2BGR)
      
    } else if (length(dim(img)) == 3L && dim(img)[3] == 4L) {
      
      img <- cv2$cvtColor(img, cv2$COLOR_BGRA2BGR)
    }
    
    results <- tryCatch({
      
      model$predict(
        source = img,
        task = "segment",
        save = FALSE,
        verbose = FALSE
      )[[1]]
      
    }, error = function(e) {
      
      message(
        "⚠️ Inference failed: ",
        basename(img_path),
        " | ",
        e$message
      )
      
      return(NULL)
    })
    
    if (is.null(results)) next
    if (is.null(results$masks)) next
    
    polys <- results$masks$xy
    
    if (length(polys) == 0) next
    
    image_preds <- list()
    
    for (i in seq_along(polys)) {
      
      p <- py_to_r(polys[[i]])
      
      if (nrow(p) < 3) next
      
      image_preds[[length(image_preds) + 1]] <- p
    }
    
    all_results[[basename(img_path)]] <- image_preds
  }
  
  return(all_results)
}

# =========================================================
# 2️⃣ SPLIT POLYGON ALONG LONGEST AXIS
# =========================================================

split_polygon_long_axis <- function(poly_coords) {
  
  pca <- prcomp(poly_coords, center = TRUE)
  
  center <- colMeans(poly_coords)
  
  axis_vec <- pca$rotation[,1]
  
  proj <- as.matrix(
    poly_coords -
      matrix(
        center,
        nrow(poly_coords),
        2,
        byrow = TRUE
      )
  ) %*% axis_vec
  
  side1 <- poly_coords[proj >= 0, , drop = FALSE]
  side2 <- poly_coords[proj < 0, , drop = FALSE]
  
  list(
    side1 = side1,
    side2 = side2,
    center = center,
    axis = axis_vec
  )
}

# =========================================================
# 3️⃣ EXPAND POLYGON OUTLINE
# =========================================================

expand_polygon <- function(coords, scale_factor = 1.10) {
  
  center <- colMeans(coords)
  
  expanded <- sweep(coords, 2, center, "-")
  expanded <- expanded * scale_factor
  expanded <- sweep(expanded, 2, center, "+")
  
  expanded
}

# =========================================================
# 4️⃣ ROTATED CROP FROM POLYGON
# =========================================================

crop_polygon_region <- function(
    img,
    coords,
    out_file
) {
  
  coords <- expand_polygon(coords, 1.10)
  
  pca <- prcomp(coords, center = TRUE)
  
  center <- colMeans(coords)
  
  axis_vec <- pca$rotation[,1]
  
  angle <- -atan2(axis_vec[2], axis_vec[1]) * 180 / pi
  
  # =====================================================
  # ROTATE IMAGE
  # =====================================================
  
  rotated <- imrotate(
    img,
    angle = angle,
    cx = center[1],
    cy = center[2],
    interpolation = 1
  )
  
  # =====================================================
  # ROTATE POLYGON COORDS
  # =====================================================
  
  theta <- angle * pi / 180
  
  rot_mat <- matrix(
    c(
      cos(theta), -sin(theta),
      sin(theta),  cos(theta)
    ),
    nrow = 2,
    byrow = TRUE
  )
  
  shifted <- sweep(coords, 2, center, "-")
  
  rotated_pts <- t(rot_mat %*% t(shifted))
  
  rotated_pts <- sweep(rotated_pts, 2, center, "+")
  
  xmin <- floor(min(rotated_pts[,1]))
  xmax <- ceiling(max(rotated_pts[,1]))
  ymin <- floor(min(rotated_pts[,2]))
  ymax <- ceiling(max(rotated_pts[,2]))
  
  xmin <- max(1, xmin)
  ymin <- max(1, ymin)
  
  xmax <- min(dim(rotated)[2], xmax)
  ymax <- min(dim(rotated)[1], ymax)
  
  cropped <- imsub(
    rotated,
    x %in% xmin:xmax,
    y %in% ymin:ymax
  )
  
  save.image(cropped, out_file)
}

# =========================================================
# 5️⃣ MAIN CROP PIPELINE
# =========================================================

crop_split_polygons <- function(
    detections,
    image_dir,
    output_dir
) {
  
  suppressPackageStartupMessages({
    library(imager)
    library(dplyr)
  })
  
  # =====================================================
  # OUTPUT FOLDERS
  # =====================================================
  
  crop_dir <- file.path(output_dir, "split_polygon_crops")
  overlay_dir <- file.path(output_dir, "debug_overlay")
  
  dir.create(
    crop_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  dir.create(
    overlay_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  metadata <- list()
  
  # =====================================================
  # PROGRESS BAR
  # =====================================================
  
  pb <- progress_bar$new(
    format = "Cropping [:bar] :percent eta: :eta",
    total = length(detections),
    clear = FALSE,
    width = 80
  )
  
  # =====================================================
  # LOOP IMAGES
  # =====================================================
  
  for (img_name in names(detections)) {
    
    pb$tick()
    
    img_path <- file.path(image_dir, img_name)
    
    img <- load.image(img_path)
    
    polys <- detections[[img_name]]
    
    # =====================================================
    # CREATE OVERLAY IMAGE
    # =====================================================
    
    overlay_file <- file.path(
      overlay_dir,
      paste0(
        file_path_sans_ext(img_name),
        "_overlay.png"
      )
    )
    
    png(
      filename = overlay_file,
      width = max(512, dim(img)[2]),
      height = max(512, dim(img)[1]),
      units = "px"
    )
    
    par(
      mar = c(0, 0, 0, 0),
      xaxs = "i",
      yaxs = "i"
    )
    
    # =====================================================
    # IMAGE DIMENSIONS
    # =====================================================
    
    img_h <- dim(img)[1]
    img_w <- dim(img)[2]
    
    plot(
      c(0, img_w),
      c(img_h, 0),
      type = "n",
      asp = 1,
      axes = FALSE,
      xlab = "",
      ylab = ""
    )
    
    rasterImage(
      as.raster(img),
      0,
      img_h,
      img_w,
      0
    )
    
    # =====================================================
    # LOOP POLYGONS
    # =====================================================
    
    for (i in seq_along(polys)) {
      
      coords <- polys[[i]]
      
      if (nrow(coords) < 3) next
      
      split_obj <- split_polygon_long_axis(coords)
      
      halves <- list(
        split_obj$side1,
        split_obj$side2
      )
      
      center <- split_obj$center
      axis <- split_obj$axis
      
      # =================================================
      # DRAW POLYGON
      # =================================================
      
      polygon(
        x = coords[,1],
        y = coords[,2],
        border = "lime",
        lwd = 2
      )
      
      # =================================================
      # SPLIT LINE
      # =================================================
      
      proj <- as.matrix(
        coords -
          matrix(
            center,
            nrow(coords),
            2,
            byrow = TRUE
          )
      ) %*% axis
      
      line_len <- max(abs(proj)) * 1.3
      
      p1 <- center - axis * line_len
      p2 <- center + axis * line_len
      
      segments(
        x0 = p1[1],
        y0 = p1[2],
        x1 = p2[1],
        y1 = p2[2],
        col = "red",
        lwd = 3
      )
      
      # =================================================
      # CENTER POINT
      # =================================================
      
      points(
        x = center[1],
        y = center[2],
        col = "blue",
        pch = 16,
        cex = 1.5
      )
    }
    # =====================================================
    # LOOP POLYGONS
    # =====================================================
    
    for (i in seq_along(polys)) {
      
      coords <- polys[[i]]
      
      if (nrow(coords) < 3) next
      
      # =================================================
      # SPLIT POLYGON
      # =================================================
      
      split_obj <- split_polygon_long_axis(coords)
      
      halves <- list(
        split_obj$side1,
        split_obj$side2
      )
      
      center <- split_obj$center
      axis <- split_obj$axis
      
      # =================================================
      # DRAW POLYGON
      # =================================================
      
      polygon(
        x = coords[,1],
        y = coords[,2],
        border = "green",
        lwd = 2
      )
      
      # =================================================
      # DRAW SPLIT LINE
      # =================================================
      
      proj <- as.matrix(
        coords -
          matrix(
            center,
            nrow(coords),
            2,
            byrow = TRUE
          )
      ) %*% axis
      
      line_len <- max(abs(proj)) * 1.3
      
      p1 <- center - axis * line_len
      p2 <- center + axis * line_len
      
      segments(
        x0 = p1[1],
        y0 = p1[2],
        x1 = p2[1],
        y1 = p2[2],
        col = "red",
        lwd = 3
      )
      
      # =================================================
      # DRAW CENTER POINT
      # =================================================
      
      points(
        x = center[1],
        y = center[2],
        col = "blue",
        pch = 16,
        cex = 1.5
      )
      
      # =================================================
      # SAVE SPLIT CROPS
      # =================================================
      
      for (h in seq_along(halves)) {
        
        half_poly <- halves[[h]]
        
        if (nrow(half_poly) < 3) next
        
        out_file <- file.path(
          crop_dir,
          paste0(
            file_path_sans_ext(img_name),
            "_obj_", i,
            "_half_", h,
            ".png"
          )
        )
        
        crop_polygon_region(
          img = img,
          coords = half_poly,
          out_file = out_file
        )
        
        metadata[[length(metadata) + 1]] <- data.frame(
          image = img_name,
          object_id = i,
          half_id = h,
          crop_file = out_file
        )
      }
    }
    
    dev.off()
  }
  
  # =====================================================
  # SAVE METADATA
  # =====================================================
  
  metadata_df <- bind_rows(metadata)
  
  write.csv(
    metadata_df,
    file.path(output_dir, "crop_metadata.csv"),
    row.names = FALSE
  )
  
  return(metadata_df)
}

# =========================================================
# 6️⃣ FULL PIPELINE
# =========================================================

individual_stomata_crop <- function(
    image_dir,
    model_path
) {
  
  message("=== YOLO SEGMENTATION INFERENCE ===")
  
  detections <- yolo_seg_inference(
    image_dir,
    model_path
  )
  
  if (length(detections) == 0) {
    message("No detections found")
    return(NULL)
  }
  
  output_dir <- file.path(
    image_dir,
    "split_polygon_output"
  )
  
  message("=== CROPPING ===")
  
  crops_df <- crop_split_polygons(
    detections,
    image_dir,
    output_dir
  )
  
  message("=== DONE ===")
  
  return(crops_df)
}

# =========================================================
# EXAMPLE
# =========================================================

image_dir <- "E:/Stomata_maize/all_images/all_images/crops/Small_test_data"

model_path <- "D:/stomata/Maize_complex.pt"

crops_df <- individual_stomata_crop(
  image_dir,
  model_path
)