df <- detections_data
V1 <- names(detections)
b <- cbind(V1 = V1, df)
# Fix column names
colnames(b) <- c("image", "x1","y1","x2","y2","x3","y3","x4","y4")

write_yolo_obb_scaled <- function(df, image_col = "image",
                                  out_dir = "labels_scaled",
                                  class_id = 0,
                                  img_width,
                                  img_height) {
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # clean base name
  df$base_name <- sub("\\.jpg.*$", "", df[[image_col]], ignore.case = TRUE)
  
  groups <- split(df, df$base_name)
  
  for (img in names(groups)) {
    g <- groups[[img]]
    
    # ensure numeric
    coords <- as.data.frame(lapply(g[, c("x1","y1","x2","y2","x3","y3","x4","y4")], as.numeric))
    
    # scale coordinates to 0-1 using the constant image size
    coords[, c("x1","x2","x3","x4")] <- coords[, c("x1","x2","x3","x4")] / img_width
    coords[, c("y1","y2","y3","y4")] <- coords[, c("y1","y2","y3","y4")] / img_height
    
    # construct lines
    lines <- apply(coords, 1, function(r) {
      paste(c(class_id, r), collapse = " ")
    })
    
    # write file
    writeLines(lines, file.path(out_dir, paste0(img, ".txt")))
  }
}
# all images are 1920x1080
write_yolo_obb_scaled(
  b,
  image_col = "image",
  out_dir = "labels_scaled",
  class_id = 0,
  img_width = 2596,
  img_height = 1944
)


# function to clean filenames
folder_path <- "C:/Users/jp19193/Downloads/all_images"

# get only filenames
all_files <- list.files(folder_path)

clean_name_safe <- function(f) {
  # remove extension
  ext <- tools::file_ext(f)
  name <- sub(paste0("\\.", ext, "$"), "", f, ignore.case = TRUE)
  
  # replace any non-alphanumeric with _
  name <- gsub("[^A-Za-z0-9]", "_", name)
  
  # collapse multiple underscores
  name <- gsub("_+", "_", name)
  
  # remove leading/trailing underscores
  name <- gsub("^_|_$", "", name)
  
  paste0(name, ".", ext)
}

# rename files
for (f in all_files) {
  old_path <- file.path(folder_path, f)
  new_path <- file.path(folder_path, clean_name_safe(f))
  
  if (old_path != new_path) file.rename(old_path, new_path)
}

list.files(folder_path)








