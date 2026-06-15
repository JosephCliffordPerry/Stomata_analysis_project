library(dplyr)
library(purrr)
library(jpeg)
library(png)
library(tools)

# -------------------------------
# Paths
# -------------------------------
stomata_segmentation <- readRDS("E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct/dual_masks_overlay/cell_components_dataframe_final.rds")
image_dir <- "E:/Stomata_maize/all_images/test_images/Training_auto_annotation_20pct"
label_dir <- "20pctlabels"
image_out_dir <- "20pctimages"
overlay_dir <- "E:/Stomata_maize/all_images/test_images/filtered_annotations"

dir.create(label_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(image_out_dir, showWarnings = FALSE, recursive = TRUE)

df <- stomata_segmentation

# -------------------------------
# Keep only images with required objects
# -------------------------------
required_objects <- c("Complex","Companion2_consensus","Companion1_consensus")

valid_images <- df %>%
  distinct(image, object) %>%
  group_by(image) %>%
  summarise(objects = list(object), .groups="drop") %>%
  filter(all(required_objects %in% unlist(objects))) %>%
  pull(image)

df <- df %>% filter(image %in% valid_images)

# -------------------------------
# Match overlay files
# -------------------------------
files <- list.files(overlay_dir, pattern = "_multipanel_overlay\\.png$", full.names = FALSE)
file_ids <- sub("_multipanel_overlay\\.png$", "", files)

df <- df %>% filter(image %in% file_ids)

# -------------------------------
# Assign class IDs
# -------------------------------
df$class_id <- dplyr::case_when(
  grepl("complex", df$object, ignore.case = TRUE) ~ 0,
  grepl("companion.*1", df$object, ignore.case = TRUE) ~ 1,
  grepl("companion.*2", df$object, ignore.case = TRUE) ~ 2,
  TRUE ~ NA_real_
)

df <- df %>% filter(!is.na(class_id))

# -------------------------------
# Helper: read image dimensions
# -------------------------------
get_image_size <- function(path){
  ext <- tolower(file_ext(path))
  
  if(ext %in% c("jpg","jpeg")){
    img <- jpeg::readJPEG(path)
  } else if(ext == "png"){
    img <- png::readPNG(path)
  } else {
    stop("Unsupported image format")
  }
  
  list(width = dim(img)[2], height = dim(img)[1])
}

# -------------------------------
# Process each image
# -------------------------------
images <- unique(df$image)

for(img_name in images){
  
  message("Processing: ", img_name)
  
  img_path <- list.files(
    image_dir,
    pattern = paste0("^", img_name),
    full.names = TRUE
  )[1]
  
  if(is.na(img_path)) next
  
  size <- get_image_size(img_path)
  w <- size$width
  h <- size$height
  
  img_df <- df %>% filter(image == img_name)
  
  objs <- split(img_df, interaction(img_df$object, img_df$image))
  
  label_lines <- list()
  
  for(i in seq_along(objs)){
    
    obj <- objs[[i]]
    
    pts <- cbind(obj$col, obj$row)
    
    if(nrow(pts) < 3) next
    
    hull_idx <- chull(pts)
    hull <- pts[hull_idx, ]
    
    x_norm <- hull[,1] / w
    y_norm <- hull[,2] / h
    
    coords <- as.vector(rbind(x_norm, y_norm))
    
    class_id <- obj$class_id[1]
    
    line <- paste(
      class_id,
      paste(sprintf("%.6f", coords), collapse = " ")
    )
    
    label_lines[[length(label_lines) + 1]] <- line
  }
  
  if(length(label_lines) == 0) next
  
  # write label
  label_file <- file.path(
    label_dir,
    paste0(file_path_sans_ext(basename(img_path)), ".txt")
  )
  writeLines(unlist(label_lines), label_file)
  
  # copy image
  file.copy(
    from = img_path,
    to = file.path(image_out_dir, basename(img_path)),
    overwrite = TRUE
  )
}