library(tidyverse)
library(caTools)
library(fs)

# ===============================
# INPUT FOLDERS
# ===============================
image_folder  <- "D:/stomata/November_image_data_8bit_tifs"
label_folder  <- "D:/stomata/semi_auto_annotation_step"

# ===============================
# OUTPUT FOLDERS
# ===============================
out_img_train <- "data/seg/images/train"
out_img_val   <- "data/seg/images/val"
out_lab_train <- "data/seg/labels/train"
out_lab_val   <- "data/seg/labels/val"

dir_create(out_img_train, recurse = TRUE)
dir_create(out_img_val, recurse = TRUE)
dir_create(out_lab_train, recurse = TRUE)
dir_create(out_lab_val, recurse = TRUE)

# ===============================
# READ ALL IMAGES
# ===============================
all_images <- list.files(image_folder, pattern = "\\.tif$", full.names = FALSE)

set.seed(42)
train_split <- caTools::sample.split(all_images, SplitRatio = 0.7)
names(train_split) <- all_images

# ===============================
# COPY IMAGE + MATCHING LABEL
# ===============================
copy_image_and_label <- function(image_name) {
  
  is_training <- train_split[[image_name]]
  
  # full path to image
  img_in <- file.path(image_folder, image_name)
  
  # original label name: *_all_cells.txt
  original_label <- gsub("\\.tif$", "_all_cells.txt", image_name)
  label_in <- file.path(label_folder, original_label)
  
  # NEW desired label name: match image, but .txt
  new_label_name <- gsub("\\.tif$", ".txt", image_name)
  
  # destination paths
  img_out <- if (is_training) file.path(out_img_train, image_name) else file.path(out_img_val, image_name)
  lab_out <- if (is_training) file.path(out_lab_train, new_label_name) else file.path(out_lab_val, new_label_name)
  
  # copy image
  file.copy(img_in, img_out, overwrite = TRUE)
  
  # copy + RENAME label if it exists
  if (file.exists(label_in)) {
    file.copy(label_in, lab_out, overwrite = TRUE)
  } else {
    warning("Label missing for image: ", image_name,
            "\n  Expected original label: ", label_in)
  }
  
  message("Processed: ", image_name, " → ", if(is_training) "train" else "val")
}

# ===============================
# PROCESS ALL IMAGES
# ===============================
for (img in all_images) {
  copy_image_and_label(img)
}

message("✔ All images and labels copied successfully (labels renamed).")

