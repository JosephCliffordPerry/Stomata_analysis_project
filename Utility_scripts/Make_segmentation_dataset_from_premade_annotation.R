library(tidyverse)
library(caTools)
library(fs)

# ===============================
# INPUT FOLDERS
# ===============================
image_folder  <- "D:/Stomatal_analysis_project/beetmip_yolo_dataset/images"
label_folder  <- "D:/Stomatal_analysis_project/beetmip_yolo_dataset/labels"

# ===============================
# OUTPUT FOLDERS
# ===============================
dataset_name<- "beetmip"
out_img_train <- paste0(dataset_name,"/seg/images/train")
out_img_val   <- paste0(dataset_name,"/seg/images/val")
out_lab_train <- paste0(dataset_name,"/seg/labels/train")
out_lab_val   <- paste0(dataset_name,"/seg/labels/val")

dir_create(out_img_train, recurse = TRUE)
dir_create(out_img_val, recurse = TRUE)
dir_create(out_lab_train, recurse = TRUE)
dir_create(out_lab_val, recurse = TRUE)

# ===============================
# READ ALL IMAGES
# ===============================
file_extension<-"\\.jpg$"
all_images <- list.files(image_folder, pattern = file_extension, full.names = FALSE)

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
  
  # original label name: eg *_all_cells.txt
  original_label <- gsub(file_extension, ".txt", image_name)
  label_in <- file.path(label_folder, original_label)
  
  # NEW desired label name: match image, but .txt
  new_label_name <- gsub(file_extension, ".txt", image_name)
  
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

