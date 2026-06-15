library(fs)
library(caTools)

# ===============================
# INPUT FOLDERS
# ===============================
image_folder  <- "D:/stomata/November_images/November_stomata_crops"
label_folder  <- "D:/stomata/November_images/November_sam_multi_guard_annotations"
setwd("D:/stomata/November_images")

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
all_images <- list.files(image_folder, pattern = "\\.png$", full.names = FALSE)

# ===============================
# HELPER: check label has both class 0 and 1
# ===============================
has_both_classes <- function(label_path) {
  if (!file.exists(label_path)) return(FALSE)
  lines <- readLines(label_path, warn = FALSE)
  if (length(lines) == 0) return(FALSE)
  classes <- sapply(strsplit(lines, "\\s+"), function(x) as.numeric(x[1]))
  return(all(c(0,1) %in% classes))
}

# ===============================
# FILTER IMAGES: only those with both classes
# ===============================
valid_images <- all_images[vapply(all_images, function(img_name) {
  label_file <- file.path(label_folder, gsub("\\.png$", ".txt", img_name))
  has_both_classes(label_file)
}, logical(1))]

message(length(valid_images), " images with both classes found.")

# ===============================
# SPLIT TRAIN / VAL
# ===============================
set.seed(42)
train_split <- caTools::sample.split(1:length(valid_images), SplitRatio = 0.7)
names(train_split) <- valid_images

# ===============================
# COPY IMAGE + LABEL
# ===============================
copy_image_and_label <- function(img_name) {
  
  is_train <- train_split[[img_name]]
  
  img_in <- file.path(image_folder, img_name)
  label_in <- file.path(label_folder, gsub("\\.png$", ".txt", img_name))
  label_out_name <- gsub("\\.png$", ".txt", img_name)
  
  img_out <- if (is_train) file.path(out_img_train, img_name) else file.path(out_img_val, img_name)
  lab_out <- if (is_train) file.path(out_lab_train, label_out_name) else file.path(out_lab_val, label_out_name)
  
  file.copy(img_in, img_out, overwrite = TRUE)
  file.copy(label_in, lab_out, overwrite = TRUE)
  
  message("Copied: ", img_name, " → ", if(is_train) "train" else "val")
}

# ===============================
# PROCESS ALL VALID IMAGES
# ===============================
for (img in valid_images) {
  copy_image_and_label(img)
}

message("✔ All images and labels with both classes copied successfully.")
