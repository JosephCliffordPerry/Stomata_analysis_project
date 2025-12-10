library(jsonlite)
library(tidyverse)
library(fs)
library(stringr)
library(caTools)

# Read JSON and extract oriented bounding box data
read.obbox.from.json <- function(file) {
  data <- jsonlite::read_json(file, simplifyVector = FALSE)
 

  process.shape <- function(shape,shape.number) {

    coords <- matrix(unlist(shape$points), ncol = 2, byrow = TRUE)
    df <- as.data.frame(coords)
    colnames(df) <- c("x", "y")

    df %>%
      mutate(
        shape = shape.number,
        file = file,
        folder = dirname(file),
        imageName = str_replace(basename(file), ".json", ".jpg"),
        label = 0
      )
  }
  all_shapes <- lapply(seq_along(data$shapes), function(i) {
    process.shape(data$shapes[[i]], i)
  })
  
  
  all_shapes <- all_shapes[!sapply(all_shapes, is.null)]  # Remove skipped ones

  if (length(all_shapes) == 0) return(NULL)

  bind_rows(all_shapes) %>%
    mutate(imageWidth = data$imageWidth,
           imageHeight = data$imageHeight)
}

# Load all files
files <- list.files(
  path = "D:/stomata/Preliminary",
  pattern = "\\.json$",
  recursive = TRUE,
  full.names = TRUE
)

# Read and process
border.data <- do.call(rbind, lapply(files, read.obbox.from.json))
border.data$label <- as.numeric(as.factor(border.data$label)) - 1  # zero-index class
library(dplyr)
convert.outlines.to.obbox <- function(df) {
  df %>%
    group_by(file, imageName, shape, label, folder, imageWidth, imageHeight) %>%
    group_modify(~ {
      coords <- as.matrix(.x[, c("x", "y")])

      if (nrow(coords) == 4) {
        # already OBB → keep as-is
        return(.x)
      } else if (nrow(coords) < 3 || sd(coords[,1]) == 0 || sd(coords[,2]) == 0) {
        # Degenerate case (line/point) → fallback to axis-aligned box
        xmin <- min(coords[,1]); xmax <- max(coords[,1])
        ymin <- min(coords[,2]); ymax <- max(coords[,2])
        rect_orig <- rbind(
          c(xmin, ymin),
          c(xmax, ymin),
          c(xmax, ymax),
          c(xmin, ymax)
        )
      } else {
        # --- PCA-based OBB ---
        coords_centered <- scale(coords, scale = FALSE)
        pca <- prcomp(coords_centered)

        # Rotate points
        rotated <- coords_centered %*% pca$rotation
        xmin <- min(rotated[,1]); xmax <- max(rotated[,1])
        ymin <- min(rotated[,2]); ymax <- max(rotated[,2])

        rect_rot <- rbind(
          c(xmin, ymin),
          c(xmax, ymin),
          c(xmax, ymax),
          c(xmin, ymax)
        )

        # Rotate back
        rect_orig <- rect_rot %*% t(pca$rotation)
        rect_orig <- sweep(rect_orig, 2, attr(coords_centered, "scaled:center"), "+")
      }

      # Build output dataframe
      obb_df <- as.data.frame(rect_orig)
      colnames(obb_df) <- c("x", "y")
      obb_df <- obb_df %>%
        mutate(
          shape = unique(.x$shape),
          file = unique(.x$file),
          folder = unique(.x$folder),
          imageName = unique(.x$imageName),
          label = unique(.x$label),
          imageWidth = unique(.x$imageWidth),
          imageHeight = unique(.x$imageHeight)
        )
      return(obb_df)
    }) %>%
    ungroup()
}


border.data.obbox <- convert.outlines.to.obbox(border.data)

# Split into training and validation
set.seed(42)
train.images <- caTools::sample.split(unique(border.data$imageName), SplitRatio = 0.7)
names(train.images) <- unique(border.data$imageName)

fs::dir_create("data/obbox/images/train/")
fs::dir_create("data/obbox/images/val/")
fs::dir_create("data/obbox/labels/train/")
fs::dir_create("data/obbox/labels/val/")

write.obbox <- function(source.image.name) {
  # Filter image and normalize coordinates
  img.data <- border.data %>%
    filter(imageName == source.image.name) %>%
    mutate(
      x = x / imageWidth,
      y = y / imageHeight
    )

  # Group shapes and create label lines
  grouped <- img.data %>%
    group_by(shape, label, folder, imageName) %>%
    summarise(
      coords = paste0(round(x, 6), " ", round(y, 6), collapse = " "),
      .groups = "drop"
    ) %>%
    mutate(line = paste(label, coords))

  # Determine training/validation
  isTraining <- train.images[[source.image.name]]
  label.dir <- ifelse(isTraining, "data/obbox/labels/train", "data/obbox/labels/val")
  image.dir <- ifelse(isTraining, "data/obbox/images/train", "data/obbox/images/val")

  folder.name <- basename(grouped$folder[1])
  img.name <- grouped$imageName[1]

  label.file <- file.path(label.dir, paste0(folder.name, "_", str_replace(img.name, ".jpg", ".txt")))
  image.file <- file.path(image.dir, paste0(folder.name, "_", img.name))
  source.image <- file.path(grouped$folder[1], img.name)

  # Write all lines at once
  writeLines(grouped$line, con = label.file)

  # Copy image if not already present
  if (!file.exists(image.file)) file.copy(source.image, image.file)
}


# Apply to all images
sapply(unique(border.data$imageName), write.obbox)

write.obbox(border.data$imageName[1])

