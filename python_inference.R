library(reticulate)
library(dplyr)

Sys.setenv(RETICULATE_PYTHON = "managed")
reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","ultralytics"), 
  python_version = "3.12.4"
)

# -------------------------------------------
# 1️⃣ Embedded Python function
# -------------------------------------------
py_run_string("
import cv2
import numpy as np
from skimage.segmentation import find_boundaries
from skimage import measure
from ultralytics import YOLO

def segment_yolo_stats_full(image_path, model_path):
    # Load image (grayscale-safe)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if len(img.shape) == 2:
        gray = img.copy()
        img_rgb = cv2.merge([gray, gray, gray])
    elif img.shape[2] == 1:
        gray = img[:,:,0]
        img_rgb = cv2.merge([gray, gray, gray])
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = img.copy()
    
    # Watershed segmentation
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    _, gaps_binary = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    gaps_binary = cv2.morphologyEx(gaps_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    gaps_closed = cv2.morphologyEx(gaps_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    cells_binary = cv2.bitwise_not(gaps_binary).astype(np.uint8)
    
    dist = cv2.distanceTransform(cells_binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(cells_binary, kernel_close, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img_rgb, markers)
    
    # Region properties for all cells
    regions = measure.regionprops_table(markers, properties=['label','area'])
    cells_areas = np.array(regions['area']).tolist()
    
    # YOLO inference for stomata (3-channel)
    model = YOLO(model_path)
    results = model.predict(source=img_rgb, task='obb', save=False, verbose=False)[0]
    
    stomata_areas = []
    for obb_tensor in results.obb.xyxyxyxy:
        obb = obb_tensor.cpu().numpy()
        x_min, y_min = obb[:,0].min(), obb[:,1].min()
        x_max, y_max = obb[:,0].max(), obb[:,1].max()
        area = (x_max - x_min) * (y_max - y_min)
        stomata_areas.append(area)
    
    return {'cells_areas': cells_areas, 'stomata_areas': stomata_areas}
")

# -------------------------------------------
# 2️⃣ Run function and retrieve data in R
# -------------------------------------------
image_path <- "D:/stomata/training_tifs/A_T2R3_Ab_15X_frame_0000.tif"
model_path <- "stomata_test1.pt"

stats <- py$segment_yolo_stats_full(image_path, model_path)

cells_df <- data.frame(area = stats$cells_areas)
stomata_df <- data.frame(area = stats$stomata_areas)

# -------------------------------------------
# 3️⃣ Remove outliers using 2×IQR
# -------------------------------------------
remove_outliers <- function(x){
  q <- quantile(x, probs=c(0.25,0.75), na.rm=TRUE)
  iqr <- q[2] - q[1]
  x[x >= (q[1]-2*iqr) & x <= (q[2]+2*iqr)]
}

cells_df <- cells_df %>% mutate(area = remove_outliers(area))
stomata_df <- stomata_df %>% mutate(area = remove_outliers(area))

# Remove any NA rows caused by filtering
cells_df <- cells_df %>% filter(!is.na(area))
stomata_df <- stomata_df %>% filter(!is.na(area))

# -------------------------------------------
# 4️⃣ Calculate stomatal index, mean areas, ratio
# -------------------------------------------
num_stomata <- nrow(stomata_df)
num_cells <- nrow(cells_df)
stomatal_index <- num_stomata / (num_stomata + num_cells)

mean_stomatal_area <- mean(stomata_df$area)
mean_epidermal_area <- mean(cells_df$area)

area_ratio <- mean_stomatal_area / mean_epidermal_area

# -------------------------------------------
# 5️⃣ Output summary
# -------------------------------------------
summary_df <- data.frame(
  num_stomata = num_stomata,
  num_epidermal_cells = num_cells,
  stomatal_index = stomatal_index,
  mean_stomatal_area = mean_stomatal_area,
  mean_epidermal_area = mean_epidermal_area,
  area_ratio = area_ratio
)

summary_df
