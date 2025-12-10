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

def segment_yolo_stats_safe(image_path, model_path):
    # Load image (grayscale-safe)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Convert 16-bit or float images to 8-bit
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Prepare RGB for visualization
    if len(img.shape) == 2:
        gray = img.copy()
        img_rgb = cv2.cvtColor(cv2.merge([img,img,img]), cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 1:
        gray = img[:,:,0]
        img_rgb = cv2.cvtColor(cv2.merge([gray,gray,gray]), cv2.COLOR_BGR2RGB)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # --- Fixed-threshold watershed ---
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    _, gaps_binary = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    gaps_binary = cv2.morphologyEx(gaps_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    gaps_closed = cv2.morphologyEx(gaps_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    cells_binary = cv2.bitwise_not(gaps_binary)
    
    # Ensure distanceTransform input is 8-bit single-channel
    cells_binary_8u = cells_binary.astype(np.uint8)
    
    dist = cv2.distanceTransform(cells_binary_8u, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(cells_binary_8u, kernel_close, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_rgb, markers)
    
    # --- Region properties ---
    regions = measure.regionprops_table(markers, properties=['label','area'])
    cells_areas = np.array(regions['area']).tolist()
    
    # --- YOLO inference ---
    model = YOLO(model_path)
    results = model.predict(source=image_path, task='obb', save=False, verbose=False)[0]
    
    stomata_areas = []
    obb_list = results.obb.xyxyxyxy
    for i in range(len(obb_list)):
        obb = obb_list[i].cpu().numpy()
        x_min, y_min = obb[:,0].min(), obb[:,1].min()
        x_max, y_max = obb[:,0].max(), obb[:,1].max()
        area = (x_max - x_min) * (y_max - y_min)
        stomata_areas.append(area)
    
    # --- Create overlay visualization ---
    boundaries = find_boundaries(markers, mode='outer').astype(np.uint8) * 255
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thick_boundaries = cv2.dilate(boundaries, thick_kernel, iterations=1)
    overlay_ws = img_rgb.copy()
    overlay_ws[thick_boundaries>0] = [0,255,0]
    
    img_composite = overlay_ws.copy()
    for i in range(len(obb_list)):
        obb = obb_list[i].cpu().numpy()
        pts = obb.reshape(-1,2).astype(int)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img_composite, [pts], isClosed=True, color=(255,0,0), thickness=2)
    
    return {'cells_areas': cells_areas, 'stomata_areas': stomata_areas, 'composite_image': img_composite}
")


# -------------------------------------------
# 2️⃣ Run function and retrieve data in R
# -------------------------------------------
stats <- py$segment_yolo_stats(
  "D:/stomata/training_tifs/A_T2R3_Ab_15X_frame_0000.tif",
  "stomata_test1.pt"
)

# Convert to R numeric vectors
cells_areas <- unlist(stats$cells_areas)
stomata_areas <- unlist(stats$stomata_areas)

# -------------------------------------------
# 3️⃣ Remove outliers using 2*IQR
# -------------------------------------------
remove_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR_val <- Q3 - Q1
  x[x >= (Q1 - 2*IQR_val) & x <= (Q3 + 2*IQR_val)]
}

cells_areas_clean <- remove_outliers(cells_areas)
stomata_areas_clean <- remove_outliers(stomata_areas)

# -------------------------------------------
# 4️⃣ Calculate stomatal index, mean areas, ratios
# -------------------------------------------
num_stomata <- length(stomata_areas_clean)
num_cells <- length(cells_areas_clean)
stomatal_index <- num_stomata / (num_stomata + num_cells)
mean_stomatal_area <- mean(stomata_areas_clean)
mean_cell_area <- mean(cells_areas_clean)
area_ratio <- mean_stomatal_area / mean_cell_area

# -------------------------------------------
# 5️⃣ Output summary dataframe
# -------------------------------------------
summary_df <- data.frame(
  num_stomata = num_stomata,
  num_epidermal_cells = num_cells,
  stomatal_index = stomatal_index,
  mean_stomatal_area = mean_stomatal_area,
  mean_epidermal_area = mean_cell_area,
  area_ratio = area_ratio
)

print(summary_df)

# -------------------------------------------
# 6️⃣ Optional: display composite image
# -------------------------------------------
library(imager)
composite_image <- stats$composite_image
plot(as.cimg(composite_image))
