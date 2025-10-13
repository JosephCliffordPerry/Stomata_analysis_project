library(reticulate)
library(dplyr)
library(grid)

Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy","opencv-python","matplotlib","scikit-image","ultralytics"),
  python_version = "3.12.4"
)

py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
from skimage.segmentation import find_boundaries
from ultralytics import YOLO

def segment_and_yolo_with_advanced_ws(image_path, model_path):
    # --- Load image ---
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==2:
        gray = img.copy()
    elif img.shape[2]==1:
        gray = img[:,:,0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to 8-bit if needed
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)
    
    # --- Advanced preprocessing for watershed ---
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)
    sharpened = cv2.addWeighted(blur, 1.5, cv2.GaussianBlur(blur, (0,0), 3), -0.5, 0)
    
    kernel = np.ones((3,3), np.uint8)
    gradient = cv2.morphologyEx(sharpened, cv2.MORPH_GRADIENT, kernel)
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    tophat = cv2.morphologyEx(sharpened, cv2.MORPH_TOPHAT, tophat_kernel)
    combined = cv2.bitwise_or(gradient, tophat)
    
    otsu_val, _ = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = int(0.7 * otsu_val)
    _, binary = cv2.threshold(combined, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = markers.astype(np.int32)
    cv2.watershed(cv2.merge([gray,gray,gray]), markers)
    
    # --- Cell properties ---
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
        stomata_areas.append((x_max-x_min)*(y_max-y_min))
    
    # --- Overlay visualization ---
    boundaries = find_boundaries(markers, mode='outer').astype(np.uint8)*255
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thick_boundaries = cv2.dilate(boundaries, thick_kernel, iterations=1)
    
    img_rgb = cv2.merge([gray,gray,gray])
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


# -------------------------------
# Run the Python function in R
# -------------------------------
image_path <- "D:/stomata/training_tifs_8bit//A_T2R3_Ab_15X_frame_0000.tif"
model_path <- "stomata_test1.pt"

stats <- py$segment_yolo_stats_full_rgb_for_yolo(image_path, model_path)

# Access results in R
cells_areas <- stats$cells_areas
stomata_areas <- stats$stomata_areas
composite_img <- stats$composite_image  # Can display using imshow via Python or save via OpenCV

library(grid)

# Retrieve the composite image from Python
img_array <- stats$composite_image  # NumPy array (H x W x 3)

# Convert NumPy array to R array
img_r <- py_to_r(img_array) / 255  # scale to 0-1 for grid.raster

# Render image
grid.raster(img_r)
