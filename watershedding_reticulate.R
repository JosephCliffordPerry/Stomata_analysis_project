library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy","opencv-python","ultralytics","scikit-image"),
  python_version = "3.12.4"
)

# -----------------------------
# 1️⃣ Watershed segmentation (Python)
# -----------------------------
py_run_string("
import cv2
import numpy as np
from skimage import measure

def watershed_cells(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Safe grayscale handling
    if len(img.shape) == 2:
        gray = img.copy()
        img_rgb = cv2.merge([gray, gray, gray])
    elif img.shape[2] == 1:
        gray = img[:,:,0]
        img_rgb = cv2.merge([gray, gray, gray])
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CLAHE + blur
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    # Threshold + morphological closing
    kernel = np.ones((3,3), np.uint8)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform + markers
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = markers.astype(np.int32)
    cv2.watershed(img_rgb, markers)

    # Cell areas
    regions = measure.regionprops_table(markers, properties=['label','area'])
    cells_areas = np.array(regions['area']).tolist()

    # Overlay for visualization
    overlay = img_rgb.copy()
    overlay[markers == -1] = [0,255,0]  # watershed boundaries in green

    return {'markers': markers, 'cells_areas': cells_areas, 'overlay': overlay}
")

# -----------------------------
# 2️⃣ YOLO inference (Python)
# -----------------------------
py_run_string("
from ultralytics import YOLO
import numpy as np
import cv2

def yolo_detect(image_rgb, yolo_model_path):
    # Ensure image is RGB
    if image_rgb.dtype != np.uint8:
        image_rgb = cv2.convertScaleAbs(image_rgb)
    model = YOLO(yolo_model_path)
    results = model.predict(source=image_rgb, task='obb', save=False, verbose=False)[0]

    stomata_areas = []
    obb_list = results.obb.xyxyxyxy
    for i in range(len(obb_list)):
        obb = obb_list[i].cpu().numpy()
        x_min, y_min = obb[:,0].min(), obb[:,1].min()
        x_max, y_max = obb[:,0].max(), obb[:,1].max()
        stomata_areas.append((x_max - x_min) * (y_max - y_min))

    return {'obb_list': obb_list, 'stomata_areas': stomata_areas}
")

# -----------------------------
# 3️⃣ R: Run both functions and combine
# -----------------------------
image_path <- "D:/stomata/training_tifs_8bit/A_T2R3_Ab_15X_frame_0000.tif"
model_path <- "stomata_test1.pt"

# Watershed segmentation
ws_results <- py$watershed_cells(image_path)

# YOLO detection
yolo_results <- py$yolo_detect(ws_results$overlay, model_path)

# Overlay YOLO boxes on watershed overlay
library(abind)
img_composite <- ws_results$overlay
obb_list <- yolo_results$obb_list
for(i in seq_along(obb_list)){
  obb <- obb_list[[i]]$cpu()$numpy()
  pts <- matrix(as.integer(obb), ncol=2, byrow=TRUE)
  pts <- array(pts, dim=c(nrow(pts),1,2))
  img_composite <- py$cv2$polylines(img_composite, [pts], isClosed=TRUE, color=c(255,0,0), thickness=2L)
}

# Collect stats
cells_areas <- ws_results$cells_areas
stomata_areas <- yolo_results$stomata_areas

# Display in R
library(grid)
grid::grid.raster(img_composite/255)  # scale to [0,1] for plotting
