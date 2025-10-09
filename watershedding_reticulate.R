
# =============================================================
# Plant Cell Segmentation in R using Python via reticulate
# =============================================================

# --- Setup Environment ---
# install.packages("reticulate")
library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(packages = c("numpy", "opencv-python", "matplotlib", "scikit-image"), python_version = "3.12.4")
# --- Import Python Modules ---
cv2 <- import("cv2")
np <- import("numpy")
plt <- import("matplotlib.pyplot")
skimage <- import("skimage", convert = TRUE)
measure <- import("skimage.measure")
color <- import("skimage.color")
exposure <- import("skimage.exposure")

# --- Define Python Function (embedded directly in R) ---
py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, color, measure

def segment_cells(filename='plant_cells_20x.png'):
    # 1️⃣ Load & Preprocess
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)

    # 2️⃣ Wall Enhancement
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
edges = cv2.Canny(blur, 50, 150)
combined = cv2.bitwise_or(gradient, edges)  # merge edge info
inverted = cv2.bitwise_not(combined)

    # 3️⃣ Binary Threshold
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4️⃣ Distance Transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5️⃣ Background / Unknown
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6️⃣ Markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7️⃣ Watershed
    markers = cv2.watershed(img, markers)
    img_ws = img.copy()
    img_ws[markers == -1] = [255, 0, 0]

    # 8️⃣ Label Image
    label_image = color.label2rgb(markers, bg_label=1, bg_color=(0,0,0))

    # 9️⃣ Visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original (Grayscale)')
    axs[0,1].imshow(gradient, cmap='gray'); axs[0,1].set_title('Morphological Gradient')
    axs[0,2].imshow(binary, cmap='gray'); axs[0,2].set_title('Binary Mask')
    axs[1,0].imshow(dist_norm, cmap='jet'); axs[1,0].set_title('Distance Transform')
    axs[1,1].imshow(cv2.cvtColor(img_ws, cv2.COLOR_BGR2RGB)); axs[1,1].set_title('Watershed Boundaries')
    axs[1,2].imshow(label_image); axs[1,2].set_title('Final Labeled Cells')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout()
    plt.show()

    props = measure.regionprops_table(markers, properties=['label', 'area'])
    print(f'Detected {len(props['label'])} cells.')
    return label_image
")

# --- Run Segmentation ---
py$segment_cells("D:/stomata/Just testing with Leica-ATC2000 - 20X/Ha-T1R1-U-20X.jpg")
