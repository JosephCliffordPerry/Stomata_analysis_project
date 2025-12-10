library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","scipy","imageio-ffmpeg", "ultralytics", "numpy"), 
  python_version = "3.12.4"
)

py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure, morphology
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def segment_cells_distance_transform(filename='plant_cells_20x.png', min_cell_area=100, max_cell_area=5000):
    # 1️⃣ Load image and grayscale
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2️⃣ CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    
   # 3️⃣ Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    
    # 4️⃣ First rough threshold to get an approximate mask
    rough_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    rough_mask_inv = cv2.bitwise_not(rough_mask)
    
    # 5️⃣ Make a blurred “probability” mask of cell regions
    mask_soft = cv2.GaussianBlur(rough_mask_inv, (21,21), 0)
    mask_soft_norm = cv2.normalize(mask_soft, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # 6️⃣ Combine with CLAHE-enhanced image to emphasise structures
    emphasized = cv2.addWeighted(
        enhanced.astype(np.float32)/255.0, 0.7,
        mask_soft_norm.astype(np.float32), 0.3,
        0
    )
    emphasized = np.uint8(emphasized * 255)
    
    # 7️⃣ Re-threshold on the emphasized image for refined mask
    gaps_binary = cv2.adaptiveThreshold(
        emphasized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )

    # Invert to get cells as foreground
    #cells_binary = cv2.bitwise_not(gaps_binary)
    cells_binary = gaps_binary
    # 5️⃣ Remove small blobs inside cells using morphological opening
    kernel = np.ones((3,3), np.uint8)
    cells_clean = cv2.morphologyEx(cells_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 6️⃣ Distance transform
    dist = cv2.distanceTransform(cells_clean, cv2.DIST_L2, 5)
    
    # Smooth to suppress small peaks inside cells
    dist_blur = cv2.GaussianBlur(dist, (7,7), 0)
    
    # Local maxima as markers
    local_max_coords = peak_local_max(
        dist_blur,
        min_distance=15,
        threshold_abs=0.4 * dist_blur.max(),
        labels=cells_clean
    )
    
    local_max_mask = np.zeros_like(dist, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    
    # Combine with eroded mask for stability
    sure_fg = cv2.erode(cells_clean, np.ones((5,5), np.uint8), iterations=2)
    combined_markers = np.logical_or(sure_fg.astype(bool), local_max_mask)
    markers = ndi.label(combined_markers)[0]
    
    # Add 1 for background
    markers = markers + 1
    markers[cells_clean == 0] = 0

    # 7️⃣ Watershed segmentation
    markers_ws = cv2.watershed(img, markers)
    img_ws = img.copy()
    img_ws[markers_ws == -1] = [255, 0, 0]  # boundaries in red
    
    # 8️⃣ Post-process: remove too small or too large regions
    label_props = measure.regionprops_table(markers_ws, properties=['label', 'area'])
    for i, area in enumerate(label_props['area']):
        if area < min_cell_area or area > max_cell_area:
            markers_ws[markers_ws == label_props['label'][i]] = 0
    
    # 9️⃣ Labeled image for visualization
    label_image = color.label2rgb(markers_ws, bg_label=1, bg_color=(0,0,0))
    
    # 🔟 Visualization
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original Grayscale')
    axs[0,1].imshow(blur, cmap='gray'); axs[0,1].set_title('Blur + CLAHE')
    axs[0,2].imshow(cells_clean, cmap='gray'); axs[0,2].set_title('Cells Binary Mask')
    axs[1,0].imshow(dist_blur, cmap='magma'); axs[1,0].set_title('Distance Transform (Smoothed)')
    axs[1,1].imshow(cv2.cvtColor(img_ws, cv2.COLOR_BGR2RGB)); axs[1,1].set_title('Watershed Boundaries')
    axs[1,2].imshow(label_image); axs[1,2].set_title('Final Labeled Cells')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 1️⃣1️⃣ Region properties
    props = measure.regionprops_table(markers_ws, properties=['label', 'area'])
    print('Detected {} cells.'.format(len(props['label'])))
    
    return label_image
")


# --- Run the updated segmentation ---
seg_result <- py$segment_cells_distance_transform("D:/stomata/training_tifs_8bit/A_T2R3_Ab_15X_frame_0001.tif")
