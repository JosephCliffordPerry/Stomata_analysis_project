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
from skimage import color, measure, morphology, segmentation, filters
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def segment_cells_random_walker(filename='plant_cells_20x.png',
                                min_cell_area=100,
                                max_cell_area=5000,
                                beta=90,
                                invert=False):
    # 1️⃣ Load image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2️⃣ Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    
    # 3️⃣ Denoise
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    
    # 4️⃣ Initial binary estimate (adaptive threshold)
    binary0 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    
    # Optional inversion (if cells appear dark on bright background)
    binary = cv2.bitwise_not(binary0) if not invert else binary0
    
    # 5️⃣ Clean binary mask
    kernel = np.ones((3,3), np.uint8)
    cells_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cells_clean = cv2.morphologyEx(cells_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 6️⃣ Distance transform for seed generation
    dist = cv2.distanceTransform(cells_clean, cv2.DIST_L2, 5)
    local_max = peak_local_max(dist, min_distance=10, labels=cells_clean)
    seed_fg = np.zeros_like(cells_clean, dtype=bool)
    seed_fg[tuple(local_max.T)] = True
    
    # Background seeds: eroded inverse
    seed_bg = cv2.erode(cv2.bitwise_not(cells_clean), np.ones((7,7), np.uint8), iterations=2)
    
    # 7️⃣ Combine seeds for random walker
    markers = np.zeros_like(cells_clean, dtype=np.int32)
    markers[seed_bg > 0] = 1      # background
    markers[seed_fg > 0] = 2      # foreground
    
    # 8️⃣ Normalize image for Random Walker
    img_norm = cv2.normalize(blur.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    
    # 9️⃣ Random Walker segmentation
    rw_labels = segmentation.random_walker(img_norm, markers, beta=beta, mode='bf')
    
    # 1️⃣0️⃣ Clean up segmentation result
    mask = (rw_labels == 2).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 1️⃣1️⃣ Label and filter regions by size
    labeled = measure.label(mask)
    props = measure.regionprops_table(labeled, properties=['label', 'area'])
    for i, area in enumerate(props['area']):
        if area < min_cell_area or area > max_cell_area:
            labeled[labeled == props['label'][i]] = 0
    
    labeled_rgb = color.label2rgb(labeled, bg_label=0, bg_color=(0,0,0))
    
    # 1️⃣2️⃣ Visualization
    fig, axs = plt.subplots(2,3, figsize=(15,10))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original Grayscale')
    axs[0,1].imshow(blur, cmap='gray'); axs[0,1].set_title('Blur + CLAHE')
    axs[0,2].imshow(cells_clean, cmap='gray'); axs[0,2].set_title('Clean Binary Mask')
    axs[1,0].imshow(dist, cmap='magma'); axs[1,0].set_title('Distance Transform')
    axs[1,1].imshow(mask, cmap='gray'); axs[1,1].set_title('Random Walker Mask')
    axs[1,2].imshow(labeled_rgb); axs[1,2].set_title('Final Labeled Cells')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()
    
    print(f'Detected {len(np.unique(labeled)) - 1} cells.')
    
    return labeled_rgb
")


# --- Run the updated segmentation ---
seg_result <- py$segment_cells_random_walker("D:/stomata/training_tifs_8bit/A_T2R3_Ab_15X_frame_0001.tif")
