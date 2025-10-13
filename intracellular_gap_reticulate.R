library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image","imageio-ffmpeg", "ultralytics", "numpy"), 
  python_version = "3.12.4"
)

py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure

def segment_cells_fixed_threshold(filename='plant_cells_20x.png'):
    # 1️⃣ Load image and grayscale
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2️⃣ CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    
    # 3️⃣ Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    
    # 4️⃣ Threshold gaps using fixed value 220
    _, gaps_binary = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
    
    # 5️⃣ Morphological openingand closing to clean small noise
    kernel = np.ones((3,3), np.uint8)
    gaps_binary = cv2.morphologyEx(gaps_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    gaps_closed = cv2.morphologyEx(gaps_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    
    # 6️⃣ Invert to get cells as foreground
    cells_binary = cv2.bitwise_not(gaps_binary)
    
    # 7️⃣ Distance transform for watershed markers
    dist = cv2.distanceTransform(cells_binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(cells_binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 8️⃣ Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 9️⃣ Watershed segmentation
    markers = cv2.watershed(img, markers)
    img_ws = img.copy()
    img_ws[markers == -1] = [255, 0, 0]  # boundaries in red
    
    # 🔟 Labeled image for visualization
    label_image = color.label2rgb(markers, bg_label=1, bg_color=(0,0,0))
    
    # 1️⃣1️⃣ Visualization
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original Grayscale')
    axs[0,1].imshow(blur, cmap='gray'); axs[0,1].set_title('Blur + CLAHE')
    axs[0,2].imshow(gaps_binary, cmap='gray'); axs[0,2].set_title('Gaps Binary Mask (Threshold=220)')
    axs[1,0].imshow(cells_binary, cmap='gray'); axs[1,0].set_title('Cells Binary Mask')
    axs[1,1].imshow(cv2.cvtColor(img_ws, cv2.COLOR_BGR2RGB)); axs[1,1].set_title('Watershed Boundaries')
    axs[1,2].imshow(label_image); axs[1,2].set_title('Final Labeled Cells')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 1️⃣2️⃣ Region properties
    props = measure.regionprops_table(markers, properties=['label', 'area'])
    print(f'Detected {len(props['label'])} cells.')
    
    return label_image
")

# --- Run the segmentation ---
seg_result <- py$segment_cells_fixed_threshold("D:/stomata/Just testing with Leica-ATC2000 - 20X/Ha-T1R1-U-20X.jpg")
