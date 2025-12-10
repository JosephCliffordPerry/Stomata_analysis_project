library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy", "opencv-python", "matplotlib", "scikit-image", "scipy", "imageio-ffmpeg", "ultralytics"),
  python_version = "3.12.4"
)

py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure, morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.filters import threshold_local

def segment_cells_active_contour_edge_refined(filename='plant_cells_20x.png',
                                              min_cell_area=100,
                                              max_cell_area=5000,
                                              iterations=300,
                                              invert=False,
                                              block_size_adaptive=25,
                                              edge_weight=1.0,
                                              min_distance_peaks=5,
                                              outline_thickness=2):
    # 1️⃣ Load image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)

    # 2️⃣ CLAHE enhancement + blur
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)
    blur = enhanced

    # 3️⃣ Sobel edge map
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined = cv2.addWeighted(blur, 0.8, sobel_mag, edge_weight, 0)

    # 4️⃣ Adaptive threshold
    adaptive_thresh = threshold_local(combined, int(block_size_adaptive), offset=-0.02)
    init_mask = combined > adaptive_thresh

    # 5️⃣ Distance transform & seeds
    dist = ndi.distance_transform_edt(init_mask)
    local_max = peak_local_max(dist, min_distance=int(min_distance_peaks), labels=init_mask)
    seeds = np.zeros_like(init_mask, dtype=bool)
    if local_max.size > 0:
        seeds[tuple(local_max.T)] = True
    init_mask = np.logical_or(init_mask, seeds)

    # 6️⃣ Active contour segmentation
    img_norm = combined.astype(np.float32)/255.0
    mask = segmentation.morphological_chan_vese(
        img_norm, num_iter=int(iterations), init_level_set=init_mask, smoothing=2
    )

    # 7️⃣ Remove small objects and watershed refine
    mask_clean = morphology.remove_small_objects(mask, min_size=int(min_cell_area*0.1))
    dist = ndi.distance_transform_edt(mask_clean)
    local_max = peak_local_max(dist, min_distance=5, labels=mask_clean)
    markers = np.zeros_like(mask_clean, dtype=np.int32)
    if local_max.size > 0:
        markers[tuple(local_max.T)] = np.arange(1, local_max.shape[0]+1)
    else:
        markers = ndi.label(mask_clean)[0]
    mask_ws = segmentation.watershed(-dist, markers, mask=mask_clean)

    # 8️⃣ Area filtering (preserve labels)
    labeled = measure.label(mask_ws)
    props = measure.regionprops_table(labeled, properties=['label','area'])
    mask_keep = np.ones_like(labeled, dtype=bool)
    for i, area in enumerate(props['area']):
        if area < (min_cell_area * 0.005) or area > (max_cell_area * 2.0):
            mask_keep[labeled == props['label'][i]] = False
    labeled[~mask_keep] = 0

    # Color map
    labeled_rgb = color.label2rgb(labeled, bg_label=0, bg_color=(0,0,0))

    # 9️⃣ Generate boundary overlay (preserve colors)
    gray_norm = gray.astype(np.float32) / 255.0
    darkened = np.power(gray_norm, 1.5)
    overlay = np.stack([darkened]*3, axis=-1)

    # Find and thicken boundaries
    boundaries = segmentation.find_boundaries(labeled, mode='outer')
    if outline_thickness > 1:
        kernel = morphology.disk(int(outline_thickness))
        boundaries = morphology.dilation(boundaries, kernel)

    # Paint boundaries using the same colors as the labeled RGB map
    overlay_with_outlines = overlay.copy()
    overlay_with_outlines[boundaries] = labeled_rgb[boundaries]

    # 🔟 Display
    fig, axs = plt.subplots(2,3, figsize=(15,10))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original Grayscale')
    axs[0,1].imshow(blur, cmap='gray'); axs[0,1].set_title('Blur + CLAHE')
    axs[0,2].imshow(sobel_mag, cmap='gray'); axs[0,2].set_title('Sobel Edges')
    axs[1,0].imshow(init_mask, cmap='gray'); axs[1,0].set_title('Initial Mask')
    axs[1,1].imshow(labeled_rgb); axs[1,1].set_title('Final Labeled Cells')
    axs[1,2].imshow(overlay_with_outlines); axs[1,2].set_title('Colored Cell Boundaries Overlay (Thicker)')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()

    print(f'Detected {len(np.unique(labeled)) - 1} cells.')
    return labeled_rgb, overlay_with_outlines
")




# ✅ Proper R wrapper
segment_cells <- function(filename,
                          min_cell_area=100,
                          max_cell_area=5000,
                          iterations=300,
                          invert=FALSE,
                          block_size_adaptive=25,
                          edge_weight=1.0,
                          min_distance_peaks=5) {
  py$segment_cells_active_contour_edge_refined(
    filename = r_to_py(as.character(filename)),
    min_cell_area = r_to_py(as.integer(min_cell_area)),
    max_cell_area = r_to_py(as.integer(max_cell_area)),
    iterations = r_to_py(as.integer(iterations)),
    invert = r_to_py(as.logical(invert)),
    block_size_adaptive = r_to_py(as.integer(block_size_adaptive)),
    edge_weight = r_to_py(as.numeric(edge_weight)),
    min_distance_peaks = r_to_py(as.integer(min_distance_peaks))
  )
}

# 🔍 Run segmentation
seg_result <- segment_cells(
  filename = "D:/stomata/training_tifs_8bit/A_T2R3_Ab_15X_frame_0001.tif",
  min_cell_area = 1,
  max_cell_area = 5000,
  iterations = 300,
  invert = TRUE,
  block_size_adaptive = 25,
  edge_weight = 1.0,
  min_distance_peaks = 3
)
