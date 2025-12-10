py_run_string("
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure, morphology
from skimage.filters import threshold_local

def segment_cells_initial_mask_filtered(filename='plant_cells_20x.png',
                                        min_cell_area=100,
                                        max_cell_area=5000,
                                        invert=False,
                                        block_size_adaptive=25,
                                        edge_weight=1.0,
                                        erosion_radius=2):
    # 1️⃣ Load image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)

    # 2️⃣ CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)

    # 3️⃣ Gaussian blur
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    # 4️⃣ Sobel edges
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5️⃣ Combine blurred + edges
    combined = cv2.addWeighted(blur, 0.8, sobel_mag, edge_weight, 0)

    # 6️⃣ Adaptive threshold
    adaptive_thresh = threshold_local(combined, int(block_size_adaptive), offset=-0.02)
    init_mask = combined > adaptive_thresh

    # 7️⃣ Morphological erosion (cleanup)
    selem = morphology.disk(int(erosion_radius))
    eroded_mask = morphology.erosion(init_mask, selem)

    # 8️⃣ Label eroded mask (before filtering)
    labeled_before = measure.label(eroded_mask)
    labeled_before_rgb = color.label2rgb(labeled_before, bg_label=0, bg_color=(0,0,0))

    # 9️⃣ Filter by area
    props = measure.regionprops_table(labeled_before, properties=['label','area'])
    for i, area in enumerate(props['area']):
        if area < min_cell_area or area > max_cell_area:
            labeled_before[labeled_before == props['label'][i]] = 0

    labeled_after = measure.label(labeled_before > 0)
    labeled_after_rgb = color.label2rgb(labeled_after, bg_label=0, bg_color=(0,0,0))

    # 🔟 Overlay on original image
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = (0.7 * img_rgb / 255.0 + 0.6 * labeled_after_rgb).clip(0, 1)

    # 1️⃣1️⃣ Show results
    fig, axs = plt.subplots(2,3, figsize=(16,9))
    axs[0,0].imshow(gray, cmap='gray'); axs[0,0].set_title('Original')
    axs[0,1].imshow(init_mask, cmap='gray'); axs[0,1].set_title('Initial Mask')
    axs[0,2].imshow(eroded_mask, cmap='gray'); axs[0,2].set_title('After Erosion')
    axs[1,0].imshow(labeled_before_rgb); axs[1,0].set_title('Labeled Before Filtering')
    axs[1,1].imshow(labeled_after_rgb); axs[1,1].set_title('Labeled After Filtering')
    axs[1,2].imshow(overlay); axs[1,2].set_title('Overlay on Original')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()

    print(f'Total objects before filtering: {len(np.unique(measure.label(eroded_mask))) - 1}')
    print(f'Objects after filtering: {len(np.unique(labeled_after)) - 1}')

    return labeled_after_rgb
")

segment_cells_initial <- function(filename,
                                  min_cell_area=100,
                                  max_cell_area=5000,
                                  invert=FALSE,
                                  block_size_adaptive=25,
                                  edge_weight=1.0,
                                  erosion_radius=2) {
  py$segment_cells_initial_mask_filtered(
    filename = r_to_py(as.character(filename)),
    min_cell_area = r_to_py(as.integer(min_cell_area)),
    max_cell_area = r_to_py(as.integer(max_cell_area)),
    invert = r_to_py(as.logical(invert)),
    block_size_adaptive = r_to_py(as.integer(block_size_adaptive)),
    edge_weight = r_to_py(as.numeric(edge_weight)),
    erosion_radius = r_to_py(as.integer(erosion_radius))
  )
}


seg_initial <- segment_cells_initial(
  filename = "D:/stomata/training_tifs_8bit/A_T2R3_Ab_15X_frame_0001.tif",
  min_cell_area = 50,
  max_cell_area = 5000,
  invert = FALSE,
  block_size_adaptive = 25,
  edge_weight = 1.0,
  erosion_radius = 2
)

