library(reticulate)
Sys.setenv(RETICULATE_PYTHON = "managed")

reticulate::py_require(
  packages = c("numpy","opencv-python","matplotlib","scikit-image","scipy","ultralytics"),
  python_version = "3.12.4"
)

py_run_string("
import cv2
import numpy as np
from skimage import color, measure, morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -------- helpers --------
def _peaks_to_mask(peaks, shape):
    \"\"\"Accept either coordinate-array peaks (N,2) or boolean mask (shape).
       Return a boolean mask of peaks with given shape.\"\"\"
    if peaks is None:
        return np.zeros(shape, dtype=bool)
    peaks = np.asarray(peaks)
    # boolean mask already
    if peaks.dtype == bool and peaks.shape == shape:
        return peaks.astype(bool)
    # coordinate list (N,2)
    if peaks.ndim == 2 and peaks.shape[1] == 2:
        mask = np.zeros(shape, dtype=bool)
        for coord in peaks:
            # ensure integer coords and inside bounds
            r = int(coord[0]); c = int(coord[1])
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
        return mask
    # fallback: empty mask
    return np.zeros(shape, dtype=bool)

# -----------------------------
# Safe grayscale loader
def load_grayscale(filename, invert=False):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Could not read image: {filename}')
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)
    if len(img.shape) == 2:
        gray = img.copy()
        rgb = cv2.merge([gray, gray, gray])
    elif len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f'Unexpected image shape: {img.shape}')
    if invert:
        gray = cv2.bitwise_not(gray)
    return gray, rgb

# -----------------------------
# Active Contour (with post-split)
def seg_active_contour(filename, invert=False):
    gray, _ = load_grayscale(filename, invert=invert)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)
    from skimage.filters import threshold_local
    adaptive = threshold_local(blur, 21, offset=-0.01)
    mask0 = blur > adaptive
    img_norm = blur.astype(np.float32)/255.0
    mask = segmentation.morphological_chan_vese(img_norm, num_iter=150, init_level_set=mask0, smoothing=3)
    # ensure boolean for remove_small_objects
    mask = morphology.remove_small_objects(mask > 0, min_size=30)

    # Post-split merged cells using distance watershed
    dist = ndi.distance_transform_edt(mask)
    raw_peaks = peak_local_max(dist, labels=mask, min_distance=5, footprint=np.ones((3,3)))
    local_max_mask = _peaks_to_mask(raw_peaks, mask.shape)
    # if no peaks found, fall back to simple labeling of mask
    if np.count_nonzero(local_max_mask) == 0:
        labeled = measure.label(mask)
        return labeled
    markers = measure.label(local_max_mask)
    # watershed to split
    split_mask = segmentation.watershed(-dist, markers, mask=mask)
    labeled = measure.label(split_mask)
    return labeled

# -----------------------------
# Random Walker
def seg_random_walker(filename, invert=False):
    gray, _ = load_grayscale(filename, invert=invert)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7,7), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    cells_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    dist = ndi.distance_transform_edt(cells_clean)
    raw_peaks = peak_local_max(dist, labels=cells_clean, min_distance=5, footprint=np.ones((3,3)))
    local_max_mask = _peaks_to_mask(raw_peaks, cells_clean.shape)
    seeds = np.zeros_like(cells_clean, dtype=bool)
    # if peaks exist fill seeds
    if np.count_nonzero(local_max_mask) > 0:
        seeds[local_max_mask] = True
    # background seeds
    bg_mask = cv2.erode(cv2.bitwise_not(cells_clean), kernel, iterations=2) > 0
    markers = np.zeros_like(cells_clean, dtype=np.int32)
    markers[bg_mask] = 1
    markers[seeds] = 2
    # if markers invalid (e.g., no seeds), fallback to simple threshold-label
    if np.count_nonzero(markers==2) == 0:
        labeled = measure.label(cells_clean > 0)
        return labeled
    img_norm = cv2.normalize(blur.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    rw = segmentation.random_walker(img_norm, markers, beta=150, mode='bf')
    mask = (rw == 2)
    mask = morphology.remove_small_objects(mask > 0, min_size=30)
    labeled = measure.label(mask)
    return labeled

# -----------------------------
# Watershed
def seg_watershed(filename, invert=False):
    gray, rgb = load_grayscale(filename, invert=invert)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    dist = ndi.distance_transform_edt(binary)
    raw_peaks = peak_local_max(dist, labels=binary, min_distance=4, footprint=np.ones((3,3)))
    local_max_mask = _peaks_to_mask(raw_peaks, binary.shape)
    # ensure at least one marker
    if np.count_nonzero(local_max_mask) == 0:
        labeled = measure.label(binary > 0)
        return labeled
    markers = measure.label(local_max_mask)
    _, sure_fg = cv2.threshold(dist, int(0.35*dist.max()), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, np.ones((3,3),np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # use markers for watershed (skimage watershed)
    split = segmentation.watershed(-dist, markers, mask=binary)
    labeled = measure.label(split)
    return labeled

# -----------------------------
# Consensus
def consensus_segmentation(seg_list):
    bin_masks = [(s > 0) for s in seg_list]
    stacked = np.stack(bin_masks, axis=0)
    vote_sum = np.sum(stacked, axis=0)
    consensus = vote_sum >= 2
    labeled = measure.label(consensus)
    return labeled

# -----------------------------
# YOLO overlay (separate)
def yolo_overlay(image_path, yolo_model_path, invert=False):
    gray, rgb = load_grayscale(image_path, invert=invert)
    model = YOLO(yolo_model_path)
    results = model.predict(source=rgb, task='obb', save=False, verbose=False)[0]
    # make copy and draw OBBs (if present)
    overlay = rgb.copy()
    try:
        obb_list = results.obb.xyxyxyxy
        import cv2
        for i in range(len(obb_list)):
            obb = obb_list[i].cpu().numpy()
            pts = obb.reshape(-1,1,2).astype(np.int32)
            overlay = cv2.polylines(overlay, [pts], isClosed=True, color=(255,0,0), thickness=2)
    except Exception:
        # if no obb or attribute missing, just return original rgb
        pass
    return overlay

# -----------------------------
# Visualization (with YOLO panel)
def visualize_consensus(image_path, seg1, seg2, seg3, consensus, overlay):
    gray, rgb = load_grayscale(image_path)
    fig, axs = plt.subplots(2,3, figsize=(18,10))
    axs[0,0].imshow(color.label2rgb(seg1, bg_label=0)); axs[0,0].set_title('Active Contour')
    axs[0,1].imshow(color.label2rgb(seg2, bg_label=0)); axs[0,1].set_title('Random Walker')
    axs[0,2].imshow(color.label2rgb(seg3, bg_label=0)); axs[0,2].set_title('Watershed')
    axs[1,0].imshow(color.label2rgb(consensus, bg_label=0, image=rgb)); axs[1,0].set_title('Consensus')
    axs[1,1].imshow(overlay); axs[1,1].set_title('YOLO Overlay')
    axs[1,2].imshow(rgb); axs[1,2].set_title('Original')
    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(); plt.show()
")

# -----------------------------
# R-side wrapper (example run)
image_path <- "D:/stomata/training_tifs_8bit/B1A_T2R5_Ab_15X_frame_0000.tif"
yolo_model_path <- "stomata_test1.pt"

# Run each segmentation (use invert=TRUE if your cells are dark on light background)
seg1 <- py$seg_active_contour(image_path, invert=TRUE)
seg2 <- py$seg_random_walker(image_path, invert=FALSE)
seg3 <- py$seg_watershed(image_path, invert=FALSE)

# Consensus (majority)
consensus <- py$consensus_segmentation(list(seg1, seg2, seg3))

# YOLO overlay kept separate
overlay <- py$yolo_overlay(image_path, yolo_model_path, invert=FALSE)

# Visualize all panels
py$visualize_consensus(image_path, seg1, seg2, seg3, consensus, overlay)
