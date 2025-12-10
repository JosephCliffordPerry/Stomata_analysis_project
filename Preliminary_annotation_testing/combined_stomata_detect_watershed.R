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

# -----------------------------
# Safe grayscale loader
def load_grayscale(filename, invert=False):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Could not load {filename}')
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)
    if len(img.shape) == 2:
        gray = img.copy()
        if invert:
            gray = cv2.bitwise_not(gray)
        rgb = cv2.merge([gray, gray, gray])
    elif len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if invert:
            gray = cv2.bitwise_not(gray)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if invert:
            gray = cv2.bitwise_not(gray)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f'Unexpected image shape: {img.shape}')
    return gray, rgb

# -----------------------------
# Active Contour Segmentation
def seg_active_contour(filename, invert=False, min_area=50, iterations=200,
                       clahe_clip=4.0, clahe_grid=16, blur_ksize=(5,5),
                       adaptive_block=25, adaptive_offset=-0.02, sobel_weight=1.0):
    gray, _ = load_grayscale(filename, invert=invert)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, blur_ksize, 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined = cv2.addWeighted(blur, 0.8, sobel_mag, sobel_weight, 0)
    from skimage.filters import threshold_local
    adaptive = threshold_local(combined, adaptive_block, offset=adaptive_offset)
    mask0 = combined > adaptive
    img_norm = combined.astype(np.float32)/255.0
    mask = segmentation.morphological_chan_vese(img_norm, num_iter=int(iterations),
                                                init_level_set=mask0, smoothing=2)
    mask = morphology.remove_small_objects(mask > 0, min_size=min_area)
    labeled = measure.label(mask)
    return labeled

# -----------------------------
# Random Walker Segmentation
def seg_random_walker(filename, invert=False, min_area=50, beta=150,
                      blur_ksize=(5,5), dist_min_distance=5, morph_kernel=(3,3)):
    gray, _ = load_grayscale(filename, invert=invert)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, blur_ksize, 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones(morph_kernel, np.uint8)
    cells_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    dist = cv2.distanceTransform(cells_clean, cv2.DIST_L2, 5)
    local_max = peak_local_max(dist, labels=cells_clean, min_distance=dist_min_distance,
                               footprint=np.ones((3,3)))
    seeds = np.zeros_like(cells_clean, dtype=bool)
    if local_max.shape[0] > 0:
        for coord in local_max:
            if coord[0] < seeds.shape[0] and coord[1] < seeds.shape[1]:
                seeds[tuple(coord)] = True
    bg = cv2.erode(cv2.bitwise_not(cells_clean), kernel, iterations=2)
    markers = np.zeros_like(cells_clean, dtype=np.int32)
    markers[bg>0] = 1
    markers[seeds] = 2
    img_norm = cv2.normalize(blur.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    rw = segmentation.random_walker(img_norm, markers, beta=beta, mode='bf')
    mask = (rw==2)
    mask = morphology.remove_small_objects(mask > 0, min_size=min_area)
    labeled = measure.label(mask)
    return labeled

# -----------------------------
# Watershed Segmentation
def seg_watershed(filename, invert=False, min_area=50, clahe_clip=4.0,
                  clahe_grid=16, blur_ksize=(3,3), dist_thresh=0.3, dilate_iter=2):
    gray, rgb = load_grayscale(filename, invert=invert)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, blur_ksize, 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_thresh*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(binary, np.ones((3,3),np.uint8), iterations=dilate_iter)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255]=0
    cv2.watershed(rgb, markers)
    labeled = np.copy(markers)
    labeled[labeled<=1]=0
    return labeled

# -----------------------------
# Consensus (majority voting)
def consensus_segmentation(seg_list):
    bin_masks = [s>0 for s in seg_list]
    stacked = np.stack(bin_masks, axis=0)
    vote_sum = np.sum(stacked, axis=0)
    consensus = vote_sum >= (len(seg_list)//2 + 1)
    labeled = measure.label(consensus)
    return labeled

# -----------------------------
# YOLO overlay (handles OBB or normal boxes)
def yolo_overlay(image_path, yolo_model_path, invert=False, conf_thresh=0.25):
    gray, rgb = load_grayscale(image_path, invert=invert)
    model = YOLO(yolo_model_path)
    results = model.predict(source=rgb, save=False, conf=conf_thresh, verbose=False)[0]

    overlay = rgb.copy()

    if hasattr(results, 'obb') and getattr(results.obb, 'xyxyxyxy', None) is not None:
        try:
            obb_list = results.obb.xyxyxyxy.cpu().numpy()
            for obb in obb_list:
                pts = obb.reshape(-1,1,2).astype(np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(255,0,0), thickness=2)
        except Exception as e:
            print('OBB draw failed:', e)
    elif hasattr(results, 'boxes') and getattr(results.boxes, 'xyxy', None) is not None:
        try:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else [None]*len(boxes)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                color = (0,255,0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                if confs[i] is not None:
                    cv2.putText(overlay, f'{confs[i]:.2f}', (x1, max(y1-5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception as e:
            print('Box draw failed:', e)
    else:
        print('⚠️ No OBB or bounding boxes found in YOLO results.')

    return overlay

# -----------------------------
# Visualization (adds YOLO + original panels)
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
# R side execution
image_path <- "D:/stomata/training_tifs_8bit/MO_T2R6_Ab_frame_0001.tif"
yolo_model_path <- "Stomata_obbox.pt"

seg1 <- py$seg_active_contour(image_path, invert=TRUE)
seg2 <- py$seg_random_walker(image_path, invert=TRUE)
seg3 <- py$seg_watershed(image_path, invert=TRUE)

consensus <- py$consensus_segmentation(list(seg1, seg2, seg3))
overlay <- py$yolo_overlay(image_path, yolo_model_path, invert=FALSE)

py$visualize_consensus(image_path, seg1, seg2, seg3, consensus, overlay)
