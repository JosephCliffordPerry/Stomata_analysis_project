library(reticulate)
py_require(
  packages = c(
    "numpy",
    "opencv-python"))

py_run_string("
import os
import numpy as np
import cv2

# -----------------------
# Paths
# -----------------------
mask_dir = r'E:/Stomata_maize/batch_cluster_annotate/task_1/overlays'
image_dir = r'E:/Stomata_maize/batch_cluster_annotate/task_1'
output_dir = r'E:/Stomata_maize/batch_cluster_annotate/task_1/graphs'

os.makedirs(output_dir, exist_ok=True)

# -----------------------
# Helper: extract base name
# -----------------------
def get_base_name(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace('_complex_mask', '')
    return name

# -----------------------
# Index images
# -----------------------
image_map = {}
for img_file in os.listdir(image_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        base = os.path.splitext(img_file)[0]
        image_map[base] = os.path.join(image_dir, img_file)

# -----------------------
# Process masks
# -----------------------
for mask_file in os.listdir(mask_dir):
    if not mask_file.endswith('.npy'):
        continue

    mask_path = os.path.join(mask_dir, mask_file)
    base = get_base_name(mask_file)

    if base not in image_map:
        print(f'No match for: {mask_file}')
        continue

    img_path = image_map[base]

    # Load data
    mask = np.load(mask_path)
    image = cv2.imread(img_path)

    if image is None:
        print(f'Failed to load image: {img_path}')
        continue

    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()

    # Normalize mask to 0–255
    mask_norm = (mask > 0).astype(np.uint8) * 255

    # Resize mask if needed
    if mask_norm.shape[:2] != image.shape[:2]:
        mask_norm = cv2.resize(mask_norm, (image.shape[1], image.shape[0]))

    # Create colored overlay (red)
    overlay = image.copy()
    overlay[mask_norm > 0] = [0, 0, 255]

    # Blend
    alpha = 0.4
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Save
    out_path = os.path.join(output_dir, base + '_overlay.png')
    cv2.imwrite(out_path, output)

    print(f'Saved: {out_path}')
")
