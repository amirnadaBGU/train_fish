import cv2
import numpy as np
import os
import albumentations as A

# ==========================================
# GLOBAL SETTINGS
# ==========================================
# Options for FLIP_MODE: "horizontal", "vertical", "180", "none"
FLIP_MODE = "none"

# Options for TASK_TYPE: "segmentation" (polygons) or "detection" (bounding boxes)
TASK_TYPE = "detection"

# Options for TURBID: True or False
TURBID = False

# Paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(script_dir, 'test images')
output_base_dir = os.path.join(script_dir, 'test images 2')


# ==========================================
# TRANSFORMATION LOGIC
# ==========================================
def get_transforms(mode, is_turbid):
    aug_list = []

    # 1. Add Flip logic
    if mode == "horizontal":
        aug_list.append(A.HorizontalFlip(p=1.0))
        sign = "horizontal_flip"
    elif mode == "vertical":
        aug_list.append(A.VerticalFlip(p=1.0))
        sign = "vertical_flip"
    elif mode == "180":
        # Double flip equals 180 degrees rotation
        aug_list.append(A.VerticalFlip(p=1.0))
        aug_list.append(A.HorizontalFlip(p=1.0))
        sign = "flip_180"
    elif mode == "none":
        sign = ""

    # 2. Add Turbid logic (Weather/Water effects)
    if is_turbid:
        aug_list.extend([
            A.RGBShift(r_shift_limit=(20, 40), g_shift_limit=(-20, 20), b_shift_limit=(-40, -10), p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-50, -20), val_shift_limit=(-50, -20), p=1.0),
            A.GaussianBlur(blur_limit=(35, 101), p=0.5),
            A.OneOf([
                A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(0.001, 0.015),
                                hole_width_range=(0.001, 0.015), fill=(255, 255, 255), p=0.95),
                A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(0.001, 0.015),
                                hole_width_range=(0.001, 0.015), fill=(180, 180, 180), p=0.95),
                A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(0.001, 0.015),
                                hole_width_range=(0.001, 0.015), fill=(100, 90, 70), p=0.95),
            ], p=1.0),
        ])
        sign += "_turbid"

    return A.Compose(aug_list), sign


transform_v2, SIGN = get_transforms(FLIP_MODE, TURBID)

subdirs = ['train', 'test', 'valid']
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

images_processed = 0
labels_updated = 0

for subdir in subdirs:
    in_images_dir = os.path.join(input_base_dir, subdir, 'images')
    in_labels_dir = os.path.join(input_base_dir, subdir, 'labels')

    if not os.path.exists(in_images_dir):
        print(f"Skipping {subdir}: 'images' directory not found.")
        continue

    print(f"\n" + "=" * 50)
    print(f"Processing directory: {subdir} | Mode: {FLIP_MODE} | Task: {TASK_TYPE} | Turbid: {TURBID}")
    print("=" * 50)

    out_images_dir = os.path.join(output_base_dir, subdir, 'images')
    out_labels_dir = os.path.join(output_base_dir, subdir, 'labels')

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    for filename in os.listdir(in_images_dir):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        input_file_path = os.path.join(in_images_dir, filename)
        name, ext = os.path.splitext(filename)

        new_filename = f"{name}_{SIGN}{ext}"
        new_txt_filename = f"{name}_{SIGN}.txt"

        output_file_path = os.path.join(out_images_dir, new_filename)
        output_txt_path = os.path.join(out_labels_dir, new_txt_filename)

        # 1. Process Image
        image = cv2.imread(input_file_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform_v2(image=image_rgb)
            final_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_file_path, final_image)
            images_processed += 1
        else:
            print(f"[!] Warning: Could not read image {filename}")
            continue

        # 2. Process Labels based on Task Type
        txt_filename = f"{name}.txt"
        txt_file_path = os.path.join(in_labels_dir, txt_filename)

        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = parts[0]
                coords = [float(val) for val in parts[1:]]
                new_coords = []

                if TASK_TYPE == "segmentation":
                    # Segmentation logic: Flip pairs of x, y coordinates
                    for i in range(len(coords)):
                        val = coords[i]
                        if FLIP_MODE == "horizontal" and i % 2 == 0:
                            val = 1.0 - val
                        elif FLIP_MODE == "vertical" and i % 2 != 0:
                            val = 1.0 - val
                        elif FLIP_MODE == "180":
                            val = 1.0 - val
                        if FLIP_MODE != "none":
                            val = max(0.0, min(val, 1.0))
                            new_coords.append(val)

                elif TASK_TYPE == "detection":
                    # Object Detection logic: format is [x_center, y_center, width, height]
                    if len(coords) != 4:
                        print(f"[!] Warning: Invalid detection format in {txt_filename}. Skipping line.")
                        continue

                    x_c, y_c, w, h = coords

                    if FLIP_MODE == "horizontal":
                        x_c = 1.0 - x_c
                    elif FLIP_MODE == "vertical":
                        y_c = 1.0 - y_c
                    elif FLIP_MODE == "180":
                        x_c = 1.0 - x_c
                        y_c = 1.0 - y_c

                    # Bounds check only for centers (width and height remain the same)
                    if FLIP_MODE != "none":
                        x_c = max(0.0, min(x_c, 1.0))
                        y_c = max(0.0, min(y_c, 1.0))

                        new_coords = [x_c, y_c, w, h]

                # Reconstruct the line
                if FLIP_MODE != "none":
                    coords_str = " ".join([f"{v:.6f}" for v in new_coords])
                    new_lines.append(f"{class_id} {coords_str}\n")

            with open(output_txt_path, 'w') as file:
                file.writelines(new_lines)
            labels_updated += 1
        else:
            print(f"[!] Note: No label file (txt) found for {filename}")

print("\n" + "=" * 50)
print(f"Done! Processed {images_processed} images and updated {labels_updated} YOLO {TASK_TYPE} label files.")
print(f"Final Sign suffix used: {SIGN}")