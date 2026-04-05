import cv2
import numpy as np
import os
import shutil

# --- הגדרות נתיבים ---
script_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(script_dir, 'test images')
output_base_dir = os.path.join(script_dir, 'test images 2')

subdirs = ['train', 'test', 'valid']
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')


def apply_filters(image):
    """פונקציה המרכזת את העיבוד הגרפי"""
    # Apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Sharpening
    clahed_then_blurred = cv2.GaussianBlur(clahed, (55, 55), 0)
    sharpened = cv2.addWeighted(clahed, 1.8, clahed_then_blurred, -0.8, 0)
    return sharpened


# --- לולאת העיבוד המרכזית ---
for subdir in subdirs:
    # הגדרת נתיבי מקור ויעד
    in_img_dir = os.path.join(input_base_dir, subdir, 'images')
    in_lbl_dir = os.path.join(input_base_dir, subdir, 'labels')

    out_img_dir = os.path.join(output_base_dir, subdir, 'images')
    out_lbl_dir = os.path.join(output_base_dir, subdir, 'labels')

    # יצירת תיקיות היעד
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"--- Processing {subdir} ---")

    # 1. עיבוד תמונות
    if os.path.exists(in_img_dir):
        for filename in os.listdir(in_img_dir):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                img_path = os.path.join(in_img_dir, filename)
                out_path = os.path.join(out_img_dir, filename)

                image = cv2.imread(img_path)
                if image is not None:
                    processed_img = apply_filters(image)
                    cv2.imwrite(out_path, processed_img)
                else:
                    shutil.copy2(img_path, out_path)  # גיבוי אם הקריאה נכשלה

    # 2. העתקת לייבלים (קבצי txt)
    if os.path.exists(in_lbl_dir):
        for filename in os.listdir(in_lbl_dir):
            if filename.lower().endswith('.txt'):
                shutil.copy2(
                    os.path.join(in_lbl_dir, filename),
                    os.path.join(out_lbl_dir, filename)
                )

print("\nFinished! Your YOLO dataset is ready in 'test images 2'.")