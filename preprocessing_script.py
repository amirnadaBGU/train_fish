import cv2
import numpy as np
import os
import shutil

# הגדרת נתיבים
script_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(script_dir, 'test images')
output_base_dir = os.path.join(script_dir, 'test images 2')

# תתי-התיקיות שעליהן נרצה לעבור
subdirs = ['train', 'test', 'valid']

# סיומות של קבצי תמונה נפוצים
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

for subdir in subdirs:
    input_subdir = os.path.join(input_base_dir, subdir)
    output_subdir = os.path.join(output_base_dir, subdir)

    # יצירת תת-תיקיית היעד אם היא לא קיימת
    os.makedirs(output_subdir, exist_ok=True)

    # בדיקה שתיקיית המקור אכן קיימת כדי למנוע שגיאות
    if not os.path.exists(input_subdir):
        print(f"Warning: Directory '{input_subdir}' not found. Skipping.")
        continue

    print(f"Processing directory: {subdir}...")

    # מעבר על כל הקבצים בתת-התיקייה
    for filename in os.listdir(input_subdir):
        input_file_path = os.path.join(input_subdir, filename)
        output_file_path = os.path.join(output_subdir, filename)

        # דילוג על תיקיות (במידה ויש תיקיות פנימיות בטעות)
        if os.path.isdir(input_file_path):
            continue

        # בדיקה אם הקובץ הוא תמונה
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            # קריאת התמונה
            image = cv2.imread(input_file_path)

            # אם cv2 לא הצליח לקרוא את התמונה, נעתיק אותה כקובץ רגיל
            if image is None:
                print(f"Failed to read image {filename}. Copying file instead.")
                shutil.copy2(input_file_path, output_file_path)
                continue

            # --- הפעלת הפילטרים שלך ---
            # Apply CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            clahed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # sharpened after clahe
            clahed_then_blurred = cv2.GaussianBlur(clahed, (55, 55), 0)
            clahed_then_sharpened = cv2.addWeighted(clahed, 1.8, clahed_then_blurred, -0.8, 0)
            # ---------------------------

            # שמירת התמונה המעובדת באותו שם
            cv2.imwrite(output_file_path, clahed_then_sharpened)

        else:
            # אם זה לא קובץ תמונה (למשל קבצי לייבלים), פשוט מעתיקים אותו
            shutil.copy2(input_file_path, output_file_path)

print("Finished processing all images and copying files!")