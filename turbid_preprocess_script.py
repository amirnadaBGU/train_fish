import cv2
import numpy as np
import os
import shutil
import albumentations as A

# הגדרת נתיבים
script_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(script_dir, 'test images')
output_base_dir = os.path.join(script_dir, 'test images 2')

# הגדרת ה-Pipeline של Albumentations
transform_v2 = A.Compose([
    # A. USE RGBShift to add HEAT (more red, less blue)
    A.RGBShift(
        r_shift_limit=(20, 40),   # תמיד יוסיף אדום
        g_shift_limit=(-20, 20),
        b_shift_limit=(-40, -10), # תמיד יוריד כחול
        p=1.0
    ),
    # B. Darkening and desaturation
    A.HueSaturationValue(
        hue_shift_limit=0,
        sat_shift_limit=(-50, -20),
        val_shift_limit=(-50, -20),
        p=1.0
    ),
    # C. Blur for murky effect
    A.GaussianBlur(
        blur_limit=(35, 101),
        p=0.5
    ),
    # D. CoarseDropout for particles
    A.OneOf([
        # אופציה 1: חלקיקים לבנים (בועות/החזרי אור)
        A.CoarseDropout(
            num_holes_range=(1, 100),
            hole_height_range=(0.001, 0.015),
            hole_width_range=(0.001, 0.015),
            fill=(255, 255, 255),
            p=0.95
        ),
        # אופציה 2: חלקיקים אפרפרים (לכלוך סטנדרטי)
        A.CoarseDropout(
            num_holes_range=(1, 100),
            hole_height_range=(0.001, 0.015),
            hole_width_range=(0.001, 0.015),
            fill=(180, 180, 180),
            p=0.95
        ),
        # אופציה 3: חלקיקים חומים/ירקרקים (בוץ/אצות)
        A.CoarseDropout(
            num_holes_range=(1, 100),
            hole_height_range=(0.001, 0.015),
            hole_width_range=(0.001, 0.015),
            fill=(100, 90, 70),
            p=0.95
        ),
    ], p=1.0),
])

# תתי-התיקיות שעליהן נרצה לעבור
subdirs = ['train', 'test', 'valid']

# סיומות של קבצי תמונה נפוצים
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

for subdir in subdirs:
    input_subdir = os.path.join(input_base_dir, subdir)
    output_subdir = os.path.join(output_base_dir, subdir)

    os.makedirs(output_subdir, exist_ok=True)

    if not os.path.exists(input_subdir):
        print(f"Warning: Directory '{input_subdir}' not found. Skipping.")
        continue

    print(f"Processing directory: {subdir}...")

    for filename in os.listdir(input_subdir):
        input_file_path = os.path.join(input_subdir, filename)
        output_file_path = os.path.join(output_subdir, filename)

        if os.path.isdir(input_file_path):
            continue

        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image = cv2.imread(input_file_path)

            if image is None:
                print(f"Failed to read image {filename}. Copying file instead.")
                shutil.copy2(input_file_path, output_file_path)
                continue

            # המרה ל-RGB כי Albumentations עובד בפורמט זה
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # הפעלת הפילטרים
            transformed = transform_v2(image=image_rgb)
            processed_image_rgb = transformed['image']

            # המרה חזרה ל-BGR לצורך שמירה עם OpenCV
            final_image = cv2.cvtColor(processed_image_rgb, cv2.COLOR_RGB2BGR)

            # שמירת התמונה המעובדת
            cv2.imwrite(output_file_path, final_image)

        else:
            shutil.copy2(input_file_path, output_file_path)

print("Finished processing all images and copying files!")