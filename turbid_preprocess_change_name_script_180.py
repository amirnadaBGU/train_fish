import cv2
import numpy as np
import os
import shutil
import albumentations as A

# ==========================================
# הגדרות גלובליות
# ==========================================
SIGN = "flip_180" # היפוך כפול שקול לסיבוב של 180 מעלות
TURBID = False

script_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(script_dir, 'test images')
output_base_dir = os.path.join(script_dir, 'test images 2')

if TURBID:
    transform_v2 = A.Compose([
    A.VerticalFlip(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.RGBShift(r_shift_limit=(20, 40), g_shift_limit=(-20, 20), b_shift_limit=(-40, -10), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-50, -20), val_shift_limit=(-50, -20), p=1.0),
    A.GaussianBlur(blur_limit=(35, 101), p=0.5),
    A.OneOf([
        A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(0.001, 0.015), hole_width_range=(0.001, 0.015), fill=(255, 255, 255), p=0.95),
        A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(0.001, 0.015), hole_width_range=(0.001, 0.015), fill=(180, 180, 180), p=0.95),
        A.CoarseDropout(num_holes_range=(1, 100), hole_height_range=(0.001, 0.015), hole_width_range=(0.001, 0.015), fill=(100, 90, 70), p=0.95),
    ], p=1.0),
])
else:
    transform_v2 = A.Compose([
    A.VerticalFlip(p=1.0),
    A.HorizontalFlip(p=1.0)])

subdirs = ['train', 'test', 'valid']
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

images_processed = 0
labels_updated = 0

for subdir in subdirs:
    # הגדרת נתיבי המקור
    in_images_dir = os.path.join(input_base_dir, subdir, 'images')
    in_labels_dir = os.path.join(input_base_dir, subdir, 'labels')

    # דילוג אם התיקייה לא קיימת
    if not os.path.exists(in_images_dir):
        print(f"Skipping {subdir}: 'images' directory not found.")
        continue

    print(f"\n==========================================")
    print(f"Processing directory: {subdir}...")
    print(f"==========================================")

    # הגדרת נתיבי היעד
    out_images_dir = os.path.join(output_base_dir, subdir, 'images')
    out_labels_dir = os.path.join(output_base_dir, subdir, 'labels')

    # יצירת התיקיות החדשות
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # מעבר על כל התמונות בתיקיית images
    for filename in os.listdir(in_images_dir):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        input_file_path = os.path.join(in_images_dir, filename)

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{SIGN}{ext}"
        new_txt_filename = f"{name}_{SIGN}.txt"

        output_file_path = os.path.join(out_images_dir, new_filename)
        output_txt_path = os.path.join(out_labels_dir, new_txt_filename)

        # 1. קריאה והיפוך כפול של התמונה
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

        # 2. חיפוש קובץ ה-TXT בתיקיית labels, היפוך כפול ושמירה
        txt_filename = f"{name}.txt"
        txt_file_path = os.path.join(in_labels_dir, txt_filename)

        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # דילוג על שורות ריקות או Bboxes רגילים ללא פוליגונים

                class_id = parts[0]
                coords = [float(val) for val in parts[1:]]

                new_coords = []
                # מכיוון שאנחנו הופכים גם X וגם Y, כל מספר ברצף מתהפך פשוט על ידי חיסור מ-1.0!
                for val in coords:
                    flipped_val = 1.0 - val
                    # מוודא שלא חרגנו מהגבולות בגלל עיגולי שגיאה זעירים של פייתון
                    flipped_val = max(0.0, min(flipped_val, 1.0))
                    new_coords.append(flipped_val)

                # הרכבת השורה מחדש
                coords_str = " ".join([f"{val:.6f}" for val in new_coords])
                new_line = f"{class_id} {coords_str}\n"
                new_lines.append(new_line)

            # שמירת קובץ ה-TXT ההפוך בתיקיית היעד
            with open(output_txt_path, 'w') as file:
                file.writelines(new_lines)
            labels_updated += 1
        else:
            print(f"[!] הערה: לא נמצא קובץ תיוג (txt) לתמונה {filename}")

print("\n==========================================")
print(f"Done! Processed {images_processed} images and mathematically flipped {labels_updated} YOLO label files by 180 degrees.")