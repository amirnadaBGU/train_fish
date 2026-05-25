import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt


def visualize_first_example(image, label_data, width, height):
    """מציג תמונה לבדיקה ויזואלית באמצעות Matplotlib"""
    class_id, x_center, y_center, w_norm, h_norm = label_data

    # המרה לפיקסלים
    w_px = w_norm * width
    h_px = h_norm * height
    x1 = int((x_center * width) - (w_px / 2))
    y1 = int((y_center * height) - (h_px / 2))
    x2 = int(x1 + w_px)
    y2 = int(y1 + h_px)

    # חישוב AR לפי הלוגיקה שלך
    aspect_ratio = round(max(w_px, h_px) / min(w_px, h_px), 3) if min(w_px, h_px) != 0 else 0

    # המרת צבעים מ-BGR (OpenCV) ל-RGB (Matplotlib)
    display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # יצירת התצוגה
    plt.figure(figsize=(10, 7))
    plt.imshow(display_img)

    # ציור תיבה ידני מעל ה-plot
    rect = plt.Rectangle((x1, y1), w_px, h_px, linewidth=2, edgecolor='g', facecolor='none')
    plt.gca().add_patch(rect)

    plt.title(f"Visual Check: W={int(w_px)} H={int(h_px)} | AR={aspect_ratio}")
    plt.axis('off')

    print("\n--- Visual Check Mode (Matplotlib) ---")
    print(f"Object Specs: Width={int(w_px)}px, Height={int(h_px)}px, Calculated AR={aspect_ratio}")
    print("Close the plot window to start full processing...")

    plt.show()


def process_dataset(base_path):
    images_dir = os.path.join(base_path, 'images')
    labels_dir = os.path.join(base_path, 'labels')
    output_dir = 'DETECTIONS'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_log = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    # וידוא קיום תיקיות
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Could not find 'images' or 'labels' folder in {base_path}")
        return

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)])

    visual_check_done = False
    print(f"Found {len(image_files)} images.")

    for img_idx, img_name in enumerate(image_files):
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        h_img, w_img, _ = image.shape
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path): continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        obj_idx = 0
        for line in lines:
            parts = line.split()
            if not parts: continue

            class_id = int(parts[0])
            if class_id == 0:
                vals = list(map(float, parts[1:]))
                w_px = vals[2] * w_img
                h_px = vals[3] * h_img

                # בדיקה ויזואלית אחת
                if not visual_check_done:
                    visualize_first_example(image, [class_id] + vals, w_img, h_img)
                    visual_check_done = True
                    print("Processing images...")

                ar = round(max(w_px, h_px) / min(w_px, h_px), 3) if min(w_px, h_px) != 0 else 0

                x1 = int((vals[0] * w_img) - (w_px / 2))
                y1 = int((vals[1] * h_img) - (h_px / 2))
                x2, y2 = int(x1 + w_px), int(y1 + h_px)

                crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]

                if crop.size == 0: continue

                file_name = f"{img_idx}-{obj_idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, file_name), crop)

                data_log.append({
                    'filename': file_name,
                    'original_image': img_name,
                    'aspect_ratio': ar,
                    'w': round(w_px, 1),
                    'h': round(h_px, 1)
                })
                obj_idx += 1

    if data_log:
        pd.DataFrame(data_log).to_csv('metadata_aspect_ratios.csv', index=False)
        print(f"\nSuccess! Saved {len(data_log)} objects and created metadata_aspect_ratios.csv")
    else:
        print("\nNo objects of class 0 found.")


if __name__ == "__main__":
    p = 'full_dataset/train'
    process_dataset(p)