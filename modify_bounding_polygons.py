# python
import os
import cv2
from pathlib import Path


def convert_segmentation_to_detection(library_dir: str):
    """Convert YOLO segmentation labels to detection format."""
    library_path = Path(library_dir)

    # Find labels directory
    labels_dir = library_path / "labels"
    images_dir = library_path / "images"

    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return

    converted_count = 0

    for txt_file in labels_dir.glob("*.txt"):
        # Find corresponding image to get dimensions
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = images_dir / (txt_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if not img_file:
            print(f"Image not found for {txt_file.name}, skipping...")
            continue

        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed to read image: {img_file}")
            continue

        H, W = img.shape[:2]

        # Convert labels
        new_lines = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = parts[0]
                coords = list(map(float, parts[1:]))

                # Check if segmentation (even number of coordinates, at least 6)
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    # Convert polygon to bounding box
                    x_coords = [coords[i] * W for i in range(0, len(coords), 2)]
                    y_coords = [coords[i] * H for i in range(1, len(coords), 2)]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    center_x = ((x_min + x_max) / 2) / W
                    center_y = ((y_min + y_max) / 2) / H
                    width = (x_max - x_min) / W
                    height = (y_max - y_min) / H

                    # Clamp to [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    new_lines.append(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                else:
                    # Already in detection format
                    new_lines.append(line)

        # Write converted labels
        with open(txt_file, "w") as f:
            f.writelines(new_lines)

        converted_count += 1
        print(f"Converted: {txt_file.name}")

    print(f"\nDone. Converted {converted_count} label files.")


if __name__ == "__main__":
    library_name = "videos/test"  # Change this to your library path
    convert_segmentation_to_detection(library_name)
