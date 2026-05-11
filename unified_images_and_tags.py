# python
from pathlib import Path

# Global variable for the directory name
name = "video_5/train"

def rename_labels_for_images(root_dir: Path):
    images_dir = root_dir / "images"
    labels_dir = root_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Both {images_dir} and {labels_dir} must exist and be directories.")
        return

    label_files = list(labels_dir.iterdir())
    if not label_files:
        print(f"No files found in {labels_dir}.")
        return

    renamed_count = 0
    skipped = 0

    for img in images_dir.iterdir():
        if not img.is_file():
            continue
        name_lower = img.name.lower()
        if "jpg" not in name_lower:
            continue

        # Extract the part before "_jpg"
        if "_jpg" in img.name:
            img_stem = img.name.split("_jpg")[0]
        else:
            img_stem = img.stem

        # Find labels that contain the extracted stem as a substring
        candidates = [f for f in label_files if img_stem in f.stem]

        if len(candidates) == 1:
            candidate = candidates[0]
            target = labels_dir / (img_stem + candidate.suffix)
            if candidate.resolve() == target.resolve():
                continue
            if target.exists():
                print(f"Target exists, skipping: {target}")
                skipped += 1
                continue
            candidate.rename(target)
            renamed_count += 1
            print(f"Renamed: {candidate.name} -> {target.name}")
            continue

        if len(candidates) > 1:
            print(f"Multiple matches found for {img.name} (stem: {img_stem}). Skipping.")
            skipped += 1
            continue

        print(f"Could not find label for image {img.name} (looking for stem: {img_stem}). Skipping.")
        skipped += 1

    print(f"Done. Renamed: {renamed_count}. Skipped: {skipped}.")

if __name__ == "__main__":
    root_directory = Path(name)
    rename_labels_for_images(root_directory)