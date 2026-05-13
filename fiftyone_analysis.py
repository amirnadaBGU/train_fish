import fiftyone as fo
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO
import os

# Suppress MongoDB version validation errors
fo.config.database_validation = False

# 1. Load the YOLO model
model = YOLO('white_edges.pt')
classes = model.names  # Dictionary of classes {0: 'fish', ...}

# 2. Dataset Setup
images_dir = os.path.abspath("dataset/test/images")
labels_dir = os.path.abspath("dataset/test/labels")

dataset_name = "fish_final_comparison"
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)

# Create the dataset from images (confirmed working for you)
dataset = fo.Dataset.from_dir(
    dataset_dir=images_dir,
    dataset_type=fo.types.ImageDirectory,
    name=dataset_name
)

# 3. Manual GT Loading - The "Bulletproof" way
print("Loading Ground Truth labels manually...")
for sample in dataset:
    img_name = os.path.splitext(os.path.basename(sample.filepath))[0]
    label_path = os.path.join(labels_dir, img_name + ".txt")

    if os.path.exists(label_path):
        detections = []
        with open(label_path, "r") as f:
            for line in f:
                # YOLO format: class_id x_center y_center width height
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xc, yc, w, h = map(float, parts)
                    # Convert YOLO (center) to FiftyOne (top-left) format
                    # FiftyOne expects [x_top_left, y_top_left, width, height]
                    detections.append(
                        fo.Detection(
                            label=classes[int(c)],
                            bounding_box=[xc - w / 2, yc - h / 2, w, h]
                        )
                    )

        sample["ground_truth"] = fo.Detections(detections=detections)
        sample.save()

# 4. Perform Inference for different IOU levels
iou_levels = [0.01,0.99]

print(f"Total samples: {len(dataset)}")
print("Starting inference for different IOU levels...")

for iou_val in iou_levels:
    field_name = f"iou_{str(iou_val).replace('.', '_')}"
    print(f"Processing IOU: {iou_val}")

    for sample in dataset:
        # Predict using current IOU
        result = model.predict(sample.filepath, conf=0.1, iou=iou_val, agnostic_nms=True, verbose=False,device='cpu')[0]
        # Convert and save
        sample[field_name] = fou.to_detections(result)
        sample.save()

print("Inference finished. Launching app...")

# 5. Launch the Interactive Interface
session = fo.launch_app(dataset)
session.wait()