import fiftyone as fo
import fiftyone.utils.ultralytics as fou
from fiftyone import ViewField as F
from ultralytics import YOLO
import os

# ---------------------------------------------------------
# 1. Setup & Configuration
# ---------------------------------------------------------
fo.config.database_validation = False

model = YOLO("version6.pt")
classes = model.names

images_dir = os.path.abspath("dataset/valid/images")
labels_dir = os.path.abspath("dataset/valid/labels")

print(f"\n[Check] Searching for images in: {images_dir}")
if not os.path.exists(images_dir):
    print(f"[ERROR] The directory '{images_dir}' does NOT exist! Please check the path.")
    exit()

dataset_name = "fish_final_comparison"
if fo.dataset_exists(dataset_name):
    print(f"Deleting existing dataset: {dataset_name}...")
    fo.load_dataset(dataset_name).delete()

print(f"Creating new dataset: {dataset_name}...")
dataset = fo.Dataset(name=dataset_name)
dataset.add_dir(dataset_dir=images_dir, dataset_type=fo.types.ImageDirectory)

if len(dataset) == 0:
    print("\n[CRITICAL] Dataset is empty. Script stopped.")
    exit()

# ---------------------------------------------------------
# 2. Load Ground Truth
# ---------------------------------------------------------
print("Loading Ground Truth labels manually...")
for sample in dataset:
    img_name = os.path.splitext(os.path.basename(sample.filepath))[0]
    label_path = os.path.join(labels_dir, img_name + ".txt")

    if os.path.exists(label_path):
        detections = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xc, yc, w, h = map(float, parts)
                    detections.append(
                        fo.Detection(
                            label=classes[int(c)],
                            bounding_box=[xc - w / 2, yc - h / 2, w, h]
                        )
                    )
        sample["ground_truth"] = fo.Detections(detections=detections)
        sample.save()

# ---------------------------------------------------------
# 3. Run Inference
# ---------------------------------------------------------
conf_thresholds = [0.05]
fixed_nms = 0.45

print("\nStarting inference for different confidence thresholds...")
for conf in conf_thresholds:
    field_name = f"conf_{str(conf).replace('.', '_')}"
    print(f"  -> Processing Conf={conf}...")

    for sample in dataset:
        result = model.predict(
            sample.filepath, conf=conf, iou=fixed_nms,
            agnostic_nms=True, verbose=False, device="cuda:0"
        )[0]

        detections = fou.to_detections(result)
        if detections is None:
            detections = fo.Detections(detections=[])

        sample[field_name] = detections
        sample.save()

# ---------------------------------------------------------
# 4. Smart Evaluation (Per Class + All)
# ---------------------------------------------------------
print("\nEvaluating results (Calculating TP, FP, FN for ALL and for specific classes)...")
for conf in conf_thresholds:
    pred_field = f"conf_{str(conf).replace('.', '_')}"

    if pred_field in dataset.get_field_schema():
        # א. הערכה כוללת
        dataset.evaluate_detections(
            pred_field=pred_field, gt_field="ground_truth",
            eval_key=f"eval_ALL_{pred_field}"
        )

        # ב. הערכה מדויקת רק ל-FISH
        dataset.evaluate_detections(
            pred_field=pred_field, gt_field="ground_truth",
            eval_key=f"eval_FISH_{pred_field}", classes=["fish"]
        )

        # ג. הערכה מדויקת רק ל-PARTIAL FISH
        dataset.evaluate_detections(
            pred_field=pred_field, gt_field="ground_truth",
            eval_key=f"eval_PARTIAL_{pred_field}", classes=["partial fish"]
        )

# ---------------------------------------------------------
# 5. Create Filtered Views
# ---------------------------------------------------------
print("\nCreating dynamic views for the UI...")

categories = ["fish", "partial fish"]
for category in categories:
    cat_view = dataset.filter_labels("ground_truth", F("label") == category)

    for conf in conf_thresholds:
        pred_field = f"conf_{str(conf).replace('.', '_')}"
        if pred_field in dataset.get_field_schema():
            cat_view = cat_view.filter_labels(pred_field, F("label") == category, only_matches=False)

    view_name = f"ONLY_{category.replace(' ', '_').upper()}"
    dataset.save_view(view_name, cat_view)

# ---------------------------------------------------------
# 6. UI Configuration & Launch App
# ---------------------------------------------------------
print("\nConfiguring UI and Launching FiftyOne App...")

# בחירת הטרשהולד והמחלקה שיוצגו כברירת מחדל בתחתית התמונות
chosen_conf = "0_05"  # ניתן לשנות ל- "0_3", "0_15" וכו'
target_eval = "eval_FISH"  # ניתן לשנות ל- "eval_PARTIAL" או "eval_ALL"

# פקודה שמגדירה מראש את הממשק כך שיציג אך ורק את השדות שבחרנו
dataset.app_config.grid_fields = [
    "id",
    "filepath",
    "tags",
    f"{target_eval}_conf_{chosen_conf}_tp",
    f"{target_eval}_conf_{chosen_conf}_fp",
    f"{target_eval}_conf_{chosen_conf}_fn"
]
dataset.save()

session = fo.launch_app(dataset)
session.wait()