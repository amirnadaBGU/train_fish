from ultralytics import YOLO

# ===================================================================================
# CONFIGURATION AREA - Tweak these parameters to see how they affect mAP and the graph
# ===================================================================================

# 1. Basic Paths
MODEL_PATH = "white_edges.pt"  # Path to your trained model weights (e.g., best.pt)
DATA_YAML = "dataset/data.yaml"  # Path to your dataset configuration file

# 2. Model Behavior Parameters (Your "Tuning Knobs")
NMS_IOU = 0.95  # Non-Maximum Suppression IoU threshold.
# 0.7 = Permissive (allows overlap), 0.4 = Strict (removes overlaps).
MAX_DET = 1000  # Maximum detections per image. *CRITICAL FOR FISH POOLS!*
# Default is 300. If a pool has >300 fish, you must increase this,
# otherwise the model ignores them, destroying Recall and F1.
IMG_SIZE = 640  # Image resolution for validation. Recommended to keep this
# identical to the resolution used during model training.

# 3. Graph Plotting Parameters (F1-Confidence)
CONF_MIN = 0.001  # Minimum confidence threshold. Set close to 0 during validation
# so the model calculates and draws the full F1 curve (from 0 to 1).
SAVE_PLOTS = True  # Must be True to automatically generate 'F1_curve.png' and other plots.

# 4. System Settings
DEVICE = 0  # 0 = Use GPU (CUDA). Change to 'cpu' if no GPU is available.


# ===================================================================================

def evaluate_and_plot():
    print(f"Loading model from path: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("\n" + "*" * 50)
    print("Starting validation run with the following parameters:")
    print(f"NMS IOU:      {NMS_IOU}")
    print(f"Max Detects:  {MAX_DET}")
    print(f"Image Size:   {IMG_SIZE}")
    print("*" * 50 + "\n")

    # Run validation with explicit parameters
    results = model.val(
        data=DATA_YAML,
        split ='test',
        project='C:/Users/amirnada/PycharmProjects/train_fish/runs',
        name='fish_evaluation',
        iou=NMS_IOU,
        max_det=MAX_DET,
        imgsz=IMG_SIZE,
        conf=CONF_MIN,
        plots=SAVE_PLOTS,
        device=DEVICE
    )

    # Print final evaluation metrics to the console
    print("\n" + "=" * 40)
    print("🏆 Evaluation Metrics:")
    print("=" * 40)

    # mAP at 50% Metric IoU
    print(f"mAP@50:         {results.box.map50:.4f}")

    # Average mAP across Metric IoU thresholds from 50% to 95%
    print(f"mAP@50-95:      {results.box.map:.4f}")
    print("=" * 40)

    # Guide the user to the results directory
    print(f"\n✅ Run completed! All results, including the F1 graph, are saved in:")
    print(f"📁 {results.save_dir}")
    print("\n💡 TIP: Navigate to the directory and open 'F1_curve.png' to view the graph,")
    print("   and open 'val_batch0_pred.jpg' to visually inspect the model's predictions.")


if __name__ == "__main__":
    evaluate_and_plot()