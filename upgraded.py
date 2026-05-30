from ultralytics import YOLO

# ===================================================================================
# CONFIGURATION AREA - FAST Confusion Matrix Generation
# ===================================================================================

# 1. Basic Paths
MODEL_PATH = "version6.pt"
datasets_dir = "C:/Users/ndvam/PycharmProjects/train_yolo_for_fish_detection/datasets"
DATA_YAML = f"{datasets_dir}/preprocess/sub_models/stretched12/data.yaml"

# 2. Specific Target Parameters
TARGET_CONF = 0.001
IOU_THRESHOLDS = [0.5]

# 3. Model Behavior Parameters
MAX_DET = 1000
IMG_SIZE = 640
DEVICE = 0


# ===================================================================================

def generate_fast_confusion_matrices():
    print(f"Loading model from path: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    for current_iou in IOU_THRESHOLDS:
        run_name = f'conf_{TARGET_CONF}_iou_{current_iou}_FAST'

        print("\n" + "=" * 50)
        print(f"🚀 Starting FAST validation run for: IOU={current_iou} | CONF={TARGET_CONF}")
        print("=" * 50)

        results = model.val(
            data=DATA_YAML,
            split='val',
            name=run_name,
            iou=current_iou,  # תזכורת: זה שולט על ה-NMS, לא על סף המטריצה
            conf=TARGET_CONF,
            max_det=MAX_DET,
            imgsz=IMG_SIZE,
            device=DEVICE,

            # === ⚡ הזרקות מהירות (Speed Optimizations) ⚡ ===
            batch=32,  # הגדלת הבאץ' מאיצה משמעותית את זמן ההסקה (Inference)
            save=False,  # מונע ציור ושמירה של תמונות עם התיבות המזוהות (הכי כבד!)
            save_txt=False,  # מונע שמירת קובצי טקסט של הקואורדינטות
            save_json=False,  # מונע יצירת קובץ JSON כבד
            save_conf=False,  # מונע כתיבת ציוני ביטחון לטקסט
            plots=True  # משאירים True רק כדי לקבל את פלט המטריצות והגרפים הסטטיסטיים
        )

        print(f"\n✅ Lightning Run completed for IOU {current_iou}!")
        print(f"📁 The Confusion Matrix is saved at: {results.save_dir}/confusion_matrix.png")


if __name__ == "__main__":
    generate_fast_confusion_matrices()