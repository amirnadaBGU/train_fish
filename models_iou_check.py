from ultralytics import YOLO

# ===================================================================================
# CONFIGURATION AREA
# ===================================================================================

MODEL_PATH = "stretched.pt"  # נתיב למשקלי המודל שלך
DATA_YAML = "dataset/data.yaml"  # נתיב לקובץ ה-YAML של הדאטה

# רשימת ספי ה-NMS שאנחנו רוצים להשוות.
# 0.45 הוא סטנדרטי/אגרסיבי, 0.95 הוא סלחני מאוד ויאפשר כפילויות.
NMS_IOU_THRESHOLDS = [0.45, 0.95]

MAX_DET = 1000  # קריטי לבריכות דגים צפופות!
IMG_SIZE = 640
CONF_MIN = 0.001
SAVE_PLOTS = True
DEVICE = 0  # ה-RTX 4060 שלך ייכנס לפעולה כאן


# ===================================================================================

def evaluate_and_compare_nms():
    print(f"Loading model from path: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # מילון לאגירת התוצאות כדי שנוכל להדפיס טבלת השוואה יפה בסוף
    summary_results = {}

    for nms_thresh in NMS_IOU_THRESHOLDS:
        print("\n" + "*" * 50)
        print(f"🚀 Starting validation run with NMS IOU: {nms_thresh}")
        print("*" * 50)

        # יצירת שם תיקייה ייעודי לכל הרצה כדי שהגרפים והתמונות לא ידרסו
        run_name = f'fish_eval_nms_{nms_thresh}'

        # הרצת הטסט על סמך הפרמטרים
        results = model.val(
            data=DATA_YAML,
            split='test',  # שימוש מפורש בסט המבחן
            name=run_name,
            iou=nms_thresh,  # <--- כאן אנחנו מזריקים את סף ה-NMS השתנה
            max_det=MAX_DET,
            imgsz=IMG_SIZE,
            conf=CONF_MIN,
            plots=SAVE_PLOTS,
            device=DEVICE
        )

        # אגירת המדדים
        summary_results[nms_thresh] = {
            'map50': results.box.map50,
            'map50_95': results.box.map,
            'save_dir': results.save_dir
        }

    # הדפסת סיכום ההשוואה
    print("\n" + "=" * 60)
    print("🏆 סיכום השוואת mAP@50 לפי סף NMS 🏆")
    print("=" * 60)

    for nms, metrics in summary_results.items():
        print(f"NMS Threshold: {nms:<4} | mAP@50: {metrics['map50']:.4f} | mAP@50-95: {metrics['map50_95']:.4f}")
        print(f"📁 שמור בתיקייה: {metrics['save_dir']}\n")

    print("💡 בדיקה ויזואלית מומלצת:")
    print("   פתח את שתי התיקיות שנוצרו והשווה את התמונה 'val_batch0_pred.jpg'.")
    print("   אתה צפוי לראות שבתיקייה של NMS=0.95 יש הרבה כפילויות על אותו דג,")
    print("   מה שיסביר את הירידה הצפויה בציון ה-mAP@50.")


if __name__ == "__main__":
    evaluate_and_compare_nms()