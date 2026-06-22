import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ===================================================================================
# CONFIGURATION
# ===================================================================================

MODEL_PATH = "version6.pt"
datasets_dir = "C:/Users/ndvam/PycharmProjects/train_yolo_for_fish_detection/datasets"
DATA_YAML = f"{datasets_dir}/preprocess/stretched/data.yaml"

# Dense sweep — free because we run val only once
#CONF_THRESHOLDS = np.round(np.arange(0.001, 0.011, 0.001), 3)

CONF_THRESHOLDS = np.round(np.arange(0.001, 0.011, 0.001), 3)

# Per-class deployment thresholds — must match CONF_THRESHOLDS in extract_crops.py.
# These are marked on the output plots so you can read P/R directly at the operating point.
DEPLOY_CONF = {0: 0.02, 1: 0.07}   # class_id → threshold (fish, partial_fish)

IOU_THRESHOLD = 0.5
MAX_DET = 1000
IMG_SIZE = 640
DEVICE = 0
BATCH_SIZE = 64
# Must match extract_crops: no half-precision there, so keep False here for consistency.
# Flip both to True together if you want speed — keep them in sync.
HALF_PRECISION = False
# Must match AGNOSTIC_NMS in extract_crops.py (False = class-aware NMS).
AGNOSTIC_NMS = False

OUTPUT_CSV = "confidence_sweep_results.csv"
OUTPUT_PLOT = "precision_recall_sweep.png"

# ===================================================================================


def run_confidence_sweep():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"Classes: {class_names}")

    # Single pass with very low conf captures predictions across the full range.
    # The validator internally computes P/R curves at 1000 confidence thresholds,
    # so we only need to sample those curves — no need to re-run for each threshold.
    print("\nRunning single validation pass (captures full confidence curve)...")
    # agnostic_nms must match AGNOSTIC_NMS in extract_crops.py.
    # Supported as a kwarg in ultralytics >= 8.1. If you get an unexpected-keyword
    # error on an older version, replace the kwarg below with:
    #   model.overrides['agnostic_nms'] = AGNOSTIC_NMS
    # placed before this model.val() call.
    val_results = model.val(
        data=DATA_YAML,
        split='val',
        name='full_sweep',
        iou=IOU_THRESHOLD,
        conf=0.001,
        max_det=MAX_DET,
        imgsz=IMG_SIZE,
        device=DEVICE,
        batch=BATCH_SIZE,
        plots=True,   # needed: ultralytics only fills the confusion matrix when plots=True
        verbose=False,
        half=HALF_PRECISION,
        agnostic_nms=AGNOSTIC_NMS,
        save=False,
        save_txt=False,
        save_json=False,
        save_conf=False,
    )

    try:
        rows = _extract_from_curves(val_results, class_names)
        print("Extracted P/R from precomputed curves.")
    except Exception as e:
        print(f"Curve extraction failed ({e}); falling back to per-threshold runs...")
        rows = _multi_run_fallback(model, class_names)

    df = pd.DataFrame(rows)
    # Group columns by class: precision, recall, tp, fp, fn for each class
    metric_order = ['precision', 'recall', 'tp', 'fp', 'fn']
    ordered_cols = ['conf']
    for class_name in class_names.values():
        for m in metric_order:
            col = f'{m}_{class_name}'
            if col in df.columns:
                ordered_cols.append(col)
    df = df[ordered_cols]

    print("\nResults:")
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    _plot(df, class_names)
    print(f"Saved: {OUTPUT_PLOT}")


def _get_n_gt(val_results, class_names):
    """
    Return {class_name: n_gt} from the confusion matrix.
    Columns of the matrix (rows=predicted, cols=actual) sum to total GT per class
    regardless of whether they were detected — so the count is threshold-independent.
    Requires plots=True in model.val(), otherwise ultralytics skips matrix population.
    """
    n_gt = {}
    try:
        cm = val_results.confusion_matrix.matrix  # shape (nc+1, nc+1)
        if cm.sum() == 0:
            print("  WARNING: confusion matrix is empty (plots=True required). n_gt will be 0.")
            return n_gt
        for class_id, class_name in class_names.items():
            if class_id < cm.shape[1] - 1:  # exclude background column
                n_gt[class_name] = int(cm[:, class_id].sum())
    except AttributeError:
        print("  WARNING: confusion_matrix not found on results object. n_gt will be 0.")
    return n_gt


def _tp_fp_fn(p, r, n_gt):
    """Derive integer TP/FP/FN from precision, recall, and ground-truth count."""
    tp = round(r * n_gt)
    fn = n_gt - tp
    fp = max(0, round(tp * (1 - p) / p)) if p > 1e-6 else 0
    return int(tp), int(fp), int(fn)


def _extract_from_curves(val_results, class_names):
    """
    Sample P/R at each desired threshold from the 1000-point curves already
    computed by the validator (results.box.p_curve / r_curve, x-axis = results.box.px).
    """
    box = val_results.box
    p_curves = np.array(box.p_curve)  # (nc, 1000)
    r_curves = np.array(box.r_curve)  # (nc, 1000)
    conf_axis = np.array(box.px)      # (1000,) confidence thresholds 0 → 1

    if p_curves.ndim != 2 or p_curves.shape[1] == 0:
        raise ValueError(f"Unexpected p_curve shape: {p_curves.shape}")

    n_gt = _get_n_gt(val_results, class_names)

    rows = []
    for conf in CONF_THRESHOLDS:
        idx = int(np.clip(np.searchsorted(conf_axis, conf), 0, len(conf_axis) - 1))
        row = {'conf': float(conf)}
        for class_id, class_name in class_names.items():
            if class_id < len(p_curves):
                p = float(p_curves[class_id, idx])
                r = float(r_curves[class_id, idx])
                tp, fp, fn = _tp_fp_fn(p, r, n_gt.get(class_name, 0))
                row[f'precision_{class_name}'] = round(p, 4)
                row[f'recall_{class_name}']    = round(r, 4)
                row[f'tp_{class_name}']        = tp
                row[f'fp_{class_name}']        = fp
                row[f'fn_{class_name}']        = fn
        rows.append(row)
    return rows


def _multi_run_fallback(model, class_names):
    """Run model.val() separately for each threshold when curve data is unavailable."""
    rows = []
    total = len(CONF_THRESHOLDS)
    for i, conf in enumerate(CONF_THRESHOLDS, 1):
        print(f"  [{i}/{total}] conf={conf:.2f}", end='', flush=True)
        results = model.val(
            data=DATA_YAML,
            split='val',
            iou=IOU_THRESHOLD,
            conf=conf,
            max_det=MAX_DET,
            imgsz=IMG_SIZE,
            device=DEVICE,
            batch=BATCH_SIZE,
            plots=False,
            verbose=False,
            half=HALF_PRECISION,
            agnostic_nms=AGNOSTIC_NMS,
            save=False,
            save_txt=False,
        )
        row = {'conf': float(conf)}
        n_gt = _get_n_gt(results, class_names)
        p_arr = np.array(results.box.p)
        r_arr = np.array(results.box.r)
        for class_id, class_name in class_names.items():
            if class_id < len(p_arr):
                p = float(p_arr[class_id])
                r = float(r_arr[class_id])
                tp, fp, fn = _tp_fp_fn(p, r, n_gt.get(class_name, 0))
                row[f'precision_{class_name}'] = round(p, 4)
                row[f'recall_{class_name}']    = round(r, 4)
                row[f'tp_{class_name}']        = tp
                row[f'fp_{class_name}']        = fp
                row[f'fn_{class_name}']        = fn
        rows.append(row)
        print(" done")
    return rows


def _plot(df, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for ax, metric, title in [
        (axes[0], 'recall',    'Recall vs Confidence Threshold'),
        (axes[1], 'precision', 'Precision vs Confidence Threshold'),
    ]:
        for (class_id, class_name), color in zip(class_names.items(), colors):
            col = f'{metric}_{class_name}'
            if col not in df.columns:
                continue
            ax.plot(df['conf'], df[col], marker='o', markersize=3,
                    label=class_name, color=color, linewidth=2)

            # Mark the deployment operating point for this class (from DEPLOY_CONF).
            deploy_conf = DEPLOY_CONF.get(class_id)
            if deploy_conf is not None:
                # Find the row closest to the deploy threshold.
                idx = (df['conf'] - deploy_conf).abs().idxmin()
                x_pt = df.loc[idx, 'conf']
                y_pt = df.loc[idx, col]
                ax.axvline(x=deploy_conf, color=color, linewidth=1,
                           linestyle='--', alpha=0.5)
                ax.scatter([x_pt], [y_pt], color=color, s=80, zorder=5,
                           marker='D', edgecolors='black', linewidths=0.8)
                ax.annotate(f'{class_name}\n@ {deploy_conf}',
                            xy=(x_pt, y_pt), xytext=(6, -18),
                            textcoords='offset points', fontsize=8,
                            color=color)

        ax.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_confidence_sweep()
