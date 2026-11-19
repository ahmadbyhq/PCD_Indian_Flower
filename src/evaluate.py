"""Evaluasi satu skenario di data TEST.

Contoh:
    python src/evaluate.py --model mobilenetv2 --prep clahe
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from tensorflow import keras

from dataset import get_class_names, make_dataset
from utils import CM_DIR, METRICS_DIR, MODELS, MODELS_DIR, PREPS, load_config, run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--prep", required=True, choices=PREPS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    name = run_name(args.model, args.prep, seed)

    classes = get_class_names()
    test_ds, y_true = make_dataset("test", args.prep, classes, cfg)
    model = keras.models.load_model(MODELS_DIR / f"{name}.keras")

    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0)

    result = {
        "name": name, "model": args.model, "prep": args.prep, "seed": seed,
        "n_test": int(len(y_true)), "accuracy": acc,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "per_class": report,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(CM_DIR / f"{name}.csv")

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 6})
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix: {name} (acc={acc:.4f})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{name}.png", dpi=150)
    plt.close()

    print(f"{name}: acc={acc:.4f} | P(macro)={p_m:.4f} R(macro)={r_m:.4f} F1(macro)={f_m:.4f}")


if __name__ == "__main__":
    main()