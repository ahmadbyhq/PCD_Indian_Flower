"""Training satu skenario.

Contoh:
    python src/train.py --model mobilenetv2 --prep clahe
    python src/train.py --model efficientnetb0 --prep minmax --seed 123
"""
import argparse
import json
import time

import pandas as pd
from tensorflow import keras

from dataset import get_class_names, make_dataset
from models import build_model, set_finetune
from utils import (LOGS_DIR, MODELS, MODELS_DIR, PREPS, load_config, run_name,
                   set_seed)


def make_callbacks(tcfg: dict) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=tcfg["early_stopping_patience"],
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=tcfg["reduce_lr_factor"],
            patience=tcfg["reduce_lr_patience"], min_lr=1e-7),
    ]


def compile_model(model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


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
    set_seed(seed)

    classes = get_class_names()
    train_ds, _ = make_dataset("train", args.prep, classes, cfg, shuffle=True, seed=seed)
    val_ds, _ = make_dataset("val", args.prep, classes, cfg)

    model, base = build_model(args.model, len(classes), cfg)
    tcfg = cfg["train"]
    t0 = time.time()

    # Tahap 1: latih head saja
    print(f"\n[{name}] Tahap 1: latih head (base dibekukan)")
    compile_model(model, tcfg["head_lr"])
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=tcfg["head_epochs"], callbacks=make_callbacks(tcfg))

    # Tahap 2: fine-tuning layer terakhir dengan LR kecil
    print(f"\n[{name}] Tahap 2: fine-tune {tcfg['unfreeze_layers']} layer terakhir")
    set_finetune(base, tcfg["unfreeze_layers"])
    compile_model(model, tcfg["finetune_lr"])  # wajib compile ulang
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=tcfg["finetune_epochs"], callbacks=make_callbacks(tcfg))

    elapsed = time.time() - t0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"{name}.keras")

    hist = pd.concat([
        pd.DataFrame(h1.history).assign(phase="head"),
        pd.DataFrame(h2.history).assign(phase="finetune"),
    ], ignore_index=True)
    hist.index.name = "epoch"
    hist.to_csv(LOGS_DIR / f"{name}_history.csv")

    meta = {
        "name": name, "model": args.model, "prep": args.prep, "seed": seed,
        "epochs_head": len(h1.epoch), "epochs_finetune": len(h2.epoch),
        "best_val_loss": float(min(h2.history["val_loss"])),
        "best_val_accuracy": float(max(h2.history["val_accuracy"])),
        "train_seconds": round(elapsed, 1),
    }
    (LOGS_DIR / f"{name}_train.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSelesai {name} dalam {elapsed / 60:.1f} menit. Model: models/{name}.keras")


if __name__ == "__main__":
    main()