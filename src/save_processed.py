from pathlib import Path
import cv2
import numpy as np
from preprocessing import to_minmax, to_clahe

SRC = Path("data/raw")
OUT = Path("data/processed")
METHODS = {"minmax": to_minmax, "clahe": to_clahe}

for class_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    for p in sorted(class_dir.glob("*.jpg")):
        bgr = cv2.imread(str(p))
        if bgr is None:
            print("Gagal baca:", p)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for name, fn in METHODS.items():
            out_img = (fn(rgb) * 255).round().astype(np.uint8)
            dst = OUT / name / class_dir.name / (p.stem + ".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print("Selesai:", class_dir.name)