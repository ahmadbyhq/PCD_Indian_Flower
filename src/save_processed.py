from pathlib import Path
import cv2
import numpy as np
from preprocessing import to_minmax, to_clahe

METHODS = {"minmax": to_minmax, "clahe": to_clahe}
total = 0

for class_dir in sorted(p for p in Path("data/raw").iterdir() if p.is_dir()):
    for p in sorted(class_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            print("Gagal baca:", p)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for name, fn in METHODS.items():
            out_img = (fn(rgb) * 255).round().astype(np.uint8)
            dst = Path("data/processed") / name / class_dir.name / (p.stem + ".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        total += 1
    print("Selesai:", class_dir.name)

print(f"Total gambar diproses: {total}")