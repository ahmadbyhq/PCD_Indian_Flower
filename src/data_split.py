from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

paths = sorted(p for p in Path("data/raw").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
df = pd.DataFrame({
    "path": [p.as_posix() for p in paths],
    "label": [p.parent.name for p in paths],
})
print(f"Total: {len(df)} gambar, {df.label.nunique()} kelas")

# Split data menjadi train, val, dan test
train, temp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
# data temp dibagi menjadi dua bagian: val dan test
val, test = train_test_split(temp, test_size=0.50, stratify=temp.label, random_state=42)

out = Path("data/splits")
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "train.csv", index=False)
val.to_csv(out / "val.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print("Min gambar per kelas di test:", test.label.value_counts().min())