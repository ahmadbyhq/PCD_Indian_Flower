from pathlib import Path

import pandas as pd
import tensorflow as tf

from utils import PROCESSED_DIR, ROOT, SPLITS_DIR


def get_class_names() -> list[str]:
    """Nama kelas dari train.csv, diurutkan, supaya indeks label konsisten."""
    df = pd.read_csv(SPLITS_DIR / "train.csv")
    return sorted(df["label"].unique())


def resolve_path(raw_path: str, prep: str) -> Path:
    """Petakan path di CSV (data/raw/...) ke file sesuai metode preprocessing."""
    p = Path(raw_path)
    if prep == "original":
        return ROOT / p
    # hasil save_processed.py: data/processed/<prep>/<kelas>/<nama>.png
    return PROCESSED_DIR / prep / p.parent.name / (p.stem + ".png")


def _check_files(paths: list[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file tidak ditemukan, contoh: {missing[:3]}. "
            "Sudah menjalankan data_split.py dan save_processed.py?"
        )


def make_dataset(split: str, prep: str, class_names: list[str], cfg: dict,
                 shuffle: bool = False, seed: int = 42):
    """Return (tf.data.Dataset, array label integer sesuai urutan dataset)."""
    df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    paths = [str(resolve_path(p, prep)) for p in df["path"]]
    _check_files(paths)

    idx = {c: i for i, c in enumerate(class_names)}
    labels = df["label"].map(idx).to_numpy()
    size = cfg["img_size"]

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                 expand_animations=False)
        img = tf.image.resize(img, (size, size))
        img.set_shape((size, size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds, labels