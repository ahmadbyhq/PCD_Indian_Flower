"""MobileNetV2 dan EfficientNetB0 dengan head klasifikasi."""
from tensorflow import keras
from tensorflow.keras import layers


def _augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def build_model(name: str, num_classes: int, cfg: dict):
    """Return (model, base_model). Input: piksel 0-255, ukuran img_size x img_size x 3."""
    size = cfg["img_size"]
    weights = "imagenet" if cfg["model"]["pretrained"] else None

    inputs = keras.Input(shape=(size, size, 3))
    x = inputs
    if cfg["augmentation"]["enabled"]:
        x = _augmentation()(x)  # hanya aktif saat training

    if name == "mobilenetv2":
        # MobileNetV2 mengharapkan [-1, 1]
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_rescale")(x)
        base = keras.applications.MobileNetV2(
            include_top=False, weights=weights, input_shape=(size, size, 3))
    elif name == "efficientnetb0":
        # EfficientNetB0 di Keras sudah punya layer rescaling sendiri (input 0-255)
        base = keras.applications.EfficientNetB0(
            include_top=False, weights=weights, input_shape=(size, size, 3))
    else:
        raise ValueError(f"Model tidak dikenal: {name}")

    base.trainable = False
    x = base(x, training=False)  # BatchNorm tetap mode inference
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg["model"]["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name), base


def set_finetune(base: keras.Model, n_layers: int) -> None:
    """Buka n_layers terakhir untuk dilatih; layer BatchNorm tetap beku."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)