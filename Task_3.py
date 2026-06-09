import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)


IMG_SIZE    = (96, 96)    
BATCH_SIZE  = 32
EPOCHS      = 20
IMG_CHANNELS = 3

print("Loading California Housing dataset…")
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print(f"Shape : {df.shape}")
print(df.describe())

target_col  = "MedHouseVal"
feature_cols = [c for c in df.columns if c != target_col]

X_tab = df[feature_cols].values.astype("float32")
y     = df[target_col].values.astype("float32")

y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std



def generate_dummy_image(house_value: float, img_size=(96, 96)) -> np.ndarray:
    brightness = np.clip(house_value / 5.0, 0.1, 1.0)   # normalise to [0,1]
    noise      = np.random.normal(0, 0.05, (*img_size, IMG_CHANNELS))
    img        = np.full((*img_size, IMG_CHANNELS), brightness) + noise
    img        = np.clip(img * 255, 0, 255).astype("uint8")
    return img

print("\nGenerating synthetic house images…")
images = np.stack([generate_dummy_image(v) for v in y], axis=0)  # (N, 96, 96, 3)
print(f"Images array shape: {images.shape}")


X_tab_tv, X_tab_test, imgs_tv, imgs_test, y_tv, y_test = train_test_split(
    X_tab, images, y_scaled,
    test_size=0.20, random_state=42,
)

X_tab_train, X_tab_val, imgs_train, imgs_val, y_train, y_val = train_test_split(
    X_tab_tv, imgs_tv, y_tv,
    test_size=0.25, random_state=42,
)

print(f"\nTrain: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab_train).astype("float32")
X_tab_val   = scaler.transform(X_tab_val).astype("float32")
X_tab_test  = scaler.transform(X_tab_test).astype("float32")


def preprocess_image(img):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img

def make_dataset(tab_feats, imgs, labels, shuffle=False):
    imgs_preprocessed = np.array([preprocess_image(i).numpy() for i in imgs], dtype="float32")
    ds = tf.data.Dataset.from_tensor_slices(
        ({"tabular_input": tab_feats, "image_input": imgs_preprocessed}, labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=2048)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_tab_train, imgs_train, y_train, shuffle=True)
val_ds   = make_dataset(X_tab_val,   imgs_val,   y_val)
test_ds  = make_dataset(X_tab_test,  imgs_test,  y_test)


def build_multimodal_model(tabular_dim: int) -> Model:
    image_input = keras.Input(shape=(*IMG_SIZE, IMG_CHANNELS), name="image_input")

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, IMG_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x_img = base_model(image_input, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(64, activation="relu")(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    tabular_input = keras.Input(shape=(tabular_dim,), name="tabular_input")
    x_tab = layers.Dense(64, activation="relu")(tabular_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Dense(32, activation="relu")(x_tab)
    x_tab = layers.Dropout(0.2)(x_tab)

    combined = layers.Concatenate()([x_img, x_tab])
    x = layers.Dense(64, activation="relu")(combined)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="linear", name="price_output")(x)

    model = Model(inputs=[image_input, tabular_input], outputs=output)
    return model


model = build_multimodal_model(tabular_dim=X_tab_train.shape[1])
model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
]

print("\nTraining multimodal model…")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


print("\nEvaluating on test set…")
y_pred_scaled = model.predict(test_ds).flatten()
y_pred_actual = y_pred_scaled * y_std + y_mean
y_test_actual = y_test        * y_std + y_mean

mae  = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print(f"  MAE  : ${mae:.4f} (×$100,000)")
print(f"  RMSE : ${rmse:.4f} (×$100,000)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("MSE Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()

axes[1].plot(history.history["mae"],     label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("MAE over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()

plt.tight_layout()
plt.savefig("multimodal_training_curves.png", dpi=120)
plt.close()
print("\nTraining curves saved → 'multimodal_training_curves.png'")


plt.figure(figsize=(7, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.4, s=10, color="steelblue")
lims = [min(y_test_actual.min(), y_pred_actual.min()),
        max(y_test_actual.max(), y_pred_actual.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Price (×$100k)")
plt.ylabel("Predicted Price (×$100k)")
plt.title("Multimodal Model: Actual vs Predicted Housing Prices")
plt.legend()
plt.tight_layout()
plt.savefig("multimodal_predictions.png", dpi=120)
plt.close()
print("Prediction plot saved → 'multimodal_predictions.png'")

model.save("multimodal_housing_model.keras")
print("\nModel saved → 'multimodal_housing_model.keras'")