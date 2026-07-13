import os
import itertools
import numpy as np
import keras
from keras import layers
from keras.applications import EfficientNetB0
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# =========================
# 1. LOAD DATASET
# =========================

image_size = (224, 224)
batch_size = 16

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "OrganizedImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)


# =========================
# 2. DATA AUGMENTATION
# =========================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)


# =========================
# 3. IMPROVE PERFORMANCE
# =========================

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


# =========================
# 4. BUILD EFFICIENTNET MODEL
# =========================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=image_size + (3,),
)

# Παγώνουμε το pretrained EfficientNetB0.
# Δηλαδή στην αρχή δεν αλλάζουμε τα ήδη εκπαιδευμένα βάρη του.
base_model.trainable = False

inputs = keras.Input(shape=image_size + (3,))

# Data augmentation
x = data_augmentation(inputs)

# Το EfficientNetB0 του Keras έχει ήδη preprocessing/rescaling.
# Άρα εδώ ΔΕΝ βάζουμε layers.Rescaling(1./255).
x = base_model(x, training=False)

# Classification head για τις 7 κλάσεις του HAM10000
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()


# =========================
# 5. TRAIN MODEL
# =========================

epochs = 3

os.makedirs("model", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "model/efficientnet_save_at_{epoch}.keras"
    ),
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)


# =========================
# 6. SAVE TRAINING PLOTS
# =========================

os.makedirs("plots", exist_ok=True)

epochs_range = range(1, len(history.history["accuracy"]) + 1)

# Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(
    epochs_range,
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)
plt.plot(
    epochs_range,
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)
plt.title("EfficientNetB0 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("plots/efficientnet_accuracy_plot.png")
plt.close()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(
    epochs_range,
    history.history["loss"],
    marker="o",
    label="Training Loss"
)
plt.plot(
    epochs_range,
    history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)
plt.title("EfficientNetB0 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("plots/efficientnet_loss_plot.png")
plt.close()

print("EfficientNet training plots saved successfully.")


# =========================
# 7. CONFUSION MATRIX
# =========================

print("Generating EfficientNet confusion matrix...")

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("EfficientNetB0 Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

threshold = cm.max() / 2

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > threshold else "black"
    )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("plots/efficientnet_confusion_matrix.png")
plt.close()

print("EfficientNet confusion matrix saved successfully.")


# =========================
# 8. SAVE FINAL MODEL
# =========================

model.save("model/efficientnet_skin_model.keras")

print("EfficientNet model saved at model/efficientnet_skin_model.keras")
print("EfficientNet training finished.")