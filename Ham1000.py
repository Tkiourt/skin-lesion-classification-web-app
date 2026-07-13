import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# HAM10000 classes:
# akiec = Actinic keratoses / Bowen's disease = ακτινική κεράτωση / Bowen
# bcc   = Basal cell carcinoma = βασικοκυτταρικό καρκίνωμα
# bkl   = Benign keratosis-like lesion = καλοήθης κεράτωση
# df    = Dermatofibroma = δερματοΐνωμα
# mel   = Melanoma = μελάνωμα
# nv    = Melanocytic nevus = σπίλος / ελιά
# vasc  = Vascular lesion = αγγειακή βλάβη


# =========================
# 1. LOAD DATASET
# =========================

image_size = (180, 180)
batch_size = 32

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
# 2. SHOW SAMPLE IMAGES
# =========================

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")

plt.tight_layout()
plt.close()


# =========================
# 3. DATA AUGMENTATION
# =========================

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[i]).astype("uint8"))
        plt.axis("off")

plt.tight_layout()
plt.close()


# =========================
# 4. IMPROVE PERFORMANCE
# =========================

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


# =========================
# 5. BUILD MODEL
# =========================

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Normalize pixels from 0-255 to 0-1
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )

        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # 7 categories -> softmax
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
model.summary()


# =========================
# 6. TRAIN MODEL
# =========================

epochs = 3

os.makedirs("model", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint("model/ham10000_save_at_{epoch}.keras"),
]

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
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
# 7. SAVE TRAINING PLOTS
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
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("plots/accuracy_plot.png")
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
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("plots/loss_plot.png")
plt.close()

print("Training plots saved successfully in plots folder.")


# =========================
# 8. CONFUSION MATRIX
# =========================

print("Generating confusion matrix...")

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
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
plt.savefig("plots/confusion_matrix.png")
plt.close()

print("Confusion matrix saved at plots/confusion_matrix.png")


# =========================
# 9. SAVE FINAL MODEL
# =========================

model.save("model/skin_model.keras")

print("Model saved successfully at model/skin_model.keras")
print("Training finished.")