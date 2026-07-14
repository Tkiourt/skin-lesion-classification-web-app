import tensorflow as tf

# Path του εκπαιδευμένου Custom CNN model
keras_model_path = "model/skin_model.keras"

# Φορτώνουμε το Keras model
model = tf.keras.models.load_model(keras_model_path)

# Μετατροπή σε TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional optimization για μικρότερο μέγεθος
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Αποθήκευση του TFLite model
output_path = "model/skin_model.tflite"

with open(output_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved at:", output_path)