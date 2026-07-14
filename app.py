import os
import numpy as np
import keras
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# =========================
# 1. FLASK APP SETUP
# =========================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# 2. LOAD TRAINED MODELS
# =========================

loaded_models ={}

def get_model(model_choice):
    if model_choice in loaded_models:
        return loaded_models[model_choice]
    
    if model_choice == "efficientnet":
        model = keras.models.load_model("model/efficientnet_skin_model.keras")
    elif model_choice == "mobilenet":
        model = keras.models.load_model("model/mobilenet_skin_model.keras")
    elif model_choice == "resnet50":
        model = keras.models.load_model("model/resnet50_skin_model.keras")
    else:
        model = keras.models.load_model("model/skin_model.keras")

    loaded_models[model_choice]=model
    return model


# =========================
# 3. CLASS NAMES
# =========================

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

class_labels = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion"
}


# =========================
# 4. IMAGE PREDICTION FUNCTION
# =========================

def predict_skin_image(image_path, model_choice):
    """
    Παίρνει την εικόνα και το μοντέλο που διάλεξε ο χρήστης.
    Αν διάλεξε Custom CNN -> χρησιμοποιεί image size 180x180.
    Αν διάλεξε EfficientNetB0 -> χρησιμοποιεί image size 224x224.
    """

    if model_choice == "efficientnet":
       image_size = (224, 224)
       selected_model = "EfficientNetB0"
    elif model_choice == "mobilenet":
       image_size = (224, 224)
       selected_model = "MobileNetV2"
    elif model_choice == "resnet50":
       image_size = (224, 224)
       selected_model = "ResNet50"
    else:
       image_size = (180, 180)
       selected_model = "Custom CNN"

    model = get_model(model_choice)
     
    # Φόρτωση εικόνας στο σωστό μέγεθος για το αντίστοιχο μοντέλο
    img = keras.utils.load_img(image_path, target_size=image_size)

    # Μετατροπή εικόνας σε NumPy array
    img_array = keras.utils.img_to_array(img)

    # Προσθήκη batch dimension: (1, height, width, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array, verbose=0)

    # Βρίσκουμε την κλάση με τη μεγαλύτερη πιθανότητα
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = 100 * np.max(predictions[0])

    # Μετατροπή short class name σε κανονικό label
    prediction_label = class_labels[predicted_class]

    # Όλες οι πιθανότητες
    probabilities = []

    for i, class_name in enumerate(class_names):
        probabilities.append({
            "label": class_labels[class_name],
            "probability": round(100 * float(predictions[0][i]), 2)
        })

    # Ταξινόμηση πιθανοτήτων από μεγαλύτερη σε μικρότερη
    probabilities = sorted(
        probabilities,
        key=lambda x: x["probability"],
        reverse=True
    )

    return prediction_label, round(confidence, 2), probabilities, selected_model


# =========================
# 5. MAIN ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    filename = None
    prediction = None
    confidence = None
    probabilities = None
    selected_model = None

    if request.method == "POST":
        file = request.files["skin_image"]

        # Παίρνουμε το μοντέλο που επέλεξε ο χρήστης από το dropdown
        model_choice = request.form.get("model_choice", "custom")

        if file.filename == "":
            message = "No image selected."
        else:
            filename = secure_filename(file.filename)

            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(save_path)

            message = "Image uploaded successfully: " + filename

            prediction, confidence, probabilities, selected_model = predict_skin_image(
                save_path,
                model_choice
            )

    return render_template(
        "index.html",
        message=message,
        filename=filename,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        selected_model=selected_model
    )


# =========================
# 6. MODEL REPORT ROUTE
# =========================

@app.route("/model-report")
def model_report():
    return render_template("model_report.html")


# =========================
# 7. RUN FLASK APP
# =========================

if __name__ == "__main__":
    app.run()