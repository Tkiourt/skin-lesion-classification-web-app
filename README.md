# Skin Lesion Classification Web App

This project is an educational deep learning web application for skin lesion image classification using the HAM10000 dataset.

The application allows the user to upload a skin lesion image and choose between different CNN-based models for prediction.

## Models Included

The project compares four different models:

- Custom CNN
- EfficientNetB0
- MobileNetV2
- ResNet50

The Custom CNN was built and trained from scratch, while EfficientNetB0, MobileNetV2 and ResNet50 were implemented using transfer learning from Keras Applications with pretrained ImageNet weights.

## Main Features

- Skin lesion image upload
- Model selection from dropdown menu
- Prediction result
- Confidence score
- Class probability distribution
- Accuracy and loss plots
- Confusion matrix evaluation
- Model comparison report page

## Dataset

The project is based on the HAM10000 dataset, which contains dermatoscopic images of skin lesions across seven classes:

- akiec: Actinic keratoses
- bcc: Basal cell carcinoma
- bkl: Benign keratosis-like lesion
- df: Dermatofibroma
- mel: Melanoma
- nv: Melanocytic nevus
- vasc: Vascular lesion

## Project Structure

```text
SkinCNN_WebApp/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── skin_model.keras
│   ├── efficientnet_skin_model.keras
│   ├── mobilenet_skin_model.keras
│   └── resnet50_skin_model.keras
├── static/
│   ├── style.css
│   ├── uploads/
│   └── plots/
├── templates/
│   ├── index.html
│   └── model_report.html
├── Ham10000.py
├── Ham10000_EfficientNet.py
├── Ham10000_MobileNetV2.py
└── Ham10000_ResNet50.py


## Docker Deployment

This project includes a Dockerfile, so it can be deployed on a server using Docker.

### Build the Docker image

```bash
docker build -t skin-cnn-app .

pip install -r requirements.txt
python app.py
http://127.0.0.1:5000
http://127.0.0.1:5000/model-report

This project is for educational and research purposes only.
It does not provide medical diagnosis and should not be used as a replacement for professional medical advice.