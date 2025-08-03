# Emotion Detection with Cognivision AI 🎭🧠

A real-time facial emotion recognition system that leverages deep learning (CNNs) and computer vision (YOLO, Haar Cascade) to detect and classify human emotions through webcam or uploaded images. Built with Cognitive AI principles for context-aware emotion analysis.

## 🚀 Features

- 🎥 Real-time emotion detection via webcam
- 🖼️ Emotion recognition from uploaded images
- 🤖 Face detection using YOLO and Haar Cascade (fallback)
- 🧠 Multiple CNN models integrated:
  - LeNet
  - VGG16
  - ResNet50
  - InceptionV3 (GoogleNet)
  - MobileNetV2
  - EfficientNetB0
- 🧪 Evaluation based on accuracy, training time, and memory usage
- 📊 Automatic model selection based on defined metrics
- 🌐 Flask web interface for smooth interaction

## 📁 Project Structure

├── app.py # Flask web application
├── evaluation_model.py # Training, evaluating and selecting best model
├── model.py # CNN model definitions and model loader
├── uploads/ # Folder for uploaded images
├── templates/
│ └── index.html # Frontend UI
├── static/
│ └── styles.css # Optional: custom styles
└── models/ # Trained models saved here

markdown
Copy
Edit

## 🧠 Cognitive AI Aspect

Cognivision AI brings in contextual understanding:
- Models trained to recognize subtle facial cues
- Enhanced real-time responsiveness
- Adaptable to lighting, occlusion, and real-world noise

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow / Keras
- **Face Detection**: YOLOv3, Haar Cascade
- **Frontend**: HTML5, CSS3 (via Jinja Templates)
- **Others**: OpenCV, NumPy, Matplotlib

## 📊 Emotions Detected

- 😃 Happy  
- 😢 Sad  
- 😠 Angry  
- 😨 Fear  
- 😮 Surprise  
- 😐 Neutral  
- 😖 Disgust

## 📸 Sample Screenshots

> Add screenshots or a demo video link here if available.

## 🧪 Model Evaluation

| Model        | Accuracy | Training Time | Memory Usage |
|--------------|----------|----------------|----------------|
| LeNet        | 78.5%    | Low            | Low            |
| VGG16        | 84.2%    | Medium         | High           |
| ResNet50     | 86.1%    | High           | High           |
| InceptionV3  | 88.7%    | High           | Medium         |
| MobileNetV2  | 83.9%    | Fast           | Low            |
| EfficientNetB0 | 87.5%  | Medium         | Medium         |
