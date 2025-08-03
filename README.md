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

## 🧪 Model Evaluation

| Model        | Accuracy | Training Time | Memory Usage |
|--------------|----------|----------------|----------------|
| LeNet        | 78.5%    | Low            | Low            |
| VGG16        | 84.2%    | Medium         | High           |
| ResNet50     | 86.1%    | High           | High           |
| InceptionV3  | 88.7%    | High           | Medium         |
| MobileNetV2  | 83.9%    | Fast           | Low            |
| EfficientNetB0 | 87.5%  | Medium         | Medium         |
