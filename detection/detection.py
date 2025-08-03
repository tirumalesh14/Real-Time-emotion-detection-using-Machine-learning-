import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_emotion(image_path, model_path):
    # Load the selected model
    model = load_model(model_path)

    # Load image
    img = cv2.imread(image_path)

    # ✅ Check if image was loaded correctly
    if img is None:
        print("❌ Error: Image not found or couldn't be read!")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        print("⚠️ No face detected in the image.")
        return
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match model input

        # Normalize based on model activation function
        face = face.astype("float32") / 255.0  # ✅ Works for ReLU-based models
        # face = (face.astype("float32") - 0.5) * 2  # ✅ Use for tanh-based models like LeNet
        
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension

        # Predict emotion
        prediction = model.predict(face)
        emotion = labels[np.argmax(prediction)]
        
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ✅ Fixing the file path issue
image_path = r"C:/Users/tirum/OneDrive/Desktop/WIN_20250325_14_56_35_Pro.jpg"

# ✅ Test with different models
detect_emotion(image_path, "resnet_emotion_model.h5")  # Change this to test different models
