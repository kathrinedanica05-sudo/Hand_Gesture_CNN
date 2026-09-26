import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("hand_gesture_model.keras")

# Gesture names
gestures = ["fist", "palm", "peace", "thumbs_up"]

# Ask for image path
image_path = input("Enter image path: ").strip().strip('"')

# Open image
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print("Error: Image not found!")
    print("Please check the image path.")
    exit()

# Resize image
image = image.resize((128, 128))

# Convert image to array
img = np.array(image)

# MobileNetV2 preprocessing
img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img, verbose=0)

# Get predicted class
index = np.argmax(prediction)
gesture = gestures[index]

# Get confidence
confidence = prediction[0][index] * 100

# Display result
print()
print("================================")
print("Predicted Gesture:", gesture.upper())
print("Confidence:", round(confidence, 2), "%")
print("================================")