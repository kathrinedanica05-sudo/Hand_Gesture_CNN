import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ==============================
# MODEL
# ==============================

MODEL_PATH = "hand_gesture_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT:
# This order must match train.py
gestures = [
    "fist",
    "palm",
    "peace",
    "thumbs_up"
]

# ==============================
# CHECK DATASET
# ==============================

DATASET_PATH = "dataset"

print()
print("======================================")
print("MODEL DATASET CHECK")
print("======================================")

for folder_index, gesture in enumerate(gestures):

    folder = os.path.join(DATASET_PATH, gesture)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print()
    print("Checking folder:", gesture)
    print("Images found:", len(files))

    correct = 0
    total = 0

    probabilities = []

    for filename in files:

        path = os.path.join(folder, filename)

        try:

            # Open image
            image = Image.open(path).convert("RGB")

            # Resize
            image = image.resize((128, 128))

            # Convert to NumPy
            image = np.array(image)

            # IMPORTANT:
            # DO NOT use preprocess_input here.
            #
            # preprocess_input is already INSIDE
            # our trained MobileNetV2 model.

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            # Prediction
            prediction = model.predict(
                image,
                verbose=0
            )[0]

            predicted_index = np.argmax(prediction)

            probabilities.append(prediction)

            if predicted_index == folder_index:
                correct += 1

            total += 1

        except Exception as e:

            print(
                "Error:",
                filename,
                e
            )

    # ==============================
    # RESULTS
    # ==============================

    if total > 0:

        average_prediction = np.mean(
            probabilities,
            axis=0
        )

        accuracy = (
            correct / total
        ) * 100

        print(
            "Expected:",
            gesture.upper()
        )

        print(
            f"Predicted correctly: "
            f"{correct}/{total}"
        )

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

        print(
            "Average probabilities:"
        )

        for i in range(4):

            print(
                f"  {gestures[i].upper():10} "
                f"{average_prediction[i] * 100:.2f}%"
            )


print()
print("======================================")
print("CHECK COMPLETE")
print("======================================")