import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 40

# ==============================
# LOAD DATASET
# ==============================

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=False
)

class_names = train_data.class_names

print("\n===================================")
print("CLASS ORDER")
print("===================================")
print(class_names)

if class_names != ["fist", "palm", "peace", "thumbs_up"]:
    raise ValueError(
        "Wrong class order: " + str(class_names)
    )

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.10)
])

# ==============================
# MOBILE NET V2
# ==============================

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

# IMPORTANT:
# Keep MobileNetV2 frozen.
# Do NOT fine-tune with this small dataset.

base_model.trainable = False

# ==============================
# BUILD MODEL
# ==============================

inputs = layers.Input(shape=(128, 128, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    4,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# ==============================
# TRAIN
# ==============================

print("\n===================================")
print("STARTING TRAINING")
print("===================================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)

# ==============================
# SAVE MODEL
# ==============================

model.save("hand_gesture_model.keras")

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print("Classes:", class_names)

print("\nModel saved as:")
print("hand_gesture_model.keras")

print("===================================")