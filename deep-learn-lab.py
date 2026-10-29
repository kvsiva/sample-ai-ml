
# ============================================================
# Deep Learning Application: Image Classification using CNN
# Dataset: CIFAR-10
# Framework: TensorFlow / Keras
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10


# ------------------------------------------------------------
# Step 1: Define Class Names
# ------------------------------------------------------------

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# ------------------------------------------------------------
# Step 2: Load the Dataset
# ------------------------------------------------------------

print("Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# ------------------------------------------------------------
# Step 3: Preprocess the Dataset
# ------------------------------------------------------------

print("\nPreprocessing images...")

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ------------------------------------------------------------
# Step 4: Build the CNN Deep Learning Model
# ------------------------------------------------------------

print("\nBuilding CNN model...")

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # First Convolution Layer
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third Convolution Layer
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Convert Feature Maps into Vector
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        64,
        activation="relu"
    ),

    # Output Layer
    layers.Dense(
        10,
        activation="softmax"
    )
])

model.summary()


# ------------------------------------------------------------
# Step 5: Compile the Model
# ------------------------------------------------------------

print("\nCompiling model...")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# Step 6: Train the Model
# ------------------------------------------------------------

print("\nTraining the Deep Learning model...")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


# ------------------------------------------------------------
# Step 7: Evaluate the Model
# ------------------------------------------------------------

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=2
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", f"{test_accuracy:.2%}")


# ------------------------------------------------------------
# Step 8: Predict an Image
# ------------------------------------------------------------

print("\nPerforming image classification...")

# Select an image from the test dataset
image_index = 25

image = x_test[image_index]

# Prepare image for prediction
input_image = np.expand_dims(image, axis=0)

# Generate prediction probabilities
prediction = model.predict(
    input_image,
    verbose=0
)

# Identify predicted class
predicted_class = np.argmax(prediction[0])

# Calculate confidence
confidence = prediction[0][predicted_class]

# Actual class
actual_class = int(y_test[image_index][0])

print("\nImage Classification Result")
print("---------------------------")

print("Actual Class:", class_names[actual_class])
print("Predicted Class:", class_names[predicted_class])
print("Confidence:", f"{confidence:.2%}")


# ------------------------------------------------------------
# Step 9: Visualize the Prediction
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(image)

plt.title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Confidence: {confidence:.2%}"
)

plt.xlabel(
    f"Actual Class: {class_names[actual_class]}"
)

plt.axis("off")

plt.show()


# ------------------------------------------------------------
# Step 10: Visualize Training and Validation Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# Application Completed
# ------------------------------------------------------------

print("\nDeep Learning Image Classification Completed!")
