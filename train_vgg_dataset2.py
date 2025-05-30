import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Define paths and parameters
dataset_dir = 'dataset2'
batch_size = 16
IMG_HEIGHT, IMG_WIDTH = 224, 224
epochs = 18

# Data preparation with image processing
image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # set validation split
)

train_data_gen = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=dataset_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    subset='training'  # set as training data
)

validation_data_gen = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=dataset_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    subset='validation'  # set as validation data
)

# Load the VGG16 network with local weights
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # path to your local weights file
baseModel = VGG16(weights=weights_path, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Construct the head of the model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(train_data_gen.class_indices), activation="softmax")(headModel)

# Place the head FC model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers in the base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=epochs,
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.samples // batch_size
)

# Save the model
model.save('model_vgg_dataset2.h5')

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the test set
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(validation_data_gen, verbose=2)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
print("Generating predictions...")
predictions = model.predict(validation_data_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_data_gen.classes

# Generate classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=train_data_gen.class_indices.keys()))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=train_data_gen.class_indices, yticklabels=train_data_gen.class_indices)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training metrics
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
