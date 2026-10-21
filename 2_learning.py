"""
Author: Leander Thielemann
Date: 2024-07-05

Beschreibung:
Algoritmus "2_learning.py" ausführen um die Anzahl, Farbe und (Position) von Büroklammern zu trainieren
"""

# import libarys:
import numpy as np
import cv2
import os
import json
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Define the label mapping for the 7 classes
label_mapping = ['pinke', 'blaue', 'orange', 'rote', 'gelbe', 'gruene', 'total']


# Augmentation function to generate additional images for training
def augment_image(image):
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Flip horizontally
    augmented_images.append(cv2.flip(image, 1))

    # Flip vertically
    augmented_images.append(cv2.flip(image, 0))

    # Rotate 90 degrees clockwise
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    # Rotate 180 degrees
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))

    # Rotate 270 degrees clockwise
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # Add Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    augmented_images.append(cv2.add(image, noise))

    # Scale image
    scale_percent = random.uniform(0.8, 1.2)
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    scaled_image = cv2.resize(image, (width, height))
    scaled_image = cv2.resize(scaled_image, (image.shape[1], image.shape[0]))
    augmented_images.append(scaled_image)

    # Random Crop
    crop_size = random.randint(90, 128)
    x_start = random.randint(0, 128 - crop_size)
    y_start = random.randint(0, 128 - crop_size)
    cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size]
    cropped_image = cv2.resize(cropped_image, (128, 128))
    augmented_images.append(cropped_image)

    # Brightness Adjustment using HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = random.uniform(0.5, 1.5)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], brightness)
    bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(bright_image)

    return augmented_images


# Preprocessing function for images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)  # Histogram equalization for enhancing contrast
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)  # Gaussian blur for noise reduction
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    image_enhanced = cv2.addWeighted(edges, 1.5, gray, -0.5, 0)  # Enhanced image using weighted sum
    image_resized = cv2.resize(image_enhanced, (128, 128))  # Resize to a fixed size
    return image_resized


# Function to plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


# Function to load labels from a JSON file
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels_list = json.load(f)
    return labels_list


# Function to save labels to a JSON file
def save_labels(labels_list, labels_path):
    with open(labels_path, 'w') as f:
        json.dump(labels_list, f)


# Function to load dataset images and corresponding labels
def load_dataset(dataset_path, labels_path):
    with open(labels_path, 'r') as f:
        labels_list = json.load(f)

    X = []
    y = []
    for item in labels_list:
        image_name = item['bild']
        label_counts = item['klammer_count']
        image_path = os.path.join(dataset_path, image_name)
        if os.path.exists(image_path):
            image = preprocess_image(image_path)
            X.append(image)
            y.append([label_counts.get(label, 0) for label in label_mapping])  # Extract label counts for each class

    X = np.array(X).reshape(-1, 128, 128, 1)  # Reshape images for CNN input
    y = np.array(y)
    return X, y


# Function to augment dataset images and save augmented images
def augment_dataset(dataset_path, labels_path, augmented_dataset_path):
    labels_list = load_labels(labels_path)
    augmented_labels_list = []

    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path)

    total_images = len(labels_list)
    processed_images = 0

    for item in labels_list:
        image_name = item['bild']
        label_counts = item['klammer_count']
        image_path = os.path.join(dataset_path, image_name)

        if os.path.exists(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure to load the image in color mode
            augmented_images = augment_image(image)

            for i, aug_img in enumerate(augmented_images):
                new_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.png"
                new_image_path = os.path.join(augmented_dataset_path, new_image_name)
                cv2.imwrite(new_image_path, aug_img)

                new_item = {
                    'bild': new_image_name,
                    'klammer_count': label_counts
                }
                augmented_labels_list.append(new_item)

        processed_images += 1
        print(f'Processed {processed_images}/{total_images} images...', end='\r')

    print('Augmentation completed.')
    save_labels(augmented_labels_list, os.path.join(augmented_dataset_path, 'all_data.json'))


# Function to define the CNN model architecture
def cnn_model_amount(input_shape, num_outputs):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_outputs, activation='linear'))  # Output layer with linear activation for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# Function to calculate accuracy within a specified tolerance
def accuracy_within_tolerance(y_true, y_pred, tolerance=1):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)


# Main function to orchestrate the data processing, model training, and evaluation
def main():
    print("Loading dataset...")
    dataset_path = 'raw_dataset'
    labels_path = 'raw_dataset/all_data.json'
    augmented_dataset_path = 'augmented_dataset'
    augmented_labels_path = 'augmented_dataset/all_data.json'

    # Augment dataset
    augment_dataset(dataset_path, labels_path, augmented_dataset_path)

    # Load augmented dataset
    X, y = load_dataset(augmented_dataset_path, augmented_labels_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (128, 128, 1)
    num_outputs = len(label_mapping)

    print("Creating model...")
    model = cnn_model_amount(input_shape, num_outputs)
    print("Model created.\n")

    print("Starting training.")

    # Callbacks for early stopping, learning rate reduction, and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Plot training and validation loss
    plot_loss(history)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    accuracy_amount = accuracy_within_tolerance(y_test, y_pred)
    tolerance = 1

    # Accuracy for each class
    for i in range(num_outputs):
        class_mae = mae[i]
        class_mse = mse[i]
        class_r2 = r2[i]
        class_accuracy = accuracy_within_tolerance(y_test[:, i], y_pred[:, i])
        accuracy_in_1 = accuracy_within_tolerance(y_test, y_pred, tolerance)
        print(
            f"{label_mapping[i]}: "
            f"mae={round(class_mae, 2)}, "
            f"mse={round(class_mse, 2)}, "
            f"R^2={round(float(class_r2), 2)}, "
            f"accuracy={round(class_accuracy * 100, 2)}%, "
            f"accuracy_in_1={round(accuracy_in_1 * 100, 2)}% ")
    print(f"\nOverall Accuracy: {round(accuracy_amount * 100, 2)}%\n")

    # Save the final trained model
    model.save('model.keras')


if __name__ == "__main__":
    main()
