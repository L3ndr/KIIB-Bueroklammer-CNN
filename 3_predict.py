"""
Author: Leander Thielemann
Date: 2024-07-05

Beschreibung:
Lege ein Bild in den Projektordner mit dem Namen "bild.jpg" und führe den Code "3_predict.py" aus
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Funktion zur Vorverarbeitung des Bildes (Wiederverwendung aus dem Trainierungsskript)
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Bild einlesen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # In Graustufen umwandeln
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian Blur anwenden
    edges = cv2.Canny(blurred, 50, 150)  # Kanten erkennen mit Canny Edge Detection
    image_enhanced = cv2.addWeighted(edges, 1.5, gray, -0.5, 0)  # Bild verbessern durch Kantenverstärkung
    image_resized = cv2.resize(image_enhanced, (128, 128))  # Bild auf Größe 128x128 verkleinern
    return image_resized


# Funktion zur Vorhersage der Anzahl der Büroklammern
def predict_clip_counts(image_path):
    image = preprocess_image(image_path)  # Bild vorverarbeiten
    image = np.expand_dims(image, axis=0).reshape(-1, 128, 128, 1)  # Dimensionen für das Modell anpassen (1 Bild, 128x128, 1 Kanal)
    prediction = model.predict(image)  # Vorhersage machen
    return prediction[0]  # Vorhersage zurückgeben (erste Dimension entfernen)


# Mapping der Labels für die Ausgabe
label_mapping = ['pinke', 'blaue', 'orange', 'rote', 'gelbe', 'gruene', 'total']

# Modell laden
model = load_model('model.keras')

# Beispiel für die Verwendung
image_path = 'bild.jpg'
prediction = predict_clip_counts(image_path)

# Ausgabe der Vorhersage für jedes Label
for label in range(len(label_mapping)):
    print(f'{label_mapping[label]}: {int(prediction[label])}')  # Labels und zugehörige Vorhersagen ausgeben
