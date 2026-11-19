"""
Author: Leander Thielemann
Date: 2024-07-05

Beschreibung:
Erstelle zuerst einen Datasatz mit "create_dataset.py" (mindestens 1000 Trainingsdaten)
"""

import cv2
import random
import os
import json
from collections import Counter

from src import config


# Funktion zur Erstellung einer rotierten und skalierten Büroklammer
def create_rotated_buero_klammer(buero_klammer, scaling_range=(0.3, 0.55), rotation_range=(-90, 90)):
    scaling_factor = random.uniform(*scaling_range)  # Zufälliger Skalierungsfaktor
    buero_klammer_resized = cv2.resize(buero_klammer, None, fx=scaling_factor, fy=scaling_factor)  # Skalierung des Bildes
    rotation_angle = random.uniform(*rotation_range)  # Zufälliger Rotationswinkel
    rotation_matrix = cv2.getRotationMatrix2D((buero_klammer_resized.shape[1] / 2, buero_klammer_resized.shape[0] / 2),
                                              rotation_angle, 1)  # Rotationsmatrix berechnen
    buero_klammer_rotated = cv2.warpAffine(buero_klammer_resized, rotation_matrix,
                                           (buero_klammer_resized.shape[1], buero_klammer_resized.shape[0]))  # Rotiertes Bild erzeugen
    rotated_h, rotated_w, _ = buero_klammer_rotated.shape
    buero_klammer_rotated = buero_klammer_rotated[:, :rotated_h]  # Beschnitt, um rechteckige Form sicherzustellen
    return buero_klammer_rotated, rotation_angle  # Rotiertes Bild und Winkel zurückgeben


# Funktion zur Platzierung einer Büroklammer auf dem Hintergrundbild
def place_buero_klammer(hintergrund, buero_klammer_rotated):
    hintergrund_h, hintergrund_w, _ = hintergrund.shape
    rotated_h, rotated_w, _ = buero_klammer_rotated.shape
    random_position_y = random.randint(0, hintergrund_h - rotated_h)  # Zufällige Position in y-Richtung
    random_position_x = random.randint(0, hintergrund_w - rotated_w)  # Zufällige Position in x-Richtung

    for c in range(0, 3):
        alpha_mask = buero_klammer_rotated[:, :, 3] / 255.0  # Alpha-Maske extrahieren (falls vorhanden)
        hintergrund[random_position_y:random_position_y + rotated_h, random_position_x:random_position_x + rotated_w, c] = \
            alpha_mask * buero_klammer_rotated[:, :, c] + \
            (1 - alpha_mask) * hintergrund[random_position_y:random_position_y + rotated_h,
                               random_position_x:random_position_x + rotated_w, c]  # Mischen der Pixel

    return hintergrund, (random_position_x, random_position_y, rotated_w, rotated_h)  # Modifiziertes Bild und Position zurückgeben


# Funktion zur Erstellung der Bounding Boxes und Speichern der Eckpunkte
def create_bounding_boxes(hintergrund_resized, original_shape, klammer_positions):
    expand_by = 5  # Erweiterung für die Bounding Box
    klammer_data = []

    for buero_klammer_path, klammer_position in klammer_positions:
        x, y, w, h = klammer_position

        # Koordinaten der Bounding Box an die Größe des verkleinerten Hintergrundbildes anpassen
        x_resized = int(x * hintergrund_resized.shape[1] / original_shape[1])
        y_resized = int(y * hintergrund_resized.shape[0] / original_shape[0])
        w_resized = int(w * hintergrund_resized.shape[1] / original_shape[1])
        h_resized = int(h * hintergrund_resized.shape[0] / original_shape[0])

        # Bounding Box um expand_by erweitern und sicherstellen, dass sie innerhalb der Bildgrenzen bleibt
        x_resized -= expand_by
        y_resized -= expand_by
        w_resized += 2 * expand_by
        h_resized += 2 * expand_by

        if x_resized < 0:
            x_resized = 0
        if y_resized < 0:
            y_resized = 0
        if x_resized + w_resized > hintergrund_resized.shape[1]:
            w_resized = hintergrund_resized.shape[1] - x_resized
        if y_resized + h_resized > hintergrund_resized.shape[0]:
            h_resized = hintergrund_resized.shape[0] - y_resized

        # Eckpunkte der Bounding Box speichern
        x_min = x_resized
        y_min = y_resized
        x_max = x_resized + w_resized
        y_max = y_resized + h_resized
        klammer_color = os.path.basename(buero_klammer_path).split('-')[0]  # Farbe aus dem Dateinamen extrahieren
        klammer_data.append({
            'color': klammer_color,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        })

    return klammer_data  # Liste mit den Eckpunkten der Bounding Boxes zurückgeben


def main(n=1000):
    # Pfade zu den Hintergrundbildern und Büroklammern
    hintergrund_pfade = [str(config.BACKGROUNDS_DIR / 'Background_1.jpg'), str(config.BACKGROUNDS_DIR / 'Background_2.jpg'),
                         str(config.BACKGROUNDS_DIR / 'Background_3.jpg'), str(config.BACKGROUNDS_DIR / 'Background_4.jpg'),
                         str(config.BACKGROUNDS_DIR / 'Background_5.jpg')]

    # Dictionary mit den Pfaden und Farben der Büroklammern
    buero_klammern = {
        'rote': str(config.CLIPS_DIR / 'rote-bueroklammer.png'),
        'gelbe': str(config.CLIPS_DIR / 'gelbe-bueroklammer.png'),
        'gruene': str(config.CLIPS_DIR / 'gruene-bueroklammer.png'),
        'blaue': str(config.CLIPS_DIR / 'blaue-bueroklammer.png'),
        'orange': str(config.CLIPS_DIR / 'orange-bueroklammer.png'),
        'pinke': str(config.CLIPS_DIR / 'pinke-bueroklammer.png')
    }

    # Erstellen des Ordners für die generierten Bilder
    output_folder = config.RAW_DATASET_DIR
    os.makedirs(output_folder, exist_ok=True)

    # Liste für alle Daten
    all_data = []

    # Schleife zur Generierung der Bilder
    for i in range(n):
        hintergrund_path = random.choice(hintergrund_pfade)
        hintergrund = cv2.imread(hintergrund_path, cv2.IMREAD_UNCHANGED)

        num_buero_klammern = random.randint(10, 30)  # Zufällige Anzahl von Büroklammern
        klammer_count = Counter()
        klammer_positions = []

        for j in range(num_buero_klammern):
            klammer_color, buero_klammer_path = random.choice(list(buero_klammern.items()))
            buero_klammer = cv2.imread(buero_klammer_path, cv2.IMREAD_UNCHANGED)
            buero_klammer_rotated, rotation_angle = create_rotated_buero_klammer(buero_klammer)
            hintergrund, klammer_position = place_buero_klammer(hintergrund, buero_klammer_rotated)
            klammer_count[klammer_color] += 1
            klammer_positions.append((buero_klammer_path, klammer_position))  # Speichern der Position jeder Büroklammer

        # Gesamtanzahl der Büroklammern im Bild hinzufügen
        klammer_count['total'] = sum(klammer_count.values())

        # Bildpfad und Klammeranzahl zum DataFrame hinzufügen
        bild_name = f'bild_{i + 1}.jpg'  # .jpg statt .png für komprimierte Bilder
        output_path = os.path.join(output_folder, bild_name)

        # Bild auf 420x315 verkleinern
        hintergrund_resized = cv2.resize(hintergrund, (420, 315))

        # Bounding Boxes hinzufügen und Daten speichern
        klammer_data = create_bounding_boxes(hintergrund_resized.copy(), hintergrund.shape, klammer_positions)

        # Bild im JPEG-Format mit Kompression speichern
        cv2.imwrite(output_path, hintergrund_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f'Bild {i + 1} erstellt und gespeichert: {output_path}')
        print(f'Anzahl der Büroklammern in Bild {i + 1}: {klammer_count}')

        # Daten für JSON sammeln
        all_data.append({
            'bild': bild_name,
            'klammer_count': klammer_count,
            'klammer_data': klammer_data
        })

    # Speichern aller Daten in einer JSON-Datei
    json_output_path = os.path.join(output_folder, 'all_data.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_data, json_file, indent=4)

    print(f'Alle Daten in einer JSON-Datei gespeichert: {json_output_path}')


if __name__ == "__main__":
    main()
