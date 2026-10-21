"""
Author: Leander Thielemann
Date: 2024-07-05

Beschreibung:
Für ein Bild "bild_1.jpg" mit einer Datei "data_bild_1.jpg" erstellt der Code "4_bounding_boxes.py" Bounding Boxen für das bild.
"""

import os
import cv2
import json

# Pfad zum Ordner mit den Bildern und JSON-Daten
json_file_path = 'data_bild_1.json'

# Laden der JSON-Daten
with open(json_file_path, 'r') as json_file:
    all_data = json.load(json_file)

# Funktion zur Erstellung der Bounding Boxes auf den Bildern
def create_images_with_bounding_boxes(data):
    for entry in data:
        bild_name = 'bild_1.jpg'
        hintergrund = cv2.imread(bild_name, cv2.IMREAD_UNCHANGED)  # Bild laden, inkl. Alpha-Kanal falls vorhanden

        # Laden der Bounding Box Daten für jedes Bild
        klammer_data = entry['klammer_data']

        for klammer_info in klammer_data:
            x_min = klammer_info['x_min']  # linke obere Ecke der Bounding Box (x-Koordinate)
            y_min = klammer_info['y_min']  # linke obere Ecke der Bounding Box (y-Koordinate)
            x_max = klammer_info['x_max']  # rechte untere Ecke der Bounding Box (x-Koordinate)
            y_max = klammer_info['y_max']  # rechte untere Ecke der Bounding Box (y-Koordinate)
            color = klammer_info['color']  # Farbe der Büroklammer

            # Farbe und Linienstärke der Bounding Box je nach Farbe der Büroklammer festlegen
            if color == 'rote':
                box_color = (0, 0, 255)  # Rot
            elif color == 'gelbe':
                box_color = (0, 255, 255)  # Gelb
            elif color == 'gruene':
                box_color = (0, 255, 0)  # Grün
            elif color == 'blaue':
                box_color = (255, 0, 0)  # Blau
            elif color == 'orange':
                box_color = (0, 127, 255)  # Orange
            elif color == 'pinke':
                box_color = (255, 0, 255)  # Pink

            # Zeichnen der Bounding Box auf dem Bild
            cv2.rectangle(hintergrund, (x_min, y_min), (x_max, y_max), box_color, 1)

        # Speichern des Bildes mit Bounding Boxes
        output_path = f'bounding_{bild_name}'
        cv2.imwrite(output_path, hintergrund)  # Bild mit Bounding Boxes speichern
        print(f'Bild mit Bounding Boxes gespeichert: {output_path}')

# Aufruf der Funktion zur Erstellung der Bilder mit Bounding Boxes
create_images_with_bounding_boxes(all_data)

print('Alle Bilder mit Bounding Boxes erstellt und gespeichert.')