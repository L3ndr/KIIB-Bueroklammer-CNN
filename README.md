# Büroklammern zählen mit einem CNN

Hausarbeit Nr. 4 – Informatik 2 (KIIB)

Author: Leander Thielemann
Date: 05.07.2024

## Worum geht es?

Ziel der Aufgabe war ein neuronales Netz, das auf einem Bild die Büroklammern erkennt und
zählt – und zwar getrennt nach Farbe. Zusätzlich sollen um die Klammern Bounding Boxen
gezeichnet werden können.

Umgesetzt ist das Ganze in Python mit Keras/TensorFlow und OpenCV. Das Projekt besteht aus
vier Skripten, die nacheinander ausgeführt werden:

1. Datensatz erzeugen
2. Netz trainieren
3. Vorhersage für ein eigenes Bild
4. Bounding Boxen einzeichnen

Erkannt werden sechs Farben: rot, gelb, grün, blau, orange und pink.

## Installation

Benötigt wird Python 3 mit folgenden Paketen:

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
```

Alle Skripte gehen davon aus, dass sie aus dem Projektordner heraus gestartet werden, weil die
Pfade relativ angegeben sind.

## Ablauf im Detail

### 1. Datensatz erzeugen – `1_create_dataset.py`

Statt hunderte Fotos von Hand zu machen und zu beschriften, werden die Trainingsbilder
synthetisch erzeugt. Dafür wird ein zufälliges Hintergrundbild aus `Hintergruende/` genommen
und darauf 10 bis 30 Büroklammern aus `Klammern/` verteilt. Jede Klammer wird dabei

- zufällig skaliert (Faktor 0,3 bis 0,55),
- zufällig gedreht (−90° bis 90°) und
- an eine zufällige Position gesetzt (über den Alpha-Kanal mit dem Hintergrund gemischt).

Anschließend wird das Bild auf 420 × 315 Pixel verkleinert und als JPEG gespeichert. Weil beim
Erzeugen bekannt ist, wo welche Klammer liegt, bekommt man die Labels gratis dazu: Für jedes
Bild landen die Anzahl pro Farbe, die Gesamtzahl und die Bounding Box jeder einzelnen Klammer
in `raw_dataset/all_data.json`.

Standardmäßig werden 1000 Bilder erzeugt (Variable `n` im Skript).

```bash
python 1_create_dataset.py
```

### 2. Training – `2_learning.py`

Das Skript macht drei Dinge:

**Augmentierung.** Aus jedem Bild werden 10 Varianten erzeugt (Original, gespiegelt,
um 90°/180°/270° gedreht, mit Rauschen, skaliert, zufällig zugeschnitten, mit veränderter
Helligkeit). Aus 1000 Bildern werden so 10.000. Die Ergebnisse liegen danach in
`augmented_dataset/`.

**Vorverarbeitung.** Jedes Bild wird in Graustufen umgewandelt, per Histogrammausgleich im
Kontrast verbessert, leicht weichgezeichnet und mit einer Canny-Kantenerkennung kombiniert.
Das Ergebnis wird auf 128 × 128 Pixel skaliert und so ins Netz gegeben.

**Netz.** Das CNN besteht aus drei Faltungsblöcken (64, 128 und 256 Filter, jeweils mit
Batch-Normalization, Max-Pooling und Dropout), danach eine Dense-Schicht mit 512 Neuronen und
eine Ausgabeschicht mit 7 Werten: die Anzahl pro Farbe plus die Gesamtzahl. Das Zählen wird
also als Regressionsproblem behandelt (Loss: Mean Squared Error, Optimizer: Adam).

Trainiert wird mit einem 80/20-Split, maximal 25 Epochen und Early Stopping. Nach dem
Training wird der Loss-Verlauf geplottet und für jede Klasse MAE, MSE, R² und eine Genauigkeit
mit Toleranz ±1 Klammer ausgegeben.

```bash
python 2_learning.py
```

Gespeichert werden `best_model.keras` (bestes Modell laut Validierungs-Loss) und
`model.keras` (Endstand), das von `3_predict.py` geladen wird.

### 3. Vorhersage – `3_predict.py`

Ein eigenes Bild als `bild.jpg` in den Projektordner legen und das Skript starten:

```bash
python 3_predict.py
```

Ausgegeben wird die geschätzte Anzahl pro Farbe sowie die Gesamtzahl, z. B.:

```
pinke: 2
blaue: 5
orange: 3
rote: 2
gelbe: 3
gruene: 2
total: 17
```

### 4. Bounding Boxen – `4_bounding_boxes.py`

Liest die Annotationen aus `data_bild_1.json` und zeichnet um jede Klammer in `bild_1.jpg` ein
Rechteck in der Farbe der jeweiligen Klammer. Das Ergebnis wird als `bounding_bild_1.jpg`
gespeichert.

```bash
python 4_bounding_boxes.py
```

## Hinweise

- `augmented_dataset/` wird beim Training jedes Mal neu geschrieben und ist ca. zehnmal so groß
  wie `raw_dataset/`.
- Da das Netz nur auf synthetischen Bildern trainiert wurde, hängt die Qualität bei echten
  Fotos stark davon ab, wie ähnlich Hintergrund, Beleuchtung und Klammern den
  Trainingsdaten sind.
