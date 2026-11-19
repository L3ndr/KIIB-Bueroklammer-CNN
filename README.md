# Büroklammern zählen mit einem CNN

Hausarbeit Nr. 4 – Informatik 2 (KIIB)

Author: Leander Thielemann
Date: 05.07.2024

## Worum geht es?

Ziel der Aufgabe war ein neuronales Netz, das auf einem Bild die Büroklammern erkennt und
zählt – und zwar getrennt nach Farbe. Zusätzlich sollen um die Klammern Bounding Boxen
gezeichnet werden können.

Umgesetzt ist das Ganze in Python mit Keras/TensorFlow und OpenCV. Alle Schritte laufen über
einen gemeinsamen Einstiegspunkt (`src/main.py`) und werden nacheinander ausgeführt:

1. `generate` – Datensatz erzeugen
2. `train` – Netz trainieren
3. `predict` – Vorhersage für ein eigenes Bild
4. `bbox` – Bounding Boxen einzeichnen

Erkannt werden sechs Farben: rot, gelb, grün, blau, orange und pink.

## Projektstruktur

```
src/
  main.py            Einstiegspunkt (Unterbefehle generate/train/predict/bbox)
  config.py          Zentrale Pfade
  create_dataset.py  1. Datensatz erzeugen
  train.py           2. Training
  predict.py         3. Vorhersage
  bounding_boxes.py  4. Bounding Boxen
data/
  Hintergruende/     Hintergrundbilder (Eingabe)
  Klammern/          Büroklammer-PNGs mit Alpha-Kanal (Eingabe)
  Bounding/          Beispielbilder der Bounding Boxen
  evaluation/        Beispielbild mit Annotationen für "bbox"
  raw_dataset/       generiert von "generate"
  augmented_dataset/ generiert von "train"
  models/            trainierte Modelle
  output/            Loss-Plot, Bilder mit Bounding Boxen
```

## Ausführen mit Docker (empfohlen)

```bash
docker compose build
docker compose run --rm app generate        # optional: generate -n 2000
docker compose run --rm app train
docker compose run --rm app predict         # nimmt data/bild.jpg
docker compose run --rm app predict data/mein_bild.jpg
docker compose run --rm app bbox
```

`data/` und `src/` werden in den Container gemountet. Alle Ergebnisse landen also direkt im
lokalen `data/`-Ordner, und Codeänderungen brauchen keinen neuen Build.

## Ausführen ohne Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main generate
python -m src.main train
python -m src.main predict
python -m src.main bbox
```

Die Befehle immer aus dem Projektordner starten (wegen `-m src.main`).

## Ablauf im Detail

### 1. Datensatz erzeugen – `generate`

Statt hunderte Fotos von Hand zu machen und zu beschriften, werden die Trainingsbilder
synthetisch erzeugt. Dafür wird ein zufälliges Hintergrundbild aus `data/Hintergruende/` genommen
und darauf 10 bis 30 Büroklammern aus `data/Klammern/` verteilt. Jede Klammer wird dabei

- zufällig skaliert (Faktor 0,3 bis 0,55),
- zufällig gedreht (−90° bis 90°) und
- an eine zufällige Position gesetzt (über den Alpha-Kanal mit dem Hintergrund gemischt).

Anschließend wird das Bild auf 420 × 315 Pixel verkleinert und als JPEG gespeichert. Weil beim
Erzeugen bekannt ist, wo welche Klammer liegt, bekommt man die Labels gratis dazu: Für jedes
Bild landen die Anzahl pro Farbe, die Gesamtzahl und die Bounding Box jeder einzelnen Klammer
in `data/raw_dataset/all_data.json`.

Standardmäßig werden 1000 Bilder erzeugt (änderbar mit `-n`).

### 2. Training – `train`

Das Skript macht drei Dinge:

**Augmentierung.** Aus jedem Bild werden 10 Varianten erzeugt (Original, gespiegelt,
um 90°/180°/270° gedreht, mit Rauschen, skaliert, zufällig zugeschnitten, mit veränderter
Helligkeit). Aus 1000 Bildern werden so 10.000. Die Ergebnisse liegen danach in
`data/augmented_dataset/`.

**Vorverarbeitung.** Jedes Bild wird in Graustufen umgewandelt, per Histogrammausgleich im
Kontrast verbessert, leicht weichgezeichnet und mit einer Canny-Kantenerkennung kombiniert.
Das Ergebnis wird auf 128 × 128 Pixel skaliert und so ins Netz gegeben.

**Netz.** Das CNN besteht aus drei Faltungsblöcken (64, 128 und 256 Filter, jeweils mit
Batch-Normalization, Max-Pooling und Dropout), danach eine Dense-Schicht mit 512 Neuronen und
eine Ausgabeschicht mit 7 Werten: die Anzahl pro Farbe plus die Gesamtzahl. Das Zählen wird
also als Regressionsproblem behandelt (Loss: Mean Squared Error, Optimizer: Adam).

Trainiert wird mit einem 80/20-Split, maximal 25 Epochen und Early Stopping. Nach dem
Training wird der Loss-Verlauf als `data/output/loss.png` gespeichert und für jede Klasse MAE, MSE, R² und eine Genauigkeit
mit Toleranz ±1 Klammer ausgegeben.

Gespeichert werden `data/models/best_model.keras` (bestes Modell laut Validierungs-Loss) und
`data/models/model.keras` (Endstand), das von `predict` geladen wird.

### 3. Vorhersage – `predict`

Ein eigenes Bild als `data/bild.jpg` ablegen oder den Pfad direkt angeben. Das Bild muss unter
`data/` liegen, damit der Container es sieht.

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

### 4. Bounding Boxen – `bbox`

Liest die Annotationen aus `data/evaluation/data_bild_1.json` und zeichnet um jede Klammer in
`data/evaluation/bild_1.jpg` ein Rechteck in der Farbe der jeweiligen Klammer. Das Ergebnis wird
als `data/output/bounding_bild_1.jpg` gespeichert.

## Hinweise

- `data/augmented_dataset/` wird beim Training jedes Mal neu geschrieben und ist ca. zehnmal so
  groß wie `data/raw_dataset/`.
- Da das Netz nur auf synthetischen Bildern trainiert wurde, hängt die Qualität bei echten
  Fotos stark davon ab, wie ähnlich Hintergrund, Beleuchtung und Klammern den
  Trainingsdaten sind.
