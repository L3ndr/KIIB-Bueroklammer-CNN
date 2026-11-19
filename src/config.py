"""
Zentrale Pfade für alle Skripte.

Alle Pfade hängen am Projektordner (lokal: Repo-Root, im Container: /app),
dadurch ist es egal, aus welchem Verzeichnis das Programm gestartet wird.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Eingabedaten (eingecheckt)
BACKGROUNDS_DIR = DATA_DIR / "Hintergruende"
CLIPS_DIR = DATA_DIR / "Klammern"
EVALUATION_DIR = DATA_DIR / "evaluation"

# Generierte Daten (gitignored)
RAW_DATASET_DIR = DATA_DIR / "raw_dataset"
AUGMENTED_DATASET_DIR = DATA_DIR / "augmented_dataset"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"

MODEL_PATH = MODELS_DIR / "model.keras"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
DEFAULT_PREDICT_IMAGE = DATA_DIR / "bild.jpg"
