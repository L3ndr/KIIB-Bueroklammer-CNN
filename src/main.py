"""
Einstiegspunkt für alle Schritte der Pipeline.

Beispiele:
    python -m src.main generate
    python -m src.main train
    python -m src.main predict data/bild.jpg
    python -m src.main bbox
"""

import argparse

from src import config


def cli():
    parser = argparse.ArgumentParser(prog="bueroklammer-cnn", description="Büroklammern zählen mit einem CNN")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_generate = sub.add_parser("generate", help="Synthetischen Datensatz erzeugen")
    p_generate.add_argument("-n", type=int, default=1000, help="Anzahl der Bilder (Standard: 1000)")

    sub.add_parser("train", help="Datensatz augmentieren und Modell trainieren")

    p_predict = sub.add_parser("predict", help="Büroklammern auf einem Bild zählen")
    p_predict.add_argument("image", nargs="?", default=str(config.DEFAULT_PREDICT_IMAGE),
                           help="Pfad zum Bild (Standard: data/bild.jpg)")

    sub.add_parser("bbox", help="Bounding Boxen für das Beispielbild einzeichnen")

    args = parser.parse_args()

    # Imports erst hier, damit TensorFlow nur bei train/predict geladen wird
    if args.cmd == "generate":
        from src.create_dataset import main
        main(n=args.n)
    elif args.cmd == "train":
        from src.train import main
        main()
    elif args.cmd == "predict":
        from src.predict import main
        main(args.image)
    elif args.cmd == "bbox":
        from src.bounding_boxes import main
        main()


if __name__ == "__main__":
    cli()
