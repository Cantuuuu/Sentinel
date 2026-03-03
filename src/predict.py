from __future__ import annotations

import argparse
from pathlib import Path


def find_latest_weights(runs_dir: Path) -> Path | None:
    preferred = sorted(
        runs_dir.glob("*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]

    fallback = sorted(
        runs_dir.glob("*/weights/last.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return fallback[0] if fallback else None


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Ejecuta inferencia con un modelo YOLO ya entrenado."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Ruta al archivo de pesos entrenados. Si se omite, busca el ultimo best.pt en runs/.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Imagen o carpeta de imagenes para inferencia.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Usa 'auto' para seleccionar GPU si CUDA esta disponible.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", help="Guarda la imagen anotada.")
    parser.add_argument("--run-name", default="predict")
    return parser


def resolve_device(device_arg: str) -> str | int:
    if device_arg != "auto":
        return device_arg

    try:
        import torch
    except ImportError:
        return "cpu"

    return 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    weights = args.weights or find_latest_weights(repo_root / "runs")

    if weights is None or not weights.exists():
        raise FileNotFoundError(
            "No se encontro un archivo de pesos utilizable. "
            "Primero entrena el modelo con src/train.py."
        )

    if not args.source.exists():
        raise FileNotFoundError(f"No existe la ruta de entrada: {args.source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "No se encontro ultralytics. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    device = resolve_device(args.device)
    print(f"Dispositivo seleccionado: {device}")
    print(f"Pesos cargados: {weights}")
    model = YOLO(str(weights))
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        device=device,
        save=args.save,
        project=str(repo_root / "outputs"),
        name=args.run_name,
        exist_ok=True,
    )

    total_boxes = 0
    for result in results:
        if result.boxes is not None:
            total_boxes += len(result.boxes)

    print(f"Predicciones realizadas. Objetos detectados: {total_boxes}")
    if args.save:
        print(f"Salidas guardadas en: {repo_root / 'outputs' / args.run_name}")


if __name__ == "__main__":
    main()
