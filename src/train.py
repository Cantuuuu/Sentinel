from __future__ import annotations

import argparse
import os
from pathlib import Path


def find_latest_checkpoint(runs_dir: Path) -> Path | None:
    candidates = sorted(
        runs_dir.glob("*/weights/last.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_workers = 0 if os.name == "nt" else max(2, min(8, (os.cpu_count() or 4) - 1))
    parser = argparse.ArgumentParser(description="Entrena un modelo YOLO localmente.")
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "configs" / "deepspace_dataset.yaml",
        help="Ruta al archivo dataset YAML.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Checkpoint base de YOLO para fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=608)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda desde el ultimo checkpoint last.pt encontrado en runs/.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Usa 'auto' para seleccionar GPU si CUDA esta disponible.",
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument(
        "--cache",
        default="ram",
        choices=("false", "ram", "disk"),
        help="Cache del dataset para acelerar entrenamiento.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximo de detecciones por imagen durante validacion e inferencia interna.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraccion del dataset a usar durante el entrenamiento. Util para pruebas rapidas.",
    )
    parser.add_argument("--run-name", default="deepspace_yolov8n")
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

    if not args.data.exists():
        raise FileNotFoundError(
            f"No existe el archivo de dataset: {args.data}. "
            "Primero ejecuta src/prepare_dataset.py."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "No se encontro ultralytics. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    repo_root = Path(__file__).resolve().parents[1]
    device = resolve_device(args.device)
    cache = False if args.cache == "false" else args.cache

    print(f"Dispositivo seleccionado: {device}")
    print(f"Workers: {args.workers}")
    print(f"Cache: {cache}")
    print(f"Max detections: {args.max_det}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Fraction: {args.fraction}")
    if os.name == "nt" and args.workers == 0:
        print("Modo Windows estable: dataloader sin workers secundarios.")

    if args.resume:
        checkpoint = find_latest_checkpoint(repo_root / "runs")
        if checkpoint is None:
            raise FileNotFoundError(
                "No se encontro ningun checkpoint last.pt en runs/. "
                "Debes iniciar al menos un entrenamiento que guarde pesos."
            )
        print(f"Reanudando desde: {checkpoint}")
        model = YOLO(str(checkpoint))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            cache=cache,
            max_det=args.max_det,
            fraction=args.fraction,
            project=str(repo_root / "runs"),
            name=args.run_name,
            exist_ok=True,
        )

    print(f"Entrenamiento finalizado. Resultados en: {results.save_dir}")
    print(f"Pesos esperados en: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
