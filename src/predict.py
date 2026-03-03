from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


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


def resolve_weights(weights: Path | None, repo_root: Path) -> Path:
    resolved = weights or find_latest_weights(repo_root / "runs")
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            "No se encontro un archivo de pesos utilizable. "
            "Primero entrena el modelo con src/train.py."
        )
    return resolved


def load_model(weights: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "No se encontro ultralytics. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    return YOLO(str(weights))


def run_prediction(
    source: Path,
    weights: Path | None = None,
    device_arg: str = "auto",
    conf: float = 0.25,
    save: bool = False,
    run_name: str = "predict",
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_weights = resolve_weights(weights, repo_root)

    if not source.exists():
        raise FileNotFoundError(f"No existe la ruta de entrada: {source}")

    device = resolve_device(device_arg)
    model = load_model(resolved_weights)
    results = model.predict(
        source=str(source),
        conf=conf,
        device=device,
        save=save,
        project=str(repo_root / "outputs"),
        name=run_name,
        exist_ok=True,
    )

    total_boxes = 0
    for result in results:
        if result.boxes is not None:
            total_boxes += len(result.boxes)

    output_dir = repo_root / "outputs" / run_name
    saved_files: list[Path] = []
    if save and output_dir.exists():
        for result in results:
            path = getattr(result, "path", None)
            if path:
                saved_path = output_dir / Path(path).name
                if saved_path.exists():
                    saved_files.append(saved_path)

    return {
        "device": device,
        "weights": resolved_weights,
        "results": results,
        "total_boxes": total_boxes,
        "output_dir": output_dir,
        "saved_files": saved_files,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prediction = run_prediction(
        source=args.source,
        weights=args.weights,
        device_arg=args.device,
        conf=args.conf,
        save=args.save,
        run_name=args.run_name,
    )
    print(f"Dispositivo seleccionado: {prediction['device']}")
    print(f"Pesos cargados: {prediction['weights']}")
    print(f"Predicciones realizadas. Objetos detectados: {prediction['total_boxes']}")
    if args.save:
        print(f"Salidas guardadas en: {prediction['output_dir']}")


if __name__ == "__main__":
    main()
