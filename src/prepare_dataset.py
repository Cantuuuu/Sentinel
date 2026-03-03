from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extrae, valida y divide un dataset YOLO en train/val/test."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Carpeta ya extraida que contiene images/ y labels/. Si se indica, no usa el zip.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=repo_root / "DeepSpaceYoloDataset.zip",
        help="Ruta al archivo zip del dataset.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directorio donde se extraera el zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "yolo",
        help="Directorio final con images/labels separados en train/val/test.",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=repo_root / "configs" / "deepspace_dataset.yaml",
        help="Ruta del archivo YAML para YOLO.",
    )
    parser.add_argument(
        "--class-name",
        default="space_object",
        help="Nombre de la clase 0 si el dataset tiene una sola clase.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Lista de nombres de clase en orden de ID, por ejemplo: --class-names star galaxy nebula",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe la salida y vuelve a extraer si hace falta.",
    )
    return parser


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Las proporciones deben sumar 1.0 y actualmente suman {total:.4f}."
        )


def extract_dataset(zip_path: Path, extract_dir: Path, force: bool) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"No existe el archivo zip: {zip_path}")

    dataset_root = extract_dir / zip_path.stem

    if dataset_root.exists() and not force:
        return dataset_root

    if dataset_root.exists() and force:
        shutil.rmtree(dataset_root)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Se extrajo el zip pero no se encontro la carpeta esperada: {dataset_root}"
        )

    return dataset_root


def resolve_dataset_root(
    dataset_dir: Path | None,
    zip_path: Path,
    extract_dir: Path,
    force: bool,
) -> tuple[Path, bool]:
    if dataset_dir is not None:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta del dataset: {dataset_dir}")
        if not dataset_dir.is_dir():
            raise NotADirectoryError(f"La ruta del dataset no es una carpeta: {dataset_dir}")
        return dataset_dir, False

    return extract_dataset(zip_path, extract_dir, force), True


def collect_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "El dataset debe contener las carpetas 'images' y 'labels'."
        )

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(label_path)
            continue
        pairs.append((image_path, label_path))

    if missing_labels:
        sample = ", ".join(str(path.name) for path in missing_labels[:5])
        raise FileNotFoundError(
            f"Faltan etiquetas para algunas imagenes. Ejemplos: {sample}"
        )

    if not pairs:
        raise ValueError("No se encontraron pares imagen/etiqueta en el dataset.")

    return pairs


def class_ids_from_labels(pairs: list[tuple[Path, Path]]) -> list[int]:
    ids: set[int] = set()
    for _, label_path in pairs:
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                ids.add(int(line.split()[0]))
    return sorted(ids)


def resolve_class_names(
    class_ids: list[int],
    class_name: str,
    class_names: list[str] | None,
) -> dict[int, str]:
    if not class_ids:
        raise ValueError("No se encontraron IDs de clase en las etiquetas.")

    expected_ids = list(range(max(class_ids) + 1))
    if class_ids != expected_ids:
        raise ValueError(
            "Los IDs de clase deben ser contiguos y empezar en 0. "
            f"Se encontraron {class_ids}, pero se esperaba {expected_ids}."
        )

    if class_names is not None:
        if len(class_names) != len(class_ids):
            raise ValueError(
                "La cantidad de nombres de clase no coincide con los IDs detectados. "
                f"IDs: {class_ids}. Nombres recibidos: {len(class_names)}."
            )
        return {index: name for index, name in enumerate(class_names)}

    if len(class_ids) == 1:
        return {0: class_name}

    return {class_id: f"class_{class_id}" for class_id in class_ids}


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        has_files = any(output_dir.rglob("*"))
        if has_files and not force:
            raise FileExistsError(
                f"La carpeta de salida ya contiene archivos: {output_dir}. "
                "Usa --force para regenerarla."
            )
        if has_files and force:
            shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_split_files(
    splits: dict[str, list[tuple[Path, Path]]],
    output_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name, items in splits.items():
        for image_path, label_path in items:
            shutil.copy2(image_path, output_dir / "images" / split_name / image_path.name)
            shutil.copy2(label_path, output_dir / "labels" / split_name / label_path.name)
        counts[split_name] = len(items)
    return counts


def write_dataset_yaml(
    yaml_path: Path,
    output_dir: Path,
    class_names: dict[int, str],
) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    relative_output = output_dir.relative_to(Path(__file__).resolve().parents[1])
    lines = [
        f"path: {relative_output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id, name in class_names.items():
        lines.append(f"  {class_id}: {name}")
    lines.append("")
    content = "\n".join(lines)
    yaml_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_root, extracted_from_zip = resolve_dataset_root(
        args.dataset_dir,
        args.zip_path,
        args.extract_dir,
        args.force,
    )
    pairs = collect_pairs(dataset_root)
    class_ids = class_ids_from_labels(pairs)
    class_names = resolve_class_names(class_ids, args.class_name, args.class_names)
    splits = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)

    prepare_output_dir(args.output_dir, args.force)
    counts = copy_split_files(splits, args.output_dir)
    write_dataset_yaml(args.yaml_path, args.output_dir, class_names)

    if extracted_from_zip:
        print(f"Dataset extraido en: {dataset_root}")
    else:
        print(f"Dataset leido desde carpeta local: {dataset_root}")
    print(f"Dataset YOLO listo en: {args.output_dir}")
    print(f"Archivo YAML generado en: {args.yaml_path}")
    print(f"Total de pares: {len(pairs)}")
    print(f"Clases encontradas en etiquetas: {class_ids}")
    print(f"Nombres de clase: {class_names}")
    print(
        f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )


if __name__ == "__main__":
    main()
