import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path


PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare processed_data for CV project.")
    parser.add_argument("--url", default=PLANTVILLAGE_URL, help="PlantVillage download URL.")
    parser.add_argument("--train_per_class", type=int, default=120, help="Train images per class.")
    parser.add_argument("--eval_per_class", type=int, default=30, help="Eval images per class.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes to keep.")
    parser.add_argument(
        "--class_names",
        default="",
        help="Optional comma-separated class names. If set, --num_classes is ignored.",
    )
    parser.add_argument(
        "--plantvillage_variant",
        choices=["color", "grayscale", "segmented"],
        default="color",
        help="Image variant for PlantVillage.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed_data directory.")
    return parser.parse_args()


def extract_archive(archive_path: Path, dst_dir: Path):
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=dst_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def ensure_archive(datasets_dir: Path, url: str) -> Path:
    archive_name = "plantvillage_master.zip"
    archive_path = datasets_dir / archive_name
    if archive_path.exists():
        print(f"Using existing archive: {archive_path}")
        return archive_path

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, str(archive_path))
    print(f"Saved: {archive_path}")
    return archive_path


def find_plantvillage_root(datasets_dir: Path, variant: str) -> Path:
    direct_candidates = [
        datasets_dir / "PlantVillage-Dataset-master" / "raw" / variant,
        datasets_dir / "PlantVillage-Dataset" / "raw" / variant,
        datasets_dir / "raw" / variant,
    ]
    for p in direct_candidates:
        if p.exists() and p.is_dir():
            return p

    for p in datasets_dir.rglob(variant):
        if p.is_dir() and p.parent.name == "raw":
            return p

    raise FileNotFoundError(
        f"Cannot locate PlantVillage raw/{variant} directory under {datasets_dir}."
    )


def collect_class_images(raw_root: Path):
    class_to_images = {}
    for cls_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        images = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        images = sorted(images)
        if images:
            class_to_images[cls_dir.name] = images
    return class_to_images


def parse_class_names(raw: str):
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def choose_classes(class_to_images, args):
    wanted = parse_class_names(args.class_names)
    if wanted:
        missing = [name for name in wanted if name not in class_to_images]
        if missing:
            available_preview = ", ".join(sorted(class_to_images.keys())[:20])
            raise ValueError(
                f"Classes not found: {missing}. Available (first 20): {available_preview}"
            )
        return wanted

    ranked = sorted(
        class_to_images.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    num_classes = max(1, int(args.num_classes))
    selected = [name for name, _ in ranked[:num_classes]]
    if len(selected) < num_classes:
        raise RuntimeError(
            f"Requested {num_classes} classes, but only found {len(selected)} classes."
        )
    return selected


def write_processed_data(processed_dir: Path, selected_classes, class_to_images, args):
    images_dir = processed_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    required = int(args.train_per_class) + int(args.eval_per_class)

    label_lines = []
    train_lines = []
    eval_lines = []

    for label_id, cls_name in enumerate(selected_classes):
        imgs = class_to_images[cls_name][:]
        rng.shuffle(imgs)
        if len(imgs) < required:
            raise RuntimeError(
                f"Class '{cls_name}' has {len(imgs)} images, need {required} "
                f"(train={args.train_per_class}, eval={args.eval_per_class})."
            )

        selected = imgs[:required]
        train_imgs = selected[: int(args.train_per_class)]
        eval_imgs = selected[int(args.train_per_class):required]

        dst_cls = images_dir / cls_name
        dst_cls.mkdir(parents=True, exist_ok=True)
        for p in train_imgs + eval_imgs:
            shutil.copy2(p, dst_cls / p.name)

        label_lines.append(f"{label_id}\t{cls_name}")
        for p in train_imgs:
            rel = (Path("images") / cls_name / p.name).as_posix()
            train_lines.append(f"{rel} {label_id}")
        for p in eval_imgs:
            rel = (Path("images") / cls_name / p.name).as_posix()
            eval_lines.append(f"{rel} {label_id}")

    (processed_dir / "label_list.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (processed_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (processed_dir / "eval.txt").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    (processed_dir / "dataset_meta.txt").write_text(
        "\n".join(
            [
                "source=plantvillage",
                f"train_per_class={args.train_per_class}",
                f"eval_per_class={args.eval_per_class}",
                f"seed={args.seed}",
                f"classes={','.join(selected_classes)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Done. classes={len(selected_classes)}, train={len(train_lines)}, eval={len(eval_lines)}")
    print(f"Output: {processed_dir}")


def main():
    args = parse_args()
    root = Path.cwd()
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = root / "processed_data"

    if processed_dir.exists():
        if args.force:
            shutil.rmtree(processed_dir)
        else:
            raise FileExistsError("processed_data already exists. Use --force to overwrite.")

    archive_path = ensure_archive(datasets_dir, args.url)

    raw_root = None
    try:
        raw_root = find_plantvillage_root(datasets_dir, args.plantvillage_variant)
        print(f"Using existing extracted folder: {raw_root}")
    except FileNotFoundError:
        pass

    if raw_root is None:
        print("Extracting archive...")
        extract_archive(archive_path, datasets_dir)
        raw_root = find_plantvillage_root(datasets_dir, args.plantvillage_variant)
        print(f"Extracted to: {raw_root}")

    class_to_images = collect_class_images(raw_root)
    if len(class_to_images) < 5:
        raise RuntimeError(
            f"Need at least 5 classes, found {len(class_to_images)} under {raw_root}."
        )

    selected_classes = choose_classes(class_to_images, args)
    print("Selected classes:")
    for cls_name in selected_classes:
        print(f"  - {cls_name}: {len(class_to_images[cls_name])} images")

    write_processed_data(processed_dir, selected_classes, class_to_images, args)


if __name__ == "__main__":
    main()
