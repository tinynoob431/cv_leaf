import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare processed_data for CV project.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset download URL.")
    parser.add_argument("--train_per_class", type=int, default=120, help="Train images per class.")
    parser.add_argument("--eval_per_class", type=int, default=30, help="Eval images per class.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed_data directory.")
    return parser.parse_args()


def extract_tgz(tgz_path: Path, dst_dir: Path):
    with tarfile.open(tgz_path, "r:gz") as tf:
        try:
            tf.extractall(path=dst_dir, filter="data")
        except TypeError:
            tf.extractall(path=dst_dir)


def main():
    args = parse_args()
    root = Path.cwd()
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = datasets_dir / "flower_photos.tgz"
    raw_root = datasets_dir / "flower_photos"
    processed_dir = root / "processed_data"
    images_dir = processed_dir / "images"

    if not tgz_path.exists():
        print(f"Downloading: {args.url}")
        urllib.request.urlretrieve(args.url, str(tgz_path))
        print(f"Saved: {tgz_path}")
    else:
        print(f"Using existing archive: {tgz_path}")

    if not raw_root.exists():
        print("Extracting archive...")
        extract_tgz(tgz_path, datasets_dir)
        print(f"Extracted to: {raw_root}")
    else:
        print(f"Using existing extracted folder: {raw_root}")

    classes = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    if len(classes) < 5:
        raise RuntimeError(f"Need >=5 classes, but found {len(classes)} classes: {classes}")

    if processed_dir.exists():
        if args.force:
            shutil.rmtree(processed_dir)
        else:
            raise FileExistsError("processed_data already exists. Use --force to overwrite.")

    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    required = args.train_per_class + args.eval_per_class

    label_lines = []
    train_lines = []
    eval_lines = []

    for label_id, cls in enumerate(classes):
        imgs = sorted((raw_root / cls).glob("*.jpg"))
        rng.shuffle(imgs)
        if len(imgs) < required:
            raise RuntimeError(f"Class {cls} has {len(imgs)} images, need {required}.")

        selected = imgs[:required]
        train_imgs = selected[:args.train_per_class]
        eval_imgs = selected[args.train_per_class:args.train_per_class + args.eval_per_class]

        dst_cls = images_dir / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        for p in train_imgs + eval_imgs:
            shutil.copy2(p, dst_cls / p.name)

        label_lines.append(f"{label_id}\t{cls}")
        for p in train_imgs:
            rel = (Path("images") / cls / p.name).as_posix()
            train_lines.append(f"{rel} {label_id}")
        for p in eval_imgs:
            rel = (Path("images") / cls / p.name).as_posix()
            eval_lines.append(f"{rel} {label_id}")

    (processed_dir / "label_list.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (processed_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (processed_dir / "eval.txt").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    print(f"Done. classes={len(classes)}, train={len(train_lines)}, eval={len(eval_lines)}")
    print(f"Output: {processed_dir}")


if __name__ == "__main__":
    main()
