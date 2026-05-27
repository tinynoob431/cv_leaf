import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create submission zip using course naming format.")
    parser.add_argument("--group_no", required=True, help="Two-digit group number, e.g. 03")
    parser.add_argument("--leader", required=True, help="Leader name")
    parser.add_argument("--member1", required=True, help="Member1 name")
    parser.add_argument("--member2", required=True, help="Member2 name")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    bundle = root / "submission_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)

    # Required materials
    required_paths = [
        root / "deliverables" / "report" / "实验报告_叶片分类_CBAM.docx",
        root / "deliverables" / "presentation" / "课程大作业汇报_叶片分类_CBAM.pptx",
        root / "leaf_classification_cbam.py",
        root / "leaf_classification_cbam.ipynb",
        root / "prepare_dataset.py",
        root / "requirements.txt",
        root / "README.md",
        root / "processed_data",
    ]

    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")
        dst = bundle / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)

    zip_name = f"{args.group_no}_{args.leader}_{args.member1}_{args.member2}"
    zip_path = shutil.make_archive(str(root / zip_name), "zip", root_dir=bundle)
    print("Created:", zip_path)


if __name__ == "__main__":
    main()
