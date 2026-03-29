# scripts/rename_images.py

from pathlib import Path

# Rename: xxx.ipg -> 001.jpg
# Gallery path
IMAGE_DIR = Path("gallery/vi-dataset")

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_images(directory: Path):
    return sorted([
        p for p in directory.iterdir()
        if p.suffix.lower() in EXTS and p.is_file()
    ])


def rename_images():
    images = get_images(IMAGE_DIR)

    print(f"Found {len(images)} images")

    # ----------Step 1: 先重命名成临时名（避免冲突） ----------
    temp_paths = []
    for i, path in enumerate(images):
        temp_path = path.with_name(f"__tmp_{i:03d}{path.suffix.lower()}")
        path.rename(temp_path)
        temp_paths.append(temp_path)

    # ---------- Step 2: 再改成最终名 ----------
    for i, path in enumerate(temp_paths, start=1):
        new_name = f"{i:03d}.jpg"
        new_path = path.with_name(new_name)

        path.rename(new_path)
        print(f"{path.name} -> {new_name}")

    print("Done.")


if __name__ == "__main__":
    rename_images()