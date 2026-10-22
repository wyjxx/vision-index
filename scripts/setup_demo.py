"""Install the 500-product release catalog and import it into SQLite."""

import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import requests

from app.config import product_catalog_path
from scripts.import_products_sqlite import main as import_catalog


DEMO_URL = "https://github.com/wyjxx/vision-index/releases/download/demo-data-500/vision-index-demo.zip"
DEMO_SHA256 = "c369d54cfe504b0777564f4e2c5f0eca312aee467e5797b7372a3c6c863b40b5"


def install_demo():
    """Verify and extract the release ZIP without overwriting an existing catalog."""
    root = product_catalog_path.parent
    if product_catalog_path.exists():
        print("Reusing the existing catalog.", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    print("Downloading the 500-product demo (11 MB)...", flush=True)
    response = requests.get(DEMO_URL, timeout=(20, 60))
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != DEMO_SHA256:
        raise ValueError("Demo archive checksum mismatch")
    with TemporaryDirectory(prefix=".demo-", dir=root) as temporary:
        staging = Path(temporary)
        with ZipFile(BytesIO(response.content)) as archive:
            for name in archive.namelist():
                if not (staging / name).resolve().is_relative_to(staging.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(staging)
        images = root / "images"
        if images.is_dir() and not any(images.iterdir()):
            images.rmdir()
        for item in staging.iterdir():
            if (root / item.name).exists():
                raise FileExistsError(f"Demo output already exists: {item.name}")
        for item in staging.iterdir():
            item.rename(root / item.name)


def main():
    """Prepare demo files and register products for the dashboard."""
    try:
        install_demo()
        import_catalog()
    except (OSError, ValueError, requests.RequestException) as error:
        raise SystemExit(f"Demo setup failed: {error}") from error
    print("Open the dashboard and click Index All Unindexed before searching.")


if __name__ == "__main__":
    main()
