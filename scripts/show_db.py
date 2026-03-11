from app.storage.db import get_all_images

rows = get_all_images()

for row in rows:
    print(dict(row))