import os
import urllib.request

# COCO download URLs
FILES = {
    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017.zip":
        "http://images.cocodataset.org/zips/val2017.zip",
}

download_dir = "./data/coco/"

for filename, url in FILES.items():
    output_path = os.path.join(download_dir, filename)

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

print("All downloads completed.")