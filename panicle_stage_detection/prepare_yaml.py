import json
import os
import cv2
import random
import shutil
import yaml
from tqdm import tqdm

# === INPUTS ===
json_file = "/content/drive/MyDrive/mtech_3rd_sem/projectA/in_field_panicle_detection/project_pretty_sep16_2.json"   # Label Studio export
image_root = "/content/drive/MyDrive/mtech_3rd_sem/projectA/in_field_panicle_detection/\
In-FieldRicePaniclesDetectionofDifferentGrowthStages/Datasets/TrainingDatasets/images/"                   # folder with original images
output_root = "/content/drive/MyDrive/mtech_3rd_sem/projectA/in_field_panicle_detection/dataset_yolo"           # final dataset folder
resize_to = (640, 640)                   # set to None to keep original size
split_ratio = (0.7, 0.2, 0.1)            # train/val/test split
class_map = {"Panicle": 0}               # detection-only, single class

# === TEMP LABELS FOLDER ===
labels_dir = os.path.join(output_root, "labels_all")
os.makedirs(labels_dir, exist_ok=True)

# === STEP 1: Convert JSON to YOLO TXT ===
with open(json_file, "r") as f:
    data = json.load(f)

for task in tqdm(data, desc="Converting JSON to YOLO labels", unit="img"):
    img_name = task.get("data", {}).get("image", "")
    if not img_name:
        continue

    img_name = os.path.basename(img_name)
    img_path = os.path.join(image_root, img_name)

    if not os.path.exists(img_path):
        continue

    # Load image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Optional resize
    if resize_to is not None:
        img = cv2.resize(img, resize_to)
        h, w = resize_to
        cv2.imwrite(img_path, img)  # overwrite with resized version

    label_lines = []

    # Process annotations
    for ann in task.get("annotations", []):
        for result in ann.get("result", []):
            if result.get("type") == "polygonlabels":
                points = result["value"].get("points", [])
                if not points:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

                # Convert % → pixels
                x1, y1 = int(xmin * w / 100), int(ymin * h / 100)
                x2, y2 = int(xmax * w / 100), int(ymax * h / 100)

                # YOLO format (normalized 0–1)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bbox_width = (x2 - x1) / w
                bbox_height = (y2 - y1) / h

                label_lines.append(
                    f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                )

    # Save YOLO label
    if label_lines:
        label_file = os.path.join(labels_dir, img_name.rsplit(".", 1)[0] + ".txt")
        with open(label_file, "w") as lf:
            lf.write("\n".join(label_lines))

# === STEP 2: Train/Val/Test Split ===
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

all_images = [f for f in os.listdir(image_root) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(split_ratio[0] * n_total)
n_val = int(split_ratio[1] * n_total)
n_test = n_total - n_train - n_val

train_files = all_images[:n_train]
val_files = all_images[n_train:n_train+n_val]
test_files = all_images[n_train+n_val:]

splits = {"train": train_files, "val": val_files, "test": test_files}

print(f"\nTotal images: {n_total}")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

for split, files in splits.items():
    for img_name in tqdm(files, desc=f"Copying {split}", unit="img"):
        # Copy image
        src_img = os.path.join(image_root, img_name)
        dst_img = os.path.join(output_root, "images", split, img_name)
        shutil.copy(src_img, dst_img)

        # Copy label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(output_root, "labels", split, label_name)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

# === STEP 3: Generate data.yaml ===
data_yaml = {
    "train": os.path.join(output_root, "images", "train"),
    "val": os.path.join(output_root, "images", "val"),
    "test": os.path.join(output_root, "images", "test"),
    "nc": len(class_map),
    "names": list(class_map.keys())
}

yaml_path = os.path.join(output_root, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\nDataset ready at {output_root}")
print(f"data.yaml created at {yaml_path}")
