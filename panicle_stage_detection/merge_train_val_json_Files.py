import json

# === Set file paths ===
train_file = "/content/drive/MyDrive/mtech_3rd_sem/projectA/\
in_field_panicle_detection/In-FieldRicePaniclesDetectionofDifferentGrowthStages/Datasets/TrainingDatasets/annotations/train.json"
val_file = "/content/drive/MyDrive/mtech_3rd_sem/projectA/\
in_field_panicle_detection/In-FieldRicePaniclesDetectionofDifferentGrowthStages/Datasets/TrainingDatasets/annotations/val.json"
output_file = "/content/drive/MyDrive/mtech_3rd_sem/projectA/\
in_field_panicle_detection/In-FieldRicePaniclesDetectionofDifferentGrowthStages/Datasets/TrainingDatasets/annotations/combined.json"

# Load both files
with open(train_file, "r") as f:
    train = json.load(f)
with open(val_file, "r") as f:
    val = json.load(f)

# Copy categories (assumed same in both)
categories = train["categories"]

# Shift IDs in val to avoid collision
max_image_id = max(img["id"] for img in train["images"])
max_ann_id = max(ann["id"] for ann in train["annotations"])

# Update val image ids
val_image_id_map = {}
for img in val["images"]:
    old_id = img["id"]
    new_id = old_id + max_image_id + 1
    val_image_id_map[old_id] = new_id
    img["id"] = new_id

# Update val annotation ids and image references
for ann in val["annotations"]:
    ann["id"] += max_ann_id + 1
    ann["image_id"] = val_image_id_map[ann["image_id"]]

# Merge
combined = {
    "images": train["images"] + val["images"],
    "annotations": train["annotations"] + val["annotations"],
    "categories": categories
}

# Save
with open(output_file, "w") as f:
    json.dump(combined, f, indent=2)

print(f"Combined COCO file saved at: {output_file}")
