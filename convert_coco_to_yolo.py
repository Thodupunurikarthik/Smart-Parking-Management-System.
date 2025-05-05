import os
import json

def convert_coco_to_yolo(coco_json_path, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    category_mapping = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    print(f"📂 Processing {len(coco_data['images'])} images...")

    for idx, image in enumerate(coco_data["images"]):
        img_id = image["id"]
        img_file = image["file_name"]

        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == img_id]

        yolo_label_path = os.path.join(output_labels_dir, img_file.replace(".jpg", ".txt"))

        with open(yolo_label_path, "w") as label_file:
            for ann in annotations:
                category_id = ann["category_id"]
                yolo_class_id = category_mapping[category_id]
                x, y, width, height = ann["bbox"]

                x_center = (x + width / 2) / image["width"]
                y_center = (y + height / 2) / image["height"]
                width /= image["width"]
                height /= image["height"]

                label_file.write(f"{yolo_class_id} {x_center} {y_center} {width} {height}\n")

        if idx % 100 == 0:  # Print progress every 100 images
            print(f"✅ Processed {idx} images...")

    print(f"✅ COCO to YOLO conversion completed! Labels saved in '{output_labels_dir}/'")

dataset_dir = "E:/Smart/dataset"

convert_coco_to_yolo(os.path.join(dataset_dir, "train/_annotations.coco.json"), os.path.join(dataset_dir, "train/labels"))
convert_coco_to_yolo(os.path.join(dataset_dir, "test/_annotations.coco.json"), os.path.join(dataset_dir, "test/labels"))
