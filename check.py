import os


def check_dataset_integrity(image_folder, label_folder, image_ext=".jpg", label_ext=".txt"):
    images = {f.rsplit('.', 1)[0] for f in os.listdir(image_folder) if f.endswith(image_ext)}
    labels = {f.rsplit('.', 1)[0] for f in os.listdir(label_folder) if f.endswith(label_ext)}
    
    missing_labels = images - labels
    missing_images = labels - images

    if missing_labels:
        print(f"⚠️ Missing labels for {len(missing_labels)} images:", missing_labels)
    if missing_images:
        print(f"⚠️ Missing images for {len(missing_images)} labels:", missing_images)
    
    if not missing_labels and not missing_images:
        print("✅ All images have corresponding labels!")

# Run for train, test, and validation sets
check_dataset_integrity("dataset/train/images", "dataset/train/labels")
check_dataset_integrity("dataset/test/images", "dataset/test/labels")
check_dataset_integrity("dataset/valid/images", "dataset/valid/labels")
