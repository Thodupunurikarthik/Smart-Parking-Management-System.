import os
import shutil

def move_images_to_folder(base_path):
    for split in ["train", "test", "valid"]:
        split_path = os.path.join(base_path, split)
        images_path = os.path.join(split_path, "images")
        
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        
        for file in os.listdir(split_path):
            if file.endswith(".jpg") or file.endswith(".png"):  # Move only images
                src = os.path.join(split_path, file)
                dst = os.path.join(images_path, file)

                try:
                    if not os.path.exists(dst):  # Skip if already moved
                        shutil.move(src, dst)
                        print(f"✅ Moved: {file}")
                    else:
                        print(f"⚠️ Skipped (already exists): {file}")
                except Exception as e:
                    print(f"❌ Error moving {file}: {e}")

dataset_path = "E:/Smart/dataset"
move_images_to_folder(dataset_path)
print("🎉 Done! All images should be in the correct folders.")
