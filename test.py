import os

dataset_path = "E:/Smart/dataset"  # Update this path if needed

for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}📁 {os.path.basename(root)}/")
    sub_indent = " " * 4 * (level + 1)
    for f in files[:5]:  # Print only first 5 files per folder
        print(f"{sub_indent}📄 {f}")
import os

if not os.path.exists("dataset/valid/labels"):
    print("❌ The folder 'dataset/valid/labels' is missing!")
else:
    print("✅ The folder 'dataset/valid/labels' exists.")
