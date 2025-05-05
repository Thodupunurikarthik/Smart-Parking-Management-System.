import os
import xml.etree.ElementTree as ET
from glob import glob

# Define paths
dataset_path = "E:\\Smart\\dataset"  # Update this with your PKLot dataset path
yolo_labels_path = "yolo_labels"  # Folder to store converted labels
os.makedirs(yolo_labels_path, exist_ok=True)

# Define class name
class_name = "parking_slot"  # YOLO expects class IDs; you can modify this if needed

# Function to convert bounding box format
def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 * dw
    y = (box[2] + box[3]) / 2.0 * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return x, y, w, h

# Process all XML files
xml_files = glob(os.path.join(dataset_path, "**", "*.xml"), recursive=True)

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_size = (int(root.find("size/width").text), int(root.find("size/height").text))
    
    yolo_file = os.path.join(yolo_labels_path, os.path.basename(xml_file).replace(".xml", ".txt"))
    with open(yolo_file, "w") as f:
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            box = [int(bbox.find("xmin").text), int(bbox.find("xmax").text),
                   int(bbox.find("ymin").text), int(bbox.find("ymax").text)]
            yolo_box = convert_bbox(image_size, box)
            f.write(f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")  # '0' is the class ID

print("Conversion completed! YOLO labels saved in:", yolo_labels_path)
