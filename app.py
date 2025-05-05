import streamlit as st
import subprocess
import os

def run_image_inference(image_path):
    try:
        result = subprocess.run(["python", "yolo_image_inference.py", image_path], capture_output=True, text=True)
        if result.stdout:
            st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)
        
        output_image_path = os.path.splitext(image_path)[0] + "_output.jpg"
        if os.path.exists(output_image_path):
            st.image(output_image_path, caption="Inference Output", use_column_width=True)
        else:
            st.error("Output image not found.")
    except Exception as e:
        st.error(f"Error running image inference: {e}")

def run_video_inference(video_path):
    try:
        result = subprocess.run(["python", "yolo_video_inf.py", video_path], capture_output=True, text=True)
        if result.stdout:
            st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)
        
        output_video_path = os.path.splitext(video_path)[0] + "_output.mp4"
        if os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.error("Output video not found.")
    except Exception as e:
        st.error(f"Error running video inference: {e}")

def main():
    st.title("Parking Lot Detection - YOLOv8")
    
    option = st.radio("Choose an option:", ("Image Inference", "Video Inference"))

    if option == "Image Inference":
        image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            image_path = os.path.join("uploaded_files", image_file.name)
            os.makedirs("uploaded_files", exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
            st.success("✅ Image uploaded successfully! Running inference...")
            run_image_inference(image_path)

    elif option == "Video Inference":
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if video_file is not None:
            video_path = os.path.join("uploaded_files", video_file.name)
            os.makedirs("uploaded_files", exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.success("✅ Video uploaded successfully! Running inference...")
            run_video_inference(video_path)

if __name__ == "__main__":
    main()
