from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import torch
import numpy as np
import os
import subprocess
import time
from ultralytics import YOLO
from monai.transforms import Compose, ScaleIntensity, EnsureType
from monai.networks.nets import UNet
import requests
import ollama

app = Flask(__name__, static_folder="static", template_folder="templates")

# I create the 'uploads' folder if it doesn't exist already
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# --- Ollama helpers ---

OLLAMA_HOST = "http://localhost:11434"

def is_ollama_running():
    """I check if Ollama is running locally."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return response.ok
    except Exception:
        return False

def start_ollama():
    """I attempt to start Ollama as a background process."""
    try:
        # If I'm on Windows
        if os.name == "nt":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["ollama", "serve"])
        # I wait a bit for it to start up
        for _ in range(10):
            if is_ollama_running():
                return True
            time.sleep(1)
        return False
    except Exception as e:
        print("Error when starting Ollama:", e)
        return False

def is_llama31_available():
    # I check if llama3.1:latest is installed
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if response.ok:
            tags = response.json().get("models", [])
            return any("llama3.1:latest" in m.get("name", "") for m in tags)
        return False
    except Exception:
        return False

def ensure_ollama_and_llama31():
    """I make sure Ollama is running and the llama3.1:latest model is available."""
    if not is_ollama_running():
        print("Ollama is not running, starting it...")
        if not start_ollama():
            raise RuntimeError("I couldn't start Ollama automatically.")
    if not is_llama31_available():
        print("Downloading llama3.1:latest model...")
        subprocess.run(["ollama", "pull", "llama3.1:latest"], check=True)
        time.sleep(2)

# --- Load models ---

def load_yolo_model():
    # I load YOLOv8n for object detection
    return YOLO('yolov8n.pt')

def load_monai_model():
    # I load my MONAI UNet for segmentation
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    )
    model.load_state_dict(torch.load("monai_model.pth", map_location="cpu"))
    model.eval()
    return model

yolo_model = load_yolo_model()
monai_model = load_monai_model()

def visualize_results(image, detections, segmentation, image_path):
    # I draw the detection boxes and segmentation mask on the image
    image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mask = (segmentation[0, 0] > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    image[mask > 0] = [0, 0, 255]
    output_path = f"uploads/processed_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, image)
    return output_path

@app.route('/')
def home():
    # I render the main page
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    # I serve static files
    return send_from_directory(app.static_folder, filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # I serve files from the uploads folder
    return send_from_directory('uploads', filename)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # I make sure Ollama and llama3.1:latest are running and available
        ensure_ollama_and_llama31()

        if 'image' not in request.files:
            return jsonify({'error': 'No image was uploaded'}), 400

        file = request.files['image']
        image_path = f"uploads/{file.filename}"
        file.save(image_path)

        # I validate the uploaded image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'Could not load the image'}), 400

        # I convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # (H, W)

        # --- MONAI transforms and segmentation FIXED ---
        # If the image is 2D, I add the channel first; if it already has a channel, I leave it
        if grayscale_image.ndim == 2:
            monai_image = np.expand_dims(grayscale_image, axis=0)  # (1, H, W)
        else:
            monai_image = grayscale_image  # Already has channel

        # I apply transforms (without EnsureChannelFirst)
        transforms = Compose([ScaleIntensity(), EnsureType()])
        monai_image = transforms(monai_image)  # (1, H, W)

        # I add the batch dimension
        monai_image = np.expand_dims(monai_image, axis=0)  # (1, 1, H, W)
        if monai_image.ndim != 4:
            return jsonify({'error': f'Image dimensions are not valid: {monai_image.shape}'}), 400

        segmentation = monai_model(torch.tensor(monai_image).float()).detach().cpu().numpy()

        # I run YOLO detection
        results = yolo_model(image, imgsz=320)
        detections = results[0].boxes.xyxy.cpu().numpy()

        # I visualize results with both models
        output_image = visualize_results(image, detections, segmentation, image_path)

        # I prepare the data for LLaMA 3.1:latest
        detection_summary = f"{len(detections)} objects detected (prosthesis, limbs, or others)."
        segmentation_summary = f"Segmented area: {int(np.sum(segmentation > 0.5))} pixels."

        # I ask Ollama/Llama 3.1:latest for a description
        try:
            description = ollama.chat(
                model="llama3.1:latest",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are a medical assistant specialized in prosthetics. Based on the provided data, "
                            "describe any potential problems with the fit of the prosthesis or anomalies in the image. "
                            "Be concise, clear, and helpful to a non-technical user. "
                            "Respond with supportive and empathetic words. "
                            f"Detections: {detection_summary}\n"
                            f"Segmentation: {segmentation_summary}\n"
                            "Example: 'The prosthesis appears misaligned at the top, which might cause discomfort.'"
                            "Respond only in Spanish.\n"
                        )
                    }
                ],
                options={
                    "num_ctx": 2048,
                    "num_predict": 1000,
                    "temperature": 0.9,
                    "num_gpu": 1,
                    "num_threads": 8
                }
            )['message']['content']
        except Exception as e:
            return jsonify({'error': f'Error communicating with LLaMA 3.1:latest: {str(e)}'}), 500

        return jsonify({
            'detections': detections.tolist(),
            'segmentation': segmentation.tolist(),
            'description': description,
            'output_image': f"uploads/processed_{os.path.basename(image_path)}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    ###FALTA ENTRENAR MONAI PARA IMAGENES