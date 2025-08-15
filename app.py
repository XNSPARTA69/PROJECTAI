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

# Crear carpeta uploads si no existe
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# --- Ollama helpers ---

OLLAMA_HOST = "http://localhost:11434"

def is_ollama_running():
    """Check if Ollama is running locally."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return response.ok
    except Exception:
        return False

def start_ollama():
    """Try to start Ollama as a background process."""
    try:
        # Windows
        if os.name == "nt":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["ollama", "serve"])
        # Espera un poco a que arranque
        for _ in range(10):
            if is_ollama_running():
                return True
            time.sleep(1)
        return False
    except Exception as e:
        print("Error al iniciar Ollama:", e)
        return False

def is_llama3_available():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if response.ok:
            tags = response.json().get("models", [])
            return any("llama3" in m.get("name", "") for m in tags)
        return False
    except Exception:
        return False

def ensure_ollama_and_llama3():
    """Arranca Ollama y asegura que el modelo llama3 está disponible."""
    if not is_ollama_running():
        print("Ollama no está corriendo, iniciando...")
        if not start_ollama():
            raise RuntimeError("No se pudo iniciar Ollama automáticamente.")
    if not is_llama3_available():
        print("Descargando modelo llama3...")
        # Descarga el modelo llama3 si no está instalado
        subprocess.run(["ollama", "pull", "llama3"], check=True)
        time.sleep(2)

# --- Load models ---

def load_yolo_model():
    return YOLO('yolov8n.pt')

def load_monai_model():
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
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # --- Verifica Ollama y llama3 ---
        ensure_ollama_and_llama3()

        if 'image' not in request.files:
            return jsonify({'error': 'No se subió ninguna imagen'}), 400

        file = request.files['image']
        image_path = f"uploads/{file.filename}"
        file.save(image_path)

        # Validar imagen
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'No se pudo cargar la imagen'}), 400

        # Convertir a escala de grises
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # (H, W)

        # --- MONAI transformaciones y segmentación CORREGIDO ---
        # Si la imagen es 2D, pon canal primero; si ya tiene canal, no lo hagas
        if grayscale_image.ndim == 2:
            monai_image = np.expand_dims(grayscale_image, axis=0)  # (1, H, W)
        else:
            monai_image = grayscale_image  # Ya tiene canal

        # Aplica transformaciones (sin EnsureChannelFirst)
        transforms = Compose([ScaleIntensity(), EnsureType()])
        monai_image = transforms(monai_image)  # (1, H, W)

        # Añade batch dimension
        monai_image = np.expand_dims(monai_image, axis=0)  # (1, 1, H, W)
        if monai_image.ndim != 4:
            return jsonify({'error': f'Las dimensiones de la imagen no son válidas: {monai_image.shape}'}), 400

        segmentation = monai_model(torch.tensor(monai_image).float()).detach().cpu().numpy()

        # YOLO detección
        results = yolo_model(image, imgsz=320)
        detections = results[0].boxes.xyxy.cpu().numpy()

        # Visualización
        output_image = visualize_results(image, detections, segmentation, image_path)

        # Procesar datos para LLaMA 3
        detection_summary = f"{len(detections)} objetos detectados (prótesis, extremidades, u otros)."
        segmentation_summary = f"Área segmentada: {int(np.sum(segmentation > 0.5))} píxeles."

        # Consulta a Ollama/Llama 3
        try:
            description = ollama.chat(
                model="llama3:latest",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Eres un asistente médico especializado en prótesis. Basándote en los datos proporcionados, "
                            "describe cualquier problema potencial con el ajuste de la prótesis o anomalías en la imagen. "
                            "Sé conciso, claro y útil para un usuario no técnico. "
                            "Responde con palabras suaves y apoyo emocional. "
                            f"Detecciones: {detection_summary}\n"
                            f"Segmentación: {segmentation_summary}\n"
                            "Ejemplo: 'La prótesis parece desalineada en la parte superior, lo que podría causar incomodidad.'"
                            "Responde únicamente en español.\n"

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
            return jsonify({'error': f'Error al comunicarse con LLaMA 3: {str(e)}'}), 500

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