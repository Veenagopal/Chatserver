import os
import gdown
import torch
import numpy as np
from fastapi import FastAPI

app = FastAPI()
generator_model = None  # Global variable for the model

# Function to download from Google Drive using file ID
def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

@app.on_event("startup")
def load_model_on_startup():
    global generator_model

    # Replace with your actual file IDs
    model_file_id = "YOUR_MODEL_FILE_ID"
    code_file_id = "YOUR_NCA_MODEL_PY_FILE_ID"

    # Download the model and code file
    download_from_drive(model_file_id, "best_generator_g2.pt")
    download_from_drive(code_file_id, "NCA_model.py")

    # Dynamically import NCA_model.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("NCA_model", "NCA_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Load model class
    GeneratorClass = module.NCAGenerator

    # Instantiate and load model
    generator_model = GeneratorClass()
    generator_model.load_state_dict(torch.load("best_generator_g2.pt", map_location="cpu"))
    generator_model.eval()

@app.get("/random-256")
def generate_random():
    global generator_model
    if generator_model is None:
        return {"error": "Model not loaded"}

    with torch.no_grad():
        z = torch.randn(1, 128)
        output = generator_model(z)
        probs = torch.sigmoid(output)
        bits = (probs > 0.5).int().squeeze().cpu().numpy()

        # Ensure exactly 256 bits
        bits = bits[:256] if len(bits) > 256 else np.pad(bits, (0, 256 - len(bits)), mode='constant')
        byte_array = np.packbits(bits)

        return {
            "random_bits": bits.tolist(),
            "random_hex": byte_array.tobytes().hex()
        }
