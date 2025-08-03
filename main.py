import os
import gdown
import torch
import numpy as np
from fastapi import FastAPI

app = FastAPI()
generator_model = None  # Global placeholder

# Function to download from Google Drive
def download_from_drive(file_id, output):
    if not os.path.exists(output):
        gdown.download(id=file_id, output=output, quiet=False)

@app.on_event("startup")
def load_model_on_startup():
    global generator_model

    # Download model code and weights
    download_from_drive("PUT_MODEL_FILE_ID_HERE", "best_generator_g2.pt")
    download_from_drive("PUT_PY_FILE_ID_HERE", "NCA_model.py")

    # ✅ Import AFTER downloading
    import importlib.util

    spec = importlib.util.spec_from_file_location("NCA_model", "NCA_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    GeneratorClass = module.NCAGenerator
    get_config = module.get_config

    # Load model
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

        if len(bits) < 256:
            bits = np.pad(bits, (0, 256 - len(bits)), mode='constant')
        elif len(bits) > 256:
            bits = bits[:256]

        byte_array = np.packbits(bits)
        
        return {"random_hex": byte_array.tobytes().hex()}
