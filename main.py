import torch
import numpy as np
from fastapi import FastAPI
from NCA_model import NCAGenerator ,get_config # Direct import

app = FastAPI()
generator_model = None  # Global model instance




@app.on_event("startup")
def load_model_on_startup():
    global generator_model

    cfg = get_config()
    generator_model = NCAGenerator(
        channels=cfg["channels"],
        hidden=cfg["hidden"],
        steps=cfg["steps"],
        dropout=cfg["dropout"],
        length=cfg["length"]
    )
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
