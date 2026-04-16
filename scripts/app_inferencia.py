"""
App de inferencia local — Clasificador CNN de enfermedades respiratorias
Clases: COVID | NEUMONIA | NORMALL
"""
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/outputs/best_model.pth")
CLASSES    = ["COVID", "NEUMONIA", "NORMALL"]
IMG_SIZE   = 224
DEVICE     = torch.device("cpu")

# ── Modelo (misma arquitectura que en entrenamiento) ───────────────────────────
class RespiratoryCNN(nn.Module):
    """CNN de 3 bloques: Conv→BN→ReLU→Pool (x3) → FC(256) → Dropout → FC(3)"""
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_model(num_classes=3, dropout=0.3):
    return RespiratoryCNN(num_classes=num_classes, dropout=dropout)

# ── Carga del modelo ───────────────────────────────────────────────────────────
print(f"Cargando modelo desde {MODEL_PATH}...")
model = build_model()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
print("Modelo listo.")

# ── Preprocesamiento ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Función de inferencia ──────────────────────────────────────────────────────
def predecir(imagen: Image.Image):
    if imagen is None:
        return {c: 0.0 for c in CLASSES}

    # Convertir a RGB (por si viene en escala de grises o RGBA)
    imagen = imagen.convert("RGB")

    tensor = transform(imagen).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    return {cls: float(prob) for cls, prob in zip(CLASSES, probs)}

# ── Interfaz Gradio ────────────────────────────────────────────────────────────
with gr.Blocks(title="CNN Respiratorio") as demo:
    gr.Markdown("""
    # Clasificador CNN — Enfermedades Respiratorias
    Sube una radiografía de tórax y el modelo indicará la probabilidad de cada diagnóstico.

    **Clases:** COVID-19 | Neumonía | Normal
    """)

    with gr.Row():
        imagen_in = gr.Image(type="pil", label="Radiografía de tórax")
        resultado = gr.Label(num_top_classes=3, label="Diagnóstico")

    btn = gr.Button("Clasificar", variant="primary")
    btn.click(fn=predecir, inputs=imagen_in, outputs=resultado)

    gr.Examples(
        examples=[],  # puedes agregar rutas a imágenes de ejemplo aquí
        inputs=imagen_in,
    )

    gr.Markdown("""
    ---
    *Modelo: ResNet-50 fine-tuned | Test accuracy: 96.12% | Macro F1: 90.39%*
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
