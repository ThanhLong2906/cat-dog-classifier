import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cat_dog.pth", map_location="cpu"))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict(img):
    # ✅ Nếu img là NumPy array (Gradio mặc định), chuyển sang PIL
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'), 'RGB')

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        preds = model(img)
        probs = torch.softmax(preds, dim=1).flatten()
        labels = ["Cat", "Dog"]
        return {labels[i]: float(probs[i]) for i in range(2)}

# Giao diện Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2),
    title="🐱 Cat vs 🐶 Dog Classifier",
    description="Upload an image of a cat or dog and let the model predict!"
)

if __name__ == "__main__":
    demo.launch()
