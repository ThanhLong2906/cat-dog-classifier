import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import io
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
    img = Image.open(io.BytesIO(img)).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(img)
        label = ["Cat", "Dog"][torch.argmax(preds).item()]
    return {label: float(torch.softmax(preds, dim=1).max().item())}

demo = gr.Interface(fn=predict, inputs="image", outputs="label", title="Cat vs Dog Classifier")

if __name__ == "__main__":
    demo.launch()
