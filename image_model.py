import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("xray_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output, dim=1)

    return {
        "Normal": float(prob[0][0]),
        "Pneumonia": float(prob[0][1])
    }
