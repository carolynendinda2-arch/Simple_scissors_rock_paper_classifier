import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gc

torch.set_num_threads(2)

class RPSmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12288, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = RPSmodel()
    model.load_state_dict(torch.load("scissors_rock_paper.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

classes = ["scissors", "rock", "paper"]

st.title("✊✋✌ Rock Paper Scissors Classifier")
st.caption("Built by Carolyne Ndinda")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔮 Predict", type="primary"):
        with st.spinner("Predicting..."):
            gc.collect()
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100

            prediction = classes[pred_idx]
            emoji = {"rock": "✊", "paper": "✋", "scissors": "✌️"}.get(prediction, "❓")
            st.success(f"{emoji} **{prediction.upper()}**")
            st.info(f"Confidence: {confidence:.1f}%")
