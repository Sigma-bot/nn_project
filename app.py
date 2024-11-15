import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
from PIL import Image
from torchvision import transforms
import json
import os
import gdown

# Функция для загрузки моделей
def download_models():
    files = {
        "models/alexnet_weights.pth": "https://drive.google.com/uc?export=download&id=10I8HNd0PBovRy_cXSkl4IBcdhfE9i1LK",
        "models/model_resnet152.pth": "https://drive.google.com/uc?export=download&id=1AF0y7ZSM_FT5P9bAmehjXnj1HaKhtOiH",
        "models/densenet121_weights.pth": "https://drive.google.com/uc?export=download&id=1IvntLfrv2wgMYmj8YSi6h3DGuXGXfPUV",
    }

    os.makedirs("models", exist_ok=True)

    for filepath, url in files.items():
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            gdown.download(url, filepath, quiet=False)
        else:
            print(f"{filepath} already exists.")

# Загрузка модели
def load_model():
    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)

    model_path = 'models/model_resnet152.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Model file {model_path} not found!")
        st.error(f"Model file {model_path} not found! Please check the file path.")

    return model

# Функция для предсказания
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)  # Добавляем batch размер

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    predictions = []
    with open('label_map.json', 'r') as f:
        label_map = json.load(f)

    label_map = {v: k for k, v in label_map.items()}

    for i in range(top5_prob.size(0)):
        class_id = top5_catid[i].item()
        class_name = label_map.get(class_id, f"Unknown Class {class_id}")
        score = top5_prob[i].item() * 100  # Преобразуем в проценты
        predictions.append(f"Prediction (index {class_id}, name {class_name}), Score: {score:.4f}")

    return "\n".join(predictions)

# Интерфейс Streamlit
def main():
    # Загружаем модели перед запуском приложения
    download_models()

    # Загружаем модель
    model = load_model()

    # Основной интерфейс Streamlit
    st.title("NN Project: Image Classification & Object Detection")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict_image(image, model)
        st.text(prediction)

if __name__ == "__main__":
    main()
