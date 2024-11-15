import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
from PIL import Image
from torchvision import transforms
import json

# Загрузка модели
# model = torch.load('model_full.pth', map_location=torch.device('cpu'))
model = models.resnet152(pretrained=False)

model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)

# Загрузка весов модели, сохраненных ранее
try:
    model.load_state_dict(torch.load('models/model_resnet152.pth', map_location=torch.device('cpu')), weights_only=True)
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")

# Перевод модели в режим оценки
model.eval()


# Функция для предсказания
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

label_map = {v: k for k, v in label_map.items()}

# Функция для предсказания
def predict_image(image):
    # Преобразования для изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = transform(image).unsqueeze(0)  # Добавляем batch размер

    # Перемещаем модель и изображение на нужное устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Предсказание
    with torch.no_grad():
        output = model(image)
    
    # Получаем индексы и вероятности для топ-5 предсказаний
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    predictions = []
    for i in range(top5_prob.size(0)):
        class_id = top5_catid[i].item()
        class_name = label_map.get(class_id, f"Unknown Class {class_id}")
        score = top5_prob[i].item() * 100  # Преобразуем в проценты
        predictions.append(f"Prediction (index {class_id}, name {class_name}), Score: {score:.4f}")

    return "\n".join(predictions)

# Интерфейс Streamlit
st.title('Streamlit with PyTorch Model')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image)
    st.text(prediction)