import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import time
import requests
import json

# Загрузка моделей
alexnet = models.alexnet(pretrained=True)
densenet = models.densenet121(pretrained=True)

# Перевод моделей в режим оценки
alexnet.eval()
densenet.eval()

# Загрузка меток классов (например, для ImageNet)
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
response = requests.get(LABELS_URL)
labels = json.loads(response.text)

# Функция для загрузки изображения и преобразования его
def load_image(image_file):
    image = Image.open(image_file)
    return image

# Функция для загрузки изображения по ссылке
def load_image_from_url(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        return image
    except:
        st.error("Error loading image from URL. Please check the URL.")
        return None

# Функция для предсказания с использованием модели
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # добавляем батч-измерение
    with torch.no_grad():
        output = model(image)
    return output

# Основная функция для работы с интерфейсом Streamlit
def main():
    st.markdown("<h1 style='text-align: center;'>Image Classification with AlexNet and DenseNet121</h1>", unsafe_allow_html=True)

    # Опции загрузки файла
    image_source = st.radio("Select image source:", ("Upload image", "Provide image URL"))

    image = None  # Инициализация переменной image как None

    if image_source == "Upload image":
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image = load_image(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        image_url = st.text_input("Enter image URL:")
        if image_url:
            image = load_image_from_url(image_url)
            if image:
                st.image(image, caption="Image from URL", use_container_width=True)

    if image is not None:
        # Запуск классификации
        start_time = time.time()

        # Предсказания для AlexNet
        alexnet_start_time = time.time()
        alexnet_output = predict_image(image, alexnet)
        alexnet_probs = torch.nn.functional.softmax(alexnet_output[0], dim=0)
        alexnet_pred = torch.topk(alexnet_probs, 5)
        alexnet_end_time = time.time()

        # Предсказания для DenseNet121
        densenet_start_time = time.time()
        densenet_output = predict_image(image, densenet)
        densenet_probs = torch.nn.functional.softmax(densenet_output[0], dim=0)
        densenet_pred = torch.topk(densenet_probs, 5)
        densenet_end_time = time.time()

        end_time = time.time()
        total_processing_time = end_time - start_time

        # Оформляем результаты
        st.markdown("<h3 style='text-align: center;'>AlexNet Predictions</h3>", unsafe_allow_html=True)
        for i in range(5):
            label = labels[str(alexnet_pred.indices[i].item())][1]
            st.write(f"{label}: {alexnet_pred.values[i].item():.4f}")
        
        # Время обработки для AlexNet
        alexnet_time = alexnet_end_time - alexnet_start_time
        st.write(f"AlexNet processing time: {alexnet_time:.2f} seconds")
        
        st.markdown("<h3 style='text-align: center;'>DenseNet121 Predictions</h3>", unsafe_allow_html=True)
        for i in range(5):
            label = labels[str(densenet_pred.indices[i].item())][1]
            st.write(f"{label}: {densenet_pred.values[i].item():.4f}")
        
        # Время обработки для DenseNet121
        densenet_time = densenet_end_time - densenet_start_time
        st.write(f"DenseNet121 processing time: {densenet_time:.2f} seconds")
        
        # Общее время
        st.write(f"Total processing time: {total_processing_time:.2f} seconds")

if __name__ == "__main__":
    main()
