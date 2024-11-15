import gdown
import os

# Список файлов с их корректными ссылками
files = {
    "models/alexnet_weights.pth": "https://drive.google.com/uc?id=10I8HNd0PBovRy_cXSkl4IBcdhfE9i1LK",
    "models/model_resnet152.pth": "https://drive.google.com/uc?id=1AF0y7ZSM_FT5P9bAmehjXnj1HaKhtOiH",
    "models/densenet121_weights.pth": "https://drive.google.com/uc?id=1IvntLfrv2wgMYmj8YSi6h3DGuXGXfPUV",
}

# Скачиваем файлы
for filepath, url in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Downloading {filepath}...")
    gdown.download(url, filepath, quiet=False)
