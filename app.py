import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1. Настройки страницы
st.set_page_config(page_title="Умная Парковка", page_icon="🚗", layout="centered")
st.title("🚗 Детектор свободных мест")
st.write("Загрузите фото фрагмента парковки, и нейросеть определит статус места.")

# 2. Функция загрузки модели-победителя (кэшируется, чтобы не грузить каждый раз)
@st.cache_resource
def load_model():
    # Создаем архитектуру чемпиона - EfficientNetB0
    model = models.efficientnet_b0(weights=None)
    
    # Меняем последний слой классификатора под наши 2 класса
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2) 
    
    # Загружаем лучшие веса (используем процессор, так надежнее для веб-сервера)
    model.load_state_dict(torch.load('models/EfficientNetB0.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. Подготовка изображения (строго как при обучении)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Свободно (Empty)', 'Занято (Occupied)']

# Пытаемся загрузить модель при старте приложения
try:
    model = load_model()
    st.success("✅ Модель EfficientNetB0 (Точность: 99.9%) успешно загружена в память!")
except Exception as e:
    st.error(f"⚠️ Ошибка загрузки модели: {e}. Проверьте наличие файла EfficientNetB0.pth в папке models.")
    st.stop()

# 4. Интерфейс загрузки файла
uploaded_file = st.file_uploader("Выберите фото (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем и показываем картинку
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное фото', width=300)
    
    # Кнопка старта
    if st.button('Анализировать', type="primary"):
        # Превращаем картинку в тензор
        img_t = transform(image).unsqueeze(0)
        
        # Прогоняем через нейросеть (без расчета градиентов для скорости)
        with torch.no_grad():
            outputs = model(img_t)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, predicted = torch.max(probabilities, 0)
        
        # Результаты
        result_class = classes[predicted.item()]
        confidence = conf.item() * 100
        
        # Красивый вывод с цветовой индикацией
        st.divider()
        if predicted.item() == 0:
            st.success(f"🟩 **Вердикт:** Место {result_class}")
        else:
            st.error(f"🟥 **Вердикт:** Место {result_class}")
            
        st.info(f"Уверенность нейросети: **{confidence:.2f}%**")