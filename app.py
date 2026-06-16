import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sqlite3
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import os

# 1. Настройки страницы
st.set_page_config(page_title="Умная Парковка", page_icon="🚗", layout="centered")
st.title("🚗 Детектор свободных мест")
st.write("Загрузите фото фрагмента парковки, и нейросеть определит статус места.")

# --- НАСТРОЙКА БАЗЫ ДАННЫХ SQLite ---
def init_db():
    conn = sqlite3.connect('parking_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  filename TEXT,
                  result TEXT,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def add_record(filename, result, confidence):
    conn = sqlite3.connect('parking_history.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (date, filename, result, confidence) VALUES (?, ?, ?, ?)",
              (date_str, filename, result, confidence))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('parking_history.db')
    # Сортируем так, чтобы свежие проверки были сверху
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

# Инициализируем БД при старте приложения
init_db()

# --- ФУНКЦИЯ СОЗДАНИЯ PDF ОТЧЕТА ---
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    
    # Заголовок
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(200, 10, txt="Parking Detection Analytics Report", ln=True, align="C")
    pdf.ln(10)
    
    # Общая статистика
    pdf.set_font("helvetica", size=12)
    pdf.cell(200, 10, txt=f"Total requests processed: {len(df)}", ln=True)
    
    occupied = len(df[df['result'] == 'Занято (Occupied)'])
    empty = len(df[df['result'] == 'Свободно (Empty)'])
    
    pdf.cell(200, 10, txt=f"Occupied spots detected: {occupied}", ln=True)
    pdf.cell(200, 10, txt=f"Empty spots detected: {empty}", ln=True)
    pdf.ln(10)
    
    # Детализация (последние 20 записей, чтобы не рвать страницы)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(200, 10, txt="Recent History (Top 20):", ln=True)
    
    pdf.set_font("helvetica", size=10)
    for index, row in df.head(20).iterrows():
        # Адаптируем русский ответ под английский PDF
        res_en = "Occupied" if "Занято" in row['result'] else "Empty"
        pdf.cell(200, 8, txt=f"[{row['date']}] File: {row['filename']} | Status: {res_en} | Conf: {row['confidence']:.1f}%", ln=True)
        
    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        return f.read()

# 2. Функция загрузки модели-победителя
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2) 
    model.load_state_dict(torch.load('models/EfficientNetB0.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# 3. Подготовка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Свободно (Empty)', 'Занято (Occupied)']

try:
    model = load_model()
    st.success("✅ Модель EfficientNetB0 (Точность: 99.9%) успешно загружена в память!")
except Exception as e:
    st.error(f"⚠️ Ошибка загрузки модели: {e}. Проверьте наличие файла EfficientNetB0.pth в папке models.")
    st.stop()

# 4. Интерфейс загрузки файла
uploaded_file = st.file_uploader("Выберите фото (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное фото', width=300)
    
    if st.button('Анализировать', type="primary"):
        img_t = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_t)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, predicted = torch.max(probabilities, 0)
        
        result_class = classes[predicted.item()]
        confidence = conf.item() * 100
        
        # Записываем результат в базу данных
        add_record(uploaded_file.name, result_class, confidence)
        
        st.divider()
        if predicted.item() == 0:
            st.success(f"🟩 **Вердикт:** Место {result_class}")
        else:
            st.error(f"🟥 **Вердикт:** Место {result_class}")
            
        st.info(f"Уверенность нейросети: **{confidence:.2f}%**")

# 5. БЛОК ИСТОРИИ И ОТЧЕТА
st.divider()
st.subheader("📊 История проверок и статистика")

history_df = get_history()

if not history_df.empty:
    # Отрисовываем красивую интерактивную таблицу в интерфейсе
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Кнопка скачивания PDF
    pdf_bytes = generate_pdf(history_df)
    st.download_button(
        label="📄 Скачать отчет (PDF)",
        data=pdf_bytes,
        file_name=f"parking_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
    )
else:
    st.write("История пока пуста. Загрузите и проанализируйте первое изображение!")

    # --- БОКОВАЯ ПАНЕЛЬ (ТЕХНИЧЕСКИЙ ПАСПОРТ МОДЕЛИ) ---
with st.sidebar:
    st.header("⚙️ Паспорт нейросети")
    st.markdown("""
    **Текущая модель:** `EfficientNetB0`
    
    Эталонные метрики качества на валидационной выборке (2000 изображений):
    * **Accuracy:** 99.90%
    * **Precision:** 99.90%
    * **Recall:** 99.90%
    * **F1-score:** 99.90%
    """)
    
    st.divider()
    st.caption("Обучение проводилось на адаптированном датасете PKLot (18 000 изображений). Инференс выполняется на CPU.")