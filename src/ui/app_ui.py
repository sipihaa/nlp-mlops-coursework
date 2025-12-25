import streamlit as st
import requests
import os


API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="VK Classifier")

st.title("Классификация постов VK")
st.markdown("Определяет категорию поста: **Авиация** или **Автотранспорт**.")

# Поле ввода
text = st.text_area("Введите текст поста:", height=150, placeholder="Например: Самолет вылетел из Шереметьево...")

if st.button("Предсказать"):
    if not text.strip():
        st.warning("Пожалуйста, введите текст.")
    else:
        try:
            with st.spinner("Анализируем..."):
                response = requests.post(f"{API_URL}/predict", json={"text": text})
                
                if response.status_code == 200:
                    result = response.json()
                    label = result.get("label")
                    conf = result.get("confidence")
                    
                    st.success(f"Категория: **{label}**")
                    st.metric("Уверенность модели", f"{conf:.2%}")
                    
                    st.progress(conf)
                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
                    st.text(response.text)
                    
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к сервису предсказаний. Проверьте, запущен ли FastAPI.")
