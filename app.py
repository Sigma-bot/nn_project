import streamlit as st

# Основная функция
def main():
    st.title("NN Project: Image Classification & Object Detection")

    # Добавляем выбор между страницами
    page = st.sidebar.selectbox("Select a page:", ("Image Classification", "Object Detection"))

    if page == "Image Classification":
        # Импортируем и запускаем страницу классификации изображений
        import pages.image_classification as image_classification
        image_classification.main()

    elif page == "Object Detection":
        # Импортируем и запускаем страницу обнаружения объектов
        import pages.object_detections as object_detections
        object_detections.main()

if __name__ == "__main__":
    main()
