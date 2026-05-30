import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(page_title="Reconocimiento MNIST", page_icon="🔢")

# Cargar el modelo en caché para que no se recargue cada vez que interactúas con el tablero
@st.cache_resource
def cargar_modelo():
    try:
        return tf.keras.models.load_model('modelo_mnist.h5')
    except OSError:
        st.error("No se encontró el archivo 'modelo_mnist.h5'. Por favor, corre el script de entrenamiento primero.")
        st.stop()

modelo = cargar_modelo()

st.title("🔢 Reconocimiento de Dígitos MNIST")
st.write("Dibuja un número del **0 al 9** en el recuadro negro de abajo y presiona predecir.")

# Configurar el lienzo (Canvas)
# Lo hacemos de 280x280 (múltiplo de 28) con fondo negro y trazo blanco para simular las imágenes de MNIST
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('🧠 Predecir Dígito'):
    if canvas_result.image_data is not None:
        # Extraer la imagen del canvas (Viene en formato RGBA)
        img = canvas_result.image_data
        
        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Verificar que el usuario realmente haya dibujado algo
        if np.sum(img_gray) == 0:
            st.warning("El lienzo está vacío. Por favor dibuja un número.")
        else:
            # Redimensionar la imagen de 280x280 a 28x28 (como fue entrenado el modelo)
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalizar los valores (0.0 - 1.0) y darle la forma correcta (1, 28, 28, 1)
            img_normalized = img_resized / 255.0
            img_reshaped = np.expand_dims(img_normalized, axis=(0, -1))

            # Hacer la predicción con la Red Neuronal
            prediccion = modelo.predict(img_reshaped)
            digito_predicho = np.argmax(prediccion)
            confianza = np.max(prediccion)

            # Mostrar resultados
            st.success(f"### El número predicho es: **{digito_predicho}**")
            st.info(f"Nivel de confianza: {confianza:.2%}")
            
            # Mostrar la imagen de 28x28 que vio la red neuronal internamente
            st.write("Así vio tu dibujo la red neuronal (28x28 píxeles):")
            st.image(img_resized, width=150)