import streamlit as st
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess_text

# Diccionario de traducción de sentimientos
SENTIMENT_TRANSLATIONS = {
    'peaceful': 'Paz',
    'mad': 'Enojo',
    'powerful': 'Poder',
    'sad': 'Tristeza',
    'joyful': 'Alegría',
    'scared': 'Miedo'
}

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="😊",
    layout="wide"
)

# Título y descripción
st.title("📊 Análisis de Sentimientos en Español")
st.markdown("""
Esta aplicación analiza el sentimiento de textos en español utilizando un modelo de machine learning.
Simplemente ingresa tu texto y obtendrás la predicción junto con las probabilidades para cada sentimiento.
""")

@st.cache_resource
def load_model():
    """Carga el modelo y el vectorizador."""
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            
        # Verificar que el vectorizador esté configurado correctamente
        if not hasattr(vectorizer, 'vocabulary_'):
            st.error("El vectorizador no está correctamente configurado. Falta el atributo 'vocabulary_'")
            return None, None
            
        # Verificar que el modelo esté configurado correctamente
        if not hasattr(model, 'classes_'):
            st.error("El modelo no está correctamente configurado. Falta el atributo 'classes_'")
            return None, None
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None

def predict_sentiment(text, model, vectorizer):
    """Realiza la predicción del sentimiento."""
    try:
        # Verificar que el vectorizador y el modelo estén cargados
        if vectorizer is None or model is None:
            st.error("El modelo o el vectorizador no están cargados correctamente")
            return None, None
            
        # Asegurarse de que el texto sea una cadena
        if not isinstance(text, str):
            text = str(text)
            
        # Preprocesar el texto
        processed_text = preprocess_text(text)
        
        # Verificar que el texto procesado no esté vacío
        if not processed_text:
            st.warning("El texto procesado está vacío. Por favor, ingresa un texto válido.")
            return None, None
            
        # Asegurarse de que el texto procesado sea una cadena
        if not isinstance(processed_text, str):
            processed_text = str(processed_text)
        
        # Vectorizar el texto
        try:
            text_vectorized = vectorizer.transform([processed_text])
        except Exception as e:
            st.error(f"Error al vectorizar el texto: {str(e)}")
            return None, None
        
        # Realizar la predicción
        try:
            prediction = model.predict(text_vectorized)
            probabilities = model.predict_proba(text_vectorized)
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")
            return None, None
        
        # Obtener la etiqueta predicha (extraer el valor del array numpy)
        predicted_sentiment = prediction[0][0]  # Obtener el primer elemento del array anidado
        
        # Obtener las probabilidades para cada clase
        class_names = model.classes_  # Usar las clases del modelo
        probs = probabilities[0] * 100
        
        # Crear un diccionario con las probabilidades
        prob_dict = {class_name: prob for class_name, prob in zip(class_names, probs)}
        
        return predicted_sentiment, prob_dict
        
    except Exception as e:
        st.error(f"Error al procesar el texto: {str(e)}")
        print(f"Error detallado: {str(e)}")  # Debug: imprimir el error detallado
        return None, None

# Cargar el modelo y el vectorizador
model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.error("No se pudo cargar el modelo. Por favor, asegúrate de que los archivos model.pkl y vectorizer.pkl existen y son válidos.")
    st.stop()

# Área de entrada de texto
text_input = st.text_area(
    "Ingresa el texto a analizar:",
    height=150,
    placeholder="Escribe o pega tu texto aquí..."
)

# Botón para realizar la predicción
if st.button("Analizar Sentimiento", type="primary"):
    if text_input:
        with st.spinner("Analizando el texto..."):
            sentiment, probabilities = predict_sentiment(text_input, model, vectorizer)
            
            if sentiment is not None:
                # Mostrar el resultado
                st.subheader("Resultado del Análisis")
                
                # Mostrar el sentimiento predicho en ambos idiomas
                st.success(f"El sentimiento del texto es: {sentiment} ({SENTIMENT_TRANSLATIONS[sentiment]})")
                
                # Mostrar las probabilidades en un gráfico
                st.subheader("Probabilidades por Clase")
                
                # Crear dos columnas para mostrar las probabilidades
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Peaceful (Paz)", f"{probabilities['peaceful']:.1f}%")
                    st.metric("Mad (Enojo)", f"{probabilities['mad']:.1f}%")
                    st.metric("Powerful (Poder)", f"{probabilities['powerful']:.1f}%")
                
                with col2:
                    st.metric("Sad (Tristeza)", f"{probabilities['sad']:.1f}%")
                    st.metric("Joyful (Alegría)", f"{probabilities['joyful']:.1f}%")
                    st.metric("Scared (Miedo)", f"{probabilities['scared']:.1f}%")
    else:
        st.warning("Por favor, ingresa un texto para analizar.")

# Pie de página
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desarrollado con ❤️ usando Streamlit</p>
</div>
""", unsafe_allow_html=True) 