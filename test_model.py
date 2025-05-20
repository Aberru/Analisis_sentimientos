import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def test_model():
    try:
        # Cargar el modelo y vectorizador
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            
        # Texto de prueba
        test_text = "Estoy feliz"
        
        # Vectorizar el texto
        text_vectorized = vectorizer.transform([test_text])
        
        # Realizar la predicción
        prediction = model.predict(text_vectorized)
        probabilities = model.predict_proba(text_vectorized)
        
        # Imprimir información detallada
        print("\n=== Información del Modelo ===")
        print(f"Tipo de modelo: {type(model)}")
        print(f"Clases del modelo: {model.classes_}")
        
        print("\n=== Información de la Predicción ===")
        print(f"Tipo de predicción: {type(prediction)}")
        print(f"Predicción: {prediction}")
        print(f"Forma de la predicción: {prediction.shape}")
        
        print("\n=== Información de las Probabilidades ===")
        print(f"Tipo de probabilidades: {type(probabilities)}")
        print(f"Probabilidades: {probabilities}")
        print(f"Forma de las probabilidades: {probabilities.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_model() 