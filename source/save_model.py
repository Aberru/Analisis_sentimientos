import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from preprocessing import preprocess_text

def save_model_and_vectorizer(model, vectorizer, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """
    Guarda el modelo y el vectorizador de manera segura.
    
    Args:
        model: El modelo entrenado
        vectorizer: El vectorizador TF-IDF ajustado
        model_path: Ruta donde guardar el modelo
        vectorizer_path: Ruta donde guardar el vectorizador
    """
    try:
        # Guardar el modelo
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo guardado exitosamente en {model_path}")
        
        # Guardar el vectorizador
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Vectorizador guardado exitosamente en {vectorizer_path}")
        
    except Exception as e:
        print(f"Error al guardar los archivos: {str(e)}")

# Ejemplo de uso:
if __name__ == "__main__":
    # Aquí deberías cargar tu modelo y vectorizador entrenados
    # Por ejemplo:
    # save_model_and_vectorizer(cb_model, tfidf_vectorizer)
    pass 