import pickle
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def save_model_and_vectorizer(model, vectorizer, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """
    Guarda el modelo y el vectorizador de manera segura.
    
    Args:
        model: El modelo CatBoost entrenado
        vectorizer: El vectorizador TF-IDF ajustado
        model_path: Ruta donde guardar el modelo
        vectorizer_path: Ruta donde guardar el vectorizador
    """
    try:
        # Verificar que el vectorizador esté ajustado
        if not hasattr(vectorizer, 'vocabulary_'):
            raise ValueError("El vectorizador no está ajustado. Asegúrate de haber llamado a fit_transform antes de guardar.")
            
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

if __name__ == "__main__":
    # Ejemplo de uso:
    # Después de entrenar el modelo y el vectorizador:
    # save_model_and_vectorizer(cb_model, tfidf_vectorizer)
    pass 