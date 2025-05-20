import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def check_pickle_files():
    try:
        # Cargar el modelo
        print("\n=== Cargando model.pkl ===")
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        print(f"Tipo de modelo: {type(model)}")
        print(f"Atributos del modelo: {dir(model)}")
        
        # Cargar el vectorizador
        print("\n=== Cargando vectorizer.pkl ===")
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print(f"Tipo de vectorizador: {type(vectorizer)}")
        print(f"Atributos del vectorizador: {dir(vectorizer)}")
        
        # Probar el vectorizador con un texto simple
        print("\n=== Probando el vectorizador ===")
        test_text = "Estoy feliz"
        try:
            # Crear un nuevo vectorizador con los mismos parámetros
            new_vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2)
            )
            
            # Copiar los atributos necesarios
            new_vectorizer.vocabulary_ = vectorizer.vocabulary_
            new_vectorizer.idf_ = vectorizer.idf_
            
            # Probar la transformación
            transformed = new_vectorizer.transform([test_text])
            print(f"Transformación exitosa. Forma: {transformed.shape}")
            
            # Probar la predicción
            prediction = model.predict(transformed)
            print(f"Predicción: {prediction}")
            
        except Exception as e:
            print(f"Error al probar el vectorizador: {str(e)}")
            
    except Exception as e:
        print(f"Error al cargar los archivos: {str(e)}")

if __name__ == "__main__":
    check_pickle_files() 