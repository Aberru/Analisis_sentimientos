#Preprocesamiento del texto
from datetime import date
from math import ulp
import re
from string import capwords

from catboost import train

def clean_text(text: str) -> str:
    if isinstance(text, str):
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones (@usuario)
        text = re.sub(r'@\w+', '', text)
        
        # Eliminar hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Eliminar caracteres especiales, números y puntuaciones
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ''
train
x_train_clean = x_train.apply(clean_text)
x_test_clean = x_test.apply(clean_text)

# Cargar stopwords en español
spanish_stopwords = capwords.words('spanish')

# Añadir algunas stopwords adicionales específicas 
additional_stopwords = ['si', 'xq', 'q', 'x', 'k', 'qe', 'pq', 'rt', 'xa', 'xo', 'jaja', 'jajaja', 'jajajaja']
spanish_stopwords.extend(additional_stopwords)


# Función para aplicar lematización y eliminar stopwords
def tokenize_aulplemmatize(text):
    if isinstance(text, str):
        # Procesar texto con spaCy
        doc = nlp(text)

        # Eliminar stopwords y puntuación, aplicar lematización
        lemmas = [token.lemma_ for token in doc if token.lemma_.lower() not in spanish_stopwords and len(token.lemma_) > 2 and not token.is_punct]

        return ' '.join(lemmas)
    else:
        return ''
    

# Aplicar tokenización y lematización
x_train_final = x_train_clean.apply(tokenize_and_lemmatize)
x_test_final = x_test_clean.apply(tokenize_and_lemmatize)


x = data['text']
y = date['sentiment']
