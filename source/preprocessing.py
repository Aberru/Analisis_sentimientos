import re
import spacy
from typing import Union
import pandas as pd
import nltk
from nltk.corpus import stopwords
import unicodedata
from nltk.tokenize import word_tokenize
import numpy as np

# Descargar recursos necesarios de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando URLs, menciones, hashtags y caracteres especiales.
    
    Args:
        text (str): Texto a limpiar
        
    Returns:
        str: Texto limpio
    """
    if not isinstance(text, str):
        return ''
        
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

# Cargar stopwords en español
spanish_stopwords = set(stopwords.words('spanish'))

# Añadir algunas stopwords adicionales específicas 
additional_stopwords = ['si', 'xq', 'q', 'x', 'k', 'qe', 'pq', 'rt', 'xa', 'xo', 'jaja', 'jajaja', 'jajajaja']
spanish_stopwords.update(additional_stopwords)

def tokenize_and_lemmatize(text: str) -> str:
    """
    Aplica tokenización y lematización al texto, eliminando stopwords.
    
    Args:
        text (str): Texto a procesar
        
    Returns:
        str: Texto procesado
    """
    if not isinstance(text, str):
        return ''
        
    # Procesar texto con spaCy
    doc = nlp(text)
    
    # Eliminar stopwords y puntuación, aplicar lematización
    lemmas = [token.lemma_ for token in doc 
             if token.lemma_.lower() not in spanish_stopwords 
             and len(token.lemma_) > 2 
             and not token.is_punct]
    
    return ' '.join(lemmas)

def preprocess_text(text):
    """
    Preprocesa el texto para el análisis de sentimientos.
    
    Args:
        text: Texto a preprocesar
        
    Returns:
        str: Texto preprocesado
    """
    # Convertir a string si no lo es
    if not isinstance(text, str):
        text = str(text)
    
    # Si el texto está vacío, devolver cadena vacía
    if not text.strip():
        return ""
    
    try:
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar acentos
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Eliminar stopwords
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Unir tokens en texto y asegurarse de que sea string
        return str(' '.join(tokens))
        
    except Exception as e:
        print(f"Error en preprocess_text: {str(e)}")
        return str(text)  # En caso de error, devolver el texto original como string 