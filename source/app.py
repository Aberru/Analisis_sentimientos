from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import sys
from pathlib import Path

# Obtener el directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from source.preprocessing import preprocess_text

app = FastAPI(
    title="Sentiment Analysis API",
    description="API para análisis de sentimientos en español",
    version="1.0.0"
)

# Cargar el modelo y el vectorizador
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    raise Exception("Los archivos model.pkl y vectorizer.pkl deben estar en el directorio raíz")

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predice el sentimiento de un texto en español.
    
    Args:
        input_data (TextInput): Texto a analizar
        
    Returns:
        SentimentResponse: Predicción del sentimiento y su confianza
    """
    try:
        # Preprocesar el texto
        processed_text = preprocess_text(input_data.text)
        
        # Vectorizar el texto
        text_vectorized = vectorizer.transform([processed_text])
        
        # Realizar la predicción
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized).max()
        
        return SentimentResponse(
            sentiment=prediction,
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Endpoint raíz que devuelve información sobre la API.
    """
    return {
        "message": "Bienvenido a la API de Análisis de Sentimientos",
        "endpoints": {
            "/predict": "POST - Analiza el sentimiento de un texto",
            "/docs": "GET - Documentación de la API"
        }
    } 