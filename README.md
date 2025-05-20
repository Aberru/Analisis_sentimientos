# Análisis de Sentimientos

Este proyecto realiza un análisis de sentimientos sobre tweets en español, utilizando técnicas de procesamiento de lenguaje natural y algoritmos de aprendizaje automático.

## Descripción

El análisis de sentimientos es una técnica de procesamiento de lenguaje natural que identifica y extrae información subjetiva de textos para determinar la actitud, opiniones o emociones del autor respecto a un tema. Este proyecto utiliza un dataset de tweets en español con etiquetas de emociones y sentimientos para construir modelos predictivos.

## Contenido del Proyecto

- `data.csv`: Dataset con tweets, emociones y sentimientos.
- `Notebooks/analisis_sentimientos.ipynb`: Notebook de Jupyter con el análisis completo.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## Características

- Exploración y visualización de datos
- Preprocesamiento de texto en español:
  - Limpieza de texto (URLs, menciones, hashtags, caracteres especiales)
  - Tokenización
  - Eliminación de stopwords
  - Stemming
- Vectorización mediante TF-IDF
- Entrenamiento y comparación de varios modelos:
  - Naive Bayes
  - SVM (Support Vector Machine)
  - KNN (K-Nearest Neighbors)
- Optimización de hiperparámetros
- Evaluación de modelos usando métricas como precisión, recall y F1-score

## Instalación

1. Clona este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Asegúrate de descargar los recursos de NLTK necesarios:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Uso

Abre el notebook `Notebooks/analisis_sentimientos.ipynb` con Jupyter para ver el análisis completo:

```bash
jupyter notebook Notebooks/analisis_sentimientos.ipynb
```

## Resultados

El notebook incluye:
- Análisis exploratorio del dataset
- Evaluación comparativa de diferentes algoritmos
- Optimización del mejor modelo
- Función para predecir el sentimiento de nuevos textos

## Licencia

Este proyecto está disponible bajo la licencia MIT.