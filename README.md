# Análisis de Sentimientos con Streamlit y FastAPI

Este proyecto implementa un sistema de análisis de sentimientos con dos interfaces diferentes: una aplicación web interactiva (Streamlit) y una API REST (FastAPI). El sistema permite analizar el sentimiento de textos en español, clasificándolos en seis categorías emocionales diferentes.

## Características

### Aplicación Web (Streamlit)
- Interfaz web intuitiva y fácil de usar
- Análisis de sentimientos en tiempo real
- Visualización interactiva de resultados
- Muestra probabilidades para cada categoría emocional
- Diseño responsivo y amigable

### API REST (FastAPI)
- Endpoint para análisis de sentimientos
- Documentación automática (Swagger UI)
- Respuestas en formato JSON
- Fácil integración con otras aplicaciones
- Manejo de errores robusto

## Modelo y Preprocesamiento

### Modelo Utilizado
El sistema utiliza un modelo CatBoostClassifier, un algoritmo de gradient boosting que ofrece excelente rendimiento en tareas de clasificación. El modelo clasifica los textos en seis categorías emocionales:
- Peaceful (Paz)
- Mad (Enojo)
- Powerful (Poder)
- Sad (Tristeza)
- Joyful (Alegría)
- Scared (Miedo)

El modelo proporciona probabilidades para cada categoría emocional, permitiendo un análisis más detallado del sentimiento del texto.

### Proceso de Preprocesamiento
El texto pasa por las siguientes etapas de preprocesamiento:

1. **Limpieza de Texto**:
   - Eliminación de URLs
   - Eliminación de menciones (@usuario)
   - Eliminación de hashtags (#)
   - Eliminación de caracteres especiales y números
   - Conversión a minúsculas

2. **Tokenización**:
   - División del texto en palabras individuales
   - Eliminación de palabras vacías (stopwords) en español
   - Aplicación de lematización

3. **Vectorización**:
   - Transformación del texto a vectores numéricos usando TF-IDF
   - Normalización de los vectores

## Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Clona este repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd Analisis_sentimientos
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
Analisis_sentimientos/
├── source/
│   ├── app.py              # API REST con FastAPI
│   ├── app_streamlit.py    # Aplicación web con Streamlit
│   ├── preprocessing.py    # Módulo de preprocesamiento de texto
│   └── model.py           # Módulo del modelo de análisis de sentimientos
├── requirements.txt       # Dependencias del proyecto
└── README.md            # Este archivo
```

## Uso

### Aplicación Web (Streamlit)

1. Inicia la aplicación Streamlit:
```bash
streamlit run source/app_streamlit.py
```

2. Abre tu navegador web y accede a la URL mostrada en la terminal (generalmente http://localhost:8501)

3. Ingresa el texto que deseas analizar en el campo de texto

4. Haz clic en "Analizar Sentimiento" para obtener los resultados

### API REST (FastAPI)

1. Inicia el servidor FastAPI:
```bash
uvicorn source.app:app --reload
```

2. Accede a la documentación de la API en http://localhost:8000/docs

3. Utiliza el endpoint `/predict` para analizar sentimientos:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Tu texto aquí"}'
```

## Tecnologías Utilizadas

- Streamlit: Framework para la interfaz de usuario web
- FastAPI: Framework para la API REST
- CatBoost: Modelo de machine learning
- NLTK: Procesamiento de lenguaje natural
- Scikit-learn: Vectorización y procesamiento de texto
- Pandas: Manipulación de datos
- Matplotlib: Visualización de datos

## Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz un fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

[Tu Nombre] - [Tu Email]

Link del Proyecto: [URL_DEL_REPOSITORIO]