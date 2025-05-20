import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load Spanish language model for spaCy
try:
    nlp = spacy.load('es_core_news_sm')
except:
    print("Downloading Spanish language model...")
    spacy.cli.download('es_core_news_sm')
    nlp = spacy.load('es_core_news_sm')

def preprocess_text(text):
    """
    Preprocess the text by removing URLs, mentions, hashtags, and special characters
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def lemmatize_text(text):
    """
    Lemmatize the text using spaCy
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def main():
    # Load the data
    print("Loading data...")
    df = pd.read_csv('data.csv')
    
    # Preprocess the text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Lemmatize the text
    print("Lemmatizing text...")
    df['lemmatized_text'] = df['processed_text'].apply(lemmatize_text)
    
    # Split the data
    X = df['lemmatized_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize models
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear'),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        }
        
        # Print classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Print comparison of models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    
    # Save results to CSV
    comparison_df.to_csv('model_comparison_results.csv')

if __name__ == "__main__":
    main() 