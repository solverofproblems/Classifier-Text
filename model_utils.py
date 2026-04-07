import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Configuração do NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = list(stopwords.words('english'))

def simple_clean(text):
    """Limpa o texto removendo menções, links e caracteres não alfabéticos."""
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9]+|https?://\S+|[^a-zA-Z\s]', '', text)
    return text.strip()

class SentimentClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train(self, csv_path, sample_size=200000):
        """Treina o modelo com base no dataset CSV."""
        cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(csv_path, header=None, names=cols, encoding='latin-1')
        
        # Seleciona amostra e mapeia sentimentos (0=Neg, 4=Pos -> 0, 1)
        df = df[['sentiment', 'text']].sample(sample_size, random_state=42)
        df['sentiment'] = df['sentiment'].replace(4, 1)
        
        # Limpeza
        df['text'] = df['text'].apply(simple_clean)
        
        # Divisão
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        # Vetorização
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=50000, ngram_range=(1,2))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Treino
        self.model = MultinomialNB()
        self.model.fit(X_train_tfidf, y_train)
        
        return self.model, self.vectorizer

    def save(self, model_path='sentiment_model.joblib', vectorizer_path='vectorizer.joblib'):
        """Salva o modelo e o vetorizador em disco."""
        if self.model and self.vectorizer:
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
        else:
            raise ValueError("Modelo não treinado.")

    def load(self, model_path='sentiment_model.joblib', vectorizer_path='vectorizer.joblib'):
        """Carrega o modelo e o vetorizador do disco."""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        """Prediz o sentimento e a probabilidade de uma frase."""
        if not self.model or not self.vectorizer:
            raise ValueError("Modelo não carregado ou treinado.")
        
        cleaned_text = simple_clean(text)
        tfidf_text = self.vectorizer.transform([cleaned_text])
        
        prediction = self.model.predict(tfidf_text)[0]
        # Obter probabilidades para o percentual de confiança
        probabilities = self.model.predict_proba(tfidf_text)[0]
        confidence = probabilities[prediction]
        
        return {
            'sentiment': 'Positivo' if prediction == 1 else 'Negativo',
            'confidence': confidence,
            'label': prediction
        }
