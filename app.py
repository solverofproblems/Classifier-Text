import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import nltk

# 1. Preparação e Carga (Amostra de 200k para ser rápido)
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 header=None, names=cols, encoding='latin-1')

# Simplificando: apenas texto e alvo (0=Neg, 4=Pos)
df = df[['sentiment', 'text']].sample(200000, random_state=42)
df['sentiment'] = df['sentiment'].replace(4, 1) 

# 2. Limpeza Ultrarrápida
def simple_clean(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+|https?://\S+|[^a-zA-Z\s]', '', text)
    return text

df['text'] = df['text'].apply(simple_clean)

# 3. Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# 4. Vetorização (Onde a mágica acontece)
# Usamos n-grams (1,2) para capturar "not good" como uma unidade
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=50000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Treino do Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Validação
y_pred = model.predict(X_test_tfidf)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

frase = ["I am having a terrible day, everything is going wrong"]
frase_vetorizada = vectorizer.transform(frase)
resultado = model.predict(frase_vetorizada)
print("Positivo" if resultado[0] == 1 else "Negativo")