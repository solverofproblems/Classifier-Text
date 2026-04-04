import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.naive_bayes import MultimodalNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']

df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                encoding='latin-1',names=cols,header=None)

df = df[['sentiment', 'text']].sample(200000, random_state=42)
df['sentimental'] = df['sentimental'].replace(4,1)


def simple_clean(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+|https?://\S+|[^a-zA-Z\s]', '', text)
    return text

df['text'] = df['text'].apply(simple_clean)

X_train, X_test, y_train, y_test = trian_test_split(
    df['text'], df['sentimental'], test_size=0.2, random_state = 42
)

vectorizer = TfidVectorizer(stop_words=stop_words, max_features=50000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transforma(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultimodalNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(f'Acurácia: {accuracy_score(y_test, y_pred):.2f}%')
print(classification_report(y_test, y_pred))