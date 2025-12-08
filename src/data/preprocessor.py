import pandas as pd
import re
from nltk.corpus import stopwords
import pymorphy3
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


aviation_data = pd.read_csv('data/raw/aviation.csv')
road_transport_data = pd.read_csv('data/raw/road_transport.csv')

data = pd.concat([aviation_data, road_transport_data])

data = data[data['marked_as_ads'] == 0]

data = data[['text', 'y']]

data.dropna(inplace=True)
data.drop_duplicates(keep='first', inplace=True)

y_map = {
    'aviation': 0,
    'road_transport': 1
}
data['y'] = data['y'].map(y_map)

morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

garbage_list = [
    'http', 'https', 'vk', 'cc', 'com', 'ru', 'me', 'avtoobmen'
]
stop_words.update(garbage_list)


def preprocess_text(text, stop_words):
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'(https?://\S+)|(www\.\S+)|(\w+\.(com|net|org|ru|me|cc|to)/\S+)', ' ', text)
    text = re.sub(r'\[(id|club)\d+\|.*?\]', '', text)
    text = re.sub(r'[^a-zа-яё\s]', ' ', text)

    words = text.split()

    lemmas = []
    for word in words:
        if len(word) > 1:
            lemma = morph.parse(word)[0].normal_form
                
            if lemma not in stop_words:
                lemmas.append(lemma)

    return " ".join(lemmas)


print("\nНачало предобработки текста...")

tqdm.pandas()
data['processed_text'] = data['text'].progress_apply(
    lambda text: preprocess_text(text, stop_words)
)

print("Предобработка текста завершена.")

data.to_csv('data/processed/processed_data.csv')

model_name = 'cointegrated/rubert-tiny2'
emb_model = SentenceTransformer(model_name, device='cpu')

print("Модель загружена")

embeddings = emb_model.encode(data['processed_text'].tolist(), show_progress_bar=True)

data['embeddings'] = list(embeddings)

data.to_pickle('data/processed/data_with_embeddings.pkl')
print("Датасет с эмбеддингами сохранен!")
