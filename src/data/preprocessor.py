import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import pymorphy3
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import emoji


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

MORPH = pymorphy3.MorphAnalyzer()
STOP_WORDS = set(stopwords.words('russian'))

GARBAGE_LIST = ['http', 'https', 'vk', 'cc', 'com', 'ru', 'avtoobmen']
STOP_WORDS.update(GARBAGE_LIST)


def preprocess_text(text):
    if not isinstance(text, str): 
        return ""
    text = emoji.demojize(text, language='ru')
    text = text.lower()
    text = re.sub(r'(https?://\S+)|(www\.\S+)|(\w+\.(com|net|org|ru|me|cc|to)/\S+)', ' ', text)
    text = re.sub(r'\[(id|club)\d+\|.*?\]', '', text)
    text = text.replace(':', ' ').replace('_', ' ')
    text = re.sub(r'[^a-zа-яё\s]', ' ', text)

    words = text.split()

    lemmas = []
    for word in words:
        if len(word) > 1:
            lemma = MORPH.parse(word)[0].normal_form
                
            if lemma not in STOP_WORDS:
                lemmas.append(lemma)

    return " ".join(lemmas)


def run_preprocessing_pipeline(input_avia_path, input_auto_path, output_processed_data_path, output_embeddings_data_path):
    aviation_data = pd.read_csv(input_avia_path)
    road_transport_data = pd.read_csv(input_auto_path)

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

    print("\nНачало предобработки текста...")

    tqdm.pandas()
    data['processed_text'] = data['text'].progress_apply(preprocess_text)

    print("Предобработка текста завершена.")

    data.to_csv(output_processed_data_path)

    model_name = 'cointegrated/rubert-tiny2'
    emb_model = SentenceTransformer(model_name, device='cpu')

    print("Модель векторайзер загружена")

    embeddings = emb_model.encode(data['processed_text'].tolist(), show_progress_bar=True)

    data['embeddings'] = list(embeddings)

    data.to_pickle(output_embeddings_data_path)
    print("Датасет с эмбеддингами сохранен!")


if __name__ == "__main__":
    input_avia_path = 'data/raw/aviation.csv'
    input_auto_path = 'data/raw/road_transport.csv'
    output_processed_data_path = 'data/processed/processed_data.csv'
    output_embeddings_data_path = 'data/processed/data_with_embeddings.pkl'
    
    run_preprocessing_pipeline(input_avia_path, input_auto_path, output_processed_data_path, output_embeddings_data_path)
