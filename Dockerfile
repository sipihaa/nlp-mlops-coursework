FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi uvicorn pandas nltk tqdm pymorphy3 sentence_transformers emoji tritonclient geventhttpclient streamlit

ENV HF_HUB_DISABLE_PROGRESS_BARS=1

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cointegrated/rubert-tiny2')"

COPY app/ app/
COPY src/ src/

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
