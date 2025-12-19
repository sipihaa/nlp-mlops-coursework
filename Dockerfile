FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi uvicorn pandas nltk tqdm pymorphy3 sentence_transformers tritonclient geventhttpclient

COPY models/rubert-tiny2 /app/rubert-tiny2

# Говорим коду искать модель здесь
ENV SENTENCE_TRANSFORMERS_HOME=/app

COPY app/ app/
COPY src/ src/

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
