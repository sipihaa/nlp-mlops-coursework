FROM python:3.12-slim

WORKDIR /app

# 3. Установка системных зависимостей (если нужны, например для сборки wheel)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# 4. Установка зависимостей Python
# Вариант с Poetry (более правильный для dev, но export надежнее для prod)
RUN pip install poetry

# Копируем файлы описания зависимостей
COPY pyproject.toml poetry.lock ./

# Экспортируем в requirements.txt и ставим через pip (чтобы не тащить poetry в рантайм, если хотим сэкономить место, 
# но здесь мы оставим poetry для простоты, или используем pip install)
# ЛУЧШАЯ ПРАКТИКА:
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# 5. Пре-загрузка BERT (чтобы не качать при старте контейнера каждый раз)
# Это критично для скорости старта
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cointegrated/rubert-tiny2')"

# 6. Копируем код проекта
# Сначала копируем папки, чтобы слой с зависимостями закешировался выше
COPY app/ app/
COPY src/ src/

# Настраиваем PYTHONPATH, чтобы python видел src и app
ENV PYTHONPATH=/app

# 7. Запуск
# Используем exec форму (массив строк), чтобы сигналы (Ctrl+C) проходили корректно
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
