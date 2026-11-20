# Образ бота: Python + библиотеки + код. Секреты (.env) в образ НЕ кладутся.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Сначала библиотеки: этот слой пересобирается только при изменении requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Код бота и документы по умолчанию
COPY bot.py rag.py admin.py qa_test.py ./
COPY docs ./docs

# Запуск не от root. Папка /data хранит данные, которые не должны пропадать при пересборке
RUN useradd -m -u 1000 app && mkdir -p /data && chown -R app:app /app /data
USER app

# Внутри контейнера админка слушает все адреса; наружу порт открывает docker-compose.yml
ENV ADMIN_HOST=0.0.0.0 \
    ADMIN_PORT=8080 \
    QA_FILE=/data/qa.json \
    UNKNOWN_FILE=/data/unknown.json \
    SETTINGS_FILE=/data/settings.json

EXPOSE 8080
CMD ["python", "bot.py"]
