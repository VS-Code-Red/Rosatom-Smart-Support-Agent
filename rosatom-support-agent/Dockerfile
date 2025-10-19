# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Создаем необходимые директории
RUN mkdir -p models data/knowledge_base data/rules data/processed

# Обучаем модель при сборке (опционально)
# RUN python train_model.py

# Открываем порт для Streamlit
EXPOSE 8501

# Проверка здоровья контейнера
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запускаем приложение
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
