#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обучения модели классификатора
"""

import os
import sys
import pandas as pd

# ВАЖНО: Импортируем из пакета agents
import agents
from agents.lightweight_classifier import LightweightClassifier


def train_model():
    print("=" * 60)
    print("Обучение модели классификатора")
    print("=" * 60)

    # Проверка наличия обучающих данных
    data_path = 'data/processed/training_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Ошибка: Файл {data_path} не найден")
        sys.exit(1)

    print(f"✓ Загрузка данных из {data_path}...")
    df = pd.read_csv(data_path)

    # Проверяем наличие нужных столбцов
    if 'text_clean' in df.columns:
        text_column = 'text_clean'
    elif 'text' in df.columns:
        text_column = 'text'
    else:
        print(f"❌ Ошибка: В файле нет столбца 'text' или 'text_clean'")
        print(f"Найденные столбцы: {df.columns.tolist()}")
        sys.exit(1)

    if 'category' not in df.columns:
        print(f"❌ Ошибка: В файле нет столбца 'category'")
        sys.exit(1)

    print(f"✓ Загружено записей: {len(df)}")
    print(f"✓ Используется столбец: {text_column}")
    print(f"✓ Категории: {df['category'].unique().tolist()}")

    texts = df[text_column].tolist()
    labels = df['category'].tolist()

    print("\n" + "=" * 60)
    print("Обучение модели...")
    print("=" * 60)

    clf = LightweightClassifier()
    clf.train(texts, labels)

    print("✓ Модель успешно обучена!")

    # Создаем папку models если её нет
    os.makedirs('models', exist_ok=True)

    # Сохраняем модель
    model_path = 'models/lightweight_clf.pkl'
    clf.save(model_path)

    print(f"✓ Модель сохранена в {model_path}")
    print("\n" + "=" * 60)
    print("✅ Обучение завершено успешно!")
    print("=" * 60)
    print("\nТеперь вы можете запустить приложение:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    train_model()
