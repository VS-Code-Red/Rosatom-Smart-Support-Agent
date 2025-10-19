# Rosatom Smart Support Agent

Интеллектуальная система автоматической обработки обращений в техподдержку: классификация по доменам (IT, HR, бухгалтерия), поиск ответов в базе знаний и веб-интерфейс на Streamlit.
## Команда
- Шедько Ю.С. — ML-инженер 
- Бендик М.А. — Data Scientist 
- Джафархонов Р.С. — Backend 
## Структура
├── app.py # Streamlit-интерфейс
├── train_model.py # Скрипт обучения классификатора
├── test_model.py # Тесты классификации
├── data/
│ └── processed/
│ └── training_data.csv
├── models/
│ └── lightweight_clf.pkl
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── notebooks/ # Jupyter-ноутбуки с анализом и RAG-демо
## Запуск

### С Docker
docker-compose up --build
Откройте http://localhost:8501

### Без Docker
python -m venv venv
venv\Scripts\activate # Windows
pip install -r requirements.txt
python train_model.py
streamlit run app.py
