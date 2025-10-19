import streamlit as st
from agents.lightweight_classifier import LightweightClassifier
from agents.simple_rag_agent import SimpleRAGAgent
from agents.escalation_agent import EscalationAgent
import os
import pandas as pd
from datetime import datetime

def _log_all_queries(query, category, confidence):
    """Логирует все запросы для построения графика частоты"""
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/all_queries.csv"
    log_df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "predicted_category": category,
        "confidence": confidence
    }])
    log_df.to_csv(
        log_path,
        mode="a",
        header=not os.path.exists(log_path),
        index=False,
        encoding="utf-8"
    )

def _log_feedback(query, category, confidence, feedback_value):
    """Логирует отзыв пользователя и сохраняет запрос для self-learning"""
    os.makedirs("logs", exist_ok=True)
    
    # Логируем отзыв
    feedback_path = "logs/feedback.csv"
    feedback_df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "predicted_category": category,
        "confidence": confidence,
        "feedback": feedback_value,
        "useful": "Да" if feedback_value == 1 else "Нет"
    }])
    feedback_df.to_csv(
        feedback_path,
        mode="a",
        header=not os.path.exists(feedback_path),
        index=False,
        encoding="utf-8"
    )

    # Если лайк — сохраняем для self-learning
    if feedback_value == 1:
        learned_path = "logs/learned_queries.csv"
        learned_df = pd.DataFrame([{
            "text": query,
            "category": category
        }])
        learned_df.to_csv(
            learned_path,
            mode="a",
            header=not os.path.exists(learned_path),
            index=False,
            encoding="utf-8"
        )

st.set_page_config(page_title="Мультиагентный саппорт", page_icon="🤖")

# Загрузка модели и агентов
@st.cache_resource
def load_models():
    try:
        clf = LightweightClassifier()
        clf.load("models/lightweight_clf.pkl")
        rag_agent = SimpleRAGAgent()
        esc_agent = EscalationAgent(threshold=0.6)
        return clf, rag_agent, esc_agent
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки: {e}")
        st.info("Убедитесь, что модель обучена. Запустите: python train_model.py")
        st.stop()
    except Exception as e:
        st.error(f"Неожиданная ошибка: {e}")
        st.stop()

clf, rag_agent, esc_agent = load_models()

st.title("Мультиагентный саппорт")
st.markdown("Система автоматической классификации и обработки обращений в техподдержку")

query = st.text_area("Опишите вашу проблему:", height=150, placeholder="Например: Не могу войти в систему...")

# === Сохраняем состояние между перезапусками ===
if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = None
    st.session_state.results = None
    st.session_state.rag_answers = None
    st.session_state.feedback_logged = False

# Кнопка отправки
if st.button("Отправить", type="primary"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите запрос")
    else:
        with st.spinner("Обрабатываю запрос..."):
            try:
                results = clf.predict([query])
                pred, conf = results[0]
                
                # ЛОГИРУЕМ ВСЕ ЗАПРОСЫ (для графика) === 
                _log_all_queries(query, pred, conf)
                
                # Сохраняем в состояние
                st.session_state.submitted_query = query
                st.session_state.results = (pred, conf)
                st.session_state.feedback_logged = False
                
                # Проверяем эскалацию
                if esc_agent.decide(conf):
                    st.session_state.rag_answers = None
                else:
                    answers = rag_agent.retrieve(query)
                    st.session_state.rag_answers = answers

            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {e}")

# Отображение результата (если есть сохранённый запрос) 
if st.session_state.submitted_query is not None:
    pred, conf = st.session_state.results
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Категория", pred)
    with col2:
        st.metric("Уверенность", f"{conf:.2%}")
    
    if st.session_state.rag_answers is None:
        st.error("Запрос передан оператору")
        st.info("Уровень уверенности модели недостаточен для автоматического ответа. "
                "Ваш запрос будет обработан специалистом технической поддержки.")
    else:
        st.success("Автоматический ответ (RAG):")
        for i, answer in enumerate(st.session_state.rag_answers, 1):
            with st.expander(f"Релевантный ответ #{i}", expanded=(i == 1)):
                st.write(answer)

        # Обратная связь 
        if not st.session_state.feedback_logged:
            st.markdown("### Был ли ответ полезен?")
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                if st.button("👍 Да", key="feedback_positive"):
                    _log_feedback(st.session_state.submitted_query, pred, conf, 1)
                    st.session_state.feedback_logged = True
            with col_fb2:
                if st.button("👎 Нет", key="feedback_negative"):
                    _log_feedback(st.session_state.submitted_query, pred, conf, 0)
                    st.session_state.feedback_logged = True

        if st.session_state.feedback_logged:
            st.success("Спасибо за обратную связь! Это поможет улучшить систему.")

    
    # ГРАФИК ЧАСТОТЫ ПРОБЛЕМ ПО КАТЕГОРИЯМ 
st.markdown("---")
st.subheader("📊 Частота обращений по категориям")

# Считаем количество обращений по каждой категории из feedback.csv
try:
    feedback_df = pd.read_csv("logs/feedback.csv", encoding="utf-8")
    category_counts = feedback_df["predicted_category"].value_counts().reset_index()
    category_counts.columns = ["Категория", "Количество"]

    # Визуализация
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=category_counts, x="Категория", y="Количество", palette="viridis", ax=ax)
    ax.set_title("Распределение обращений по категориям")
    ax.set_xlabel("Категория")
    ax.set_ylabel("Количество")
    plt.xticks(rotation=45)
    st.pyplot(fig)

except FileNotFoundError:
    st.info("Пока нет данных для отображения графика. Начните обрабатывать запросы!")
