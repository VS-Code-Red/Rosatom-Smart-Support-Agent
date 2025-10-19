import streamlit as st
from agents.lightweight_classifier import LightweightClassifier
from agents.simple_rag_agent import SimpleRAGAgent
from agents.escalation_agent import EscalationAgent
import os

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

if st.button("Отправить", type="primary"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите запрос")
    else:
        with st.spinner("Обрабатываю запрос..."):
            try:
                # Получаем предсказание
                results = clf.predict([query])
                pred, conf = results[0]
                
                # Отображаем результаты классификации
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Категория", pred)
                with col2:
                    st.metric("Уверенность", f"{conf:.2%}")
                
                # Проверяем необходимость эскалации
                if esc_agent.decide(conf):
                    st.error("⚠️ Запрос передан оператору")
                    st.info("Уровень уверенности модели недостаточен для автоматического ответа. "
                           "Ваш запрос будет обработан специалистом технической поддержки.")
                else:
                    st.success("✅ Автоматический ответ (RAG):")
                    answers = rag_agent.retrieve(query)
                    
                    for i, answer in enumerate(answers, 1):
                        with st.expander(f"Релевантный ответ #{i}", expanded=(i==1)):
                            st.write(answer)
            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {e}")

