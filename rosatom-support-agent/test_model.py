#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для быстрого тестирования модели на примерах
"""

import agents
from agents.lightweight_classifier import LightweightClassifier
from agents.simple_rag_agent import SimpleRAGAgent
from agents.escalation_agent import EscalationAgent

def test_model():
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ МУЛЬТИАГЕНТНОЙ СИСТЕМЫ")
    print("=" * 80)
    
    # Загрузка модели и агентов
    print("\n📥 Загрузка модели и агентов...")
    try:
        clf = LightweightClassifier()
        clf.load("models/lightweight_clf.pkl")
        rag_agent = SimpleRAGAgent()
        esc_agent = EscalationAgent(threshold=0.6)
        print("✅ Все компоненты загружены успешно!\n")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Тестовые примеры
    test_cases = [
        # Категория: errors
        ("Ошибка при входе в систему, не могу авторизоваться", "errors"),
        ("Программа выдает error при попытке сохранения", "errors"),
        
        # Категория: software
        ("Проблемы с 1С после обновления, не запускается", "software"),
        ("SAP не подключается к базе данных", "software"),
        
        # Категория: requests
        ("Прошу оформить заявку на отпуск с 15 по 30 числа", "requests"),
        ("Нужно добавить нового пользователя в систему срочно", "requests"),
        
        # Категория: documents
        ("Документ на НДС формируется с неправильными данными", "documents"),
        ("Проблема с печатью документов из базы", "documents"),
        
        # Категория: access
        ("Не могу войти в систему, пароль не подходит", "access"),
        ("Требуется настроить MFA для учетной записи", "access"),
        
        # Сложные случаи
        ("Не могу оформить заявку на доступ к 1С, выдает ошибку", "errors/software"),
    ]
    
    print("=" * 80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 80)
    
    correct = 0
    total = len(test_cases)
    escalated = 0
    
    for i, (text, expected_category) in enumerate(test_cases, 1):
        print(f"\n🧪 Тест {i}/{total}")
        print(f"   Запрос: {text}")
        print(f"   Ожидаемая категория: {expected_category}")
        print("-" * 80)
        
        # Предсказание
        results = clf.predict([text])
        pred, conf = results[0]
        
        print(f"   📊 Предсказанная категория: {pred}")
        print(f"   📈 Уверенность: {conf:.2%}")
        
        # Проверка правильности
        is_correct = pred in expected_category or expected_category in pred
        if is_correct:
            print(f"   ✅ ПРАВИЛЬНО")
            correct += 1
        else:
            print(f"   ❌ НЕПРАВИЛЬНО")
        
        # Проверка эскалации
        if esc_agent.decide(conf):
            print(f"   ⚠️  ЭСКАЛАЦИЯ: Передано оператору (уверенность < 60%)")
            escalated += 1
        else:
            print(f"   ✅ АВТООТВЕТ: Ищем релевантные решения...")
            try:
                answers = rag_agent.retrieve(text, top_k=1)
                print(f"   💬 Найденный ответ: {answers[0][:100]}...")
            except Exception as e:
                print(f"   ⚠️  Ошибка RAG: {e}")
    
    # Итоговая статистика
    print("\n" + "=" * 80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"✅ Правильных ответов: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"❌ Неправильных ответов: {total-correct}/{total} ({(total-correct)/total*100:.1f}%)")
    print(f"⚠️  Эскалаций: {escalated}/{total} ({escalated/total*100:.1f}%)")
    
    # Оценка качества
    accuracy = correct / total
    print("\n" + "=" * 80)
    print("🎯 ОЦЕНКА КАЧЕСТВА")
    print("=" * 80)
    
    if accuracy >= 0.8:
        print("✅ ОТЛИЧНО! Модель работает очень хорошо")
    elif accuracy >= 0.6:
        print("⚠️  ХОРОШО. Есть пространство для улучшения")
    else:
        print("❌ ТРЕБУЕТСЯ УЛУЧШЕНИЕ. Рекомендуется:")
        print("   - Добавить больше обучающих данных")
        print("   - Обновить правила в rules.json")
        print("   - Переобучить модель")
    
    if escalated / total > 0.3:
        print("\n⚠️  ВНИМАНИЕ: Высокий процент эскалаций")
        print("   Рекомендация: Увеличьте порог уверенности в app.py")
    elif escalated / total < 0.1:
        print("\n⚠️  ВНИМАНИЕ: Низкий процент эскалаций")
        print("   Рекомендация: Уменьшите порог уверенности в app.py")
    else:
        print("\n✅ Процент эскалаций оптимален (10-30%)")
    
    print("\n" + "=" * 80)
    print("Тестирование завершено!")
    print("=" * 80)

if __name__ == "__main__":
    test_model()
