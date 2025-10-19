import json
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class RuleBasedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, rules_path='data/rules/rules.json'):
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Файл правил {rules_path} не найден")
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def apply_rules(text):
            tags = []
            for cat, keywords in self.rules.items():
                for kw in keywords:
                    if kw.lower() in text.lower():
                        tags.append(cat)
                        break  # Оптимизация: выходим после первого совпадения
            return text + ' ' + ' '.join(set(tags))  # Удаляем дубликаты тегов
        return [apply_rules(doc) for doc in X]

class LightweightClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('rules', RuleBasedPreprocessor()),
            ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2))),
            ('clf', MultinomialNB(alpha=0.1))
        ])
    
    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)
    
    def predict(self, texts):
        preds = self.pipeline.predict(texts)
        probs = self.pipeline.predict_proba(texts)
        confidences = probs.max(axis=1)
        return list(zip(preds, confidences))
    
    def save(self, path='models/lightweight_clf.pkl'):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
    
    def load(self, path='models/lightweight_clf.pkl'):
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель {path} не найдена")
        self.pipeline = joblib.load(path)
