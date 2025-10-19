from sentence_transformers import SentenceTransformer, util
import os

class SimpleRAGAgent:
    def __init__(self, kb_folder='data/knowledge_base/', embedding_model='paraphrase-multilingual-MiniLM-L12-v2'):
        self.kb_folder = kb_folder
        self.model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.load_documents()
    
    def load_documents(self):
        # Проверка существования папки
        if not os.path.exists(self.kb_folder):
            raise FileNotFoundError(f"Папка {self.kb_folder} не существует")
        
        # Поиск txt файлов
        files = [f for f in os.listdir(self.kb_folder) if f.endswith('.txt')]
        
        if not files:
            raise ValueError(f"В папке {self.kb_folder} нет .txt файлов")
        
        self.documents = []
        for fname in files:
            try:
                with open(os.path.join(self.kb_folder, fname), 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.documents.append(content)
            except Exception as e:
                print(f"Ошибка при чтении файла {fname}: {e}")
        
        if not self.documents:
            raise ValueError("Все документы в базе знаний пусты")
        
        # Создание эмбеддингов
        self.embeddings = self.model.encode(self.documents, convert_to_tensor=True)
    
    def retrieve(self, query, top_k=3):
        if not self.documents:
            return ["База знаний пуста"]
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        return [self.documents[hit['corpus_id']] for hit in hits]
