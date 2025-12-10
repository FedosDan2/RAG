import os
import json
import hashlib
from typing import Optional, Tuple
import faiss
import numpy as np

PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "dataset/rag_history_cache")


class RAGHistoryCache:
    """Кэш истории вопросов и ответов с поддержкой семантического поиска на основе FAISS.

    Этот класс позволяет сохранять пары (вопрос, ответ), строить FAISS-индекс по эмбеддингам
    вопросов и находить ранее данные ответы на семантически похожие запросы.
    Используется для ускорения ответов в RAG-системах за счёт кэширования.

    Атрибуты:
        embedding_model: Модель для генерации эмбеддингов (должна иметь метод encode).
        cache_dir (str): Директория для хранения кэшированных файлов (индекса и данных).
        distance_threshold (float): Порог расстояния L2: запрос считается похожим,
            если расстояние до ближайшего вектора меньше этого значения.
        top_k (int): Количество соседей, возвращаемых при поиске (обычно 1).
        questions (list[str]): Список сохранённых вопросов.
        answers (list[str]): Список сохранённых ответов (соответствует questions по индексу).
        index (faiss.Index or None): FAISS-индекс для поиска ближайших соседей.
    """

    def __init__(self, embedding_model, cache_dir: str = DATA_DIR, distance_threshold: float = 0.2, top_k: int = 1):
        """Инициализирует кэш истории вопросов и ответов.

        Создаёт директорию кэша при необходимости и пытается загрузить существующие данные.
        Если файлы повреждены или отсутствуют — инициализирует пустой кэш.

        Args:
            embedding_model: Модель эмбеддингов с методом encode (например, SentenceTransformer).
            cache_dir (str, optional): Путь к директории для хранения кэша. По умолчанию — DATA_DIR.
            distance_threshold (float, optional): Максимальное L2-расстояние для признания
                запроса «похожим». Чем меньше значение — тем строже совпадение. По умолчанию — 0.2.
            top_k (int, optional): Количество возвращаемых соседей при поиске. Обычно используется 1.
                По умолчанию — 1.
        """
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_history_index()


    def _init_history_index(self):
        """Загружает FAISS-индекс и данные истории из диска или инициализирует пустой кэш.

        Пытается прочитать файлы `history.faiss` и `history.json`. При ошибке или отсутствии
        файлов сбрасывает состояние до пустого.
        """
        self.questions = []     
        self.answers = []       
        self.index = None

        index_path = os.path.join(self.cache_dir, "history.faiss")
        data_path = os.path.join(self.cache_dir, "history.json")

        if os.path.exists(index_path) and os.path.exists(data_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.questions = data["questions"]
                    self.answers = data["answers"]
                print(f"✅ Загружено {len(self.questions)} записей истории.")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки истории: {e}")
                self._reset_index()
        else:
            self._reset_index()


    def _reset_index(self):
        """Сбрасывает внутреннее состояние кэша до пустого."""
        self.questions = []
        self.answers = []
        self.index = None


    def _save_history(self):
        """Сохраняет текущее состояние кэша на диск.

        Записывает FAISS-индекс в бинарный файл и данные (вопросы/ответы) в JSON.
        Если индекс не инициализирован — ничего не сохраняет.
        """
        if self.index is None:
            return

        index_path = os.path.join(self.cache_dir, "history.faiss")
        data_path = os.path.join(self.cache_dir, "history.json")

        faiss.write_index(self.index, index_path)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump({
                "questions": self.questions,
                "answers": self.answers
            }, f, ensure_ascii=False, indent=2)


    def find_similar_answer(self, query: str) -> Optional[str]:
        """Ищет в истории вопрос, семантически похожий на заданный запрос.

        Генерирует эмбеддинг запроса, выполняет поиск в FAISS-индексе и возвращает
        сохранённый ответ, если найден подходящий (расстояние < порога).

        Args:
            query (str): Текст нового вопроса пользователя.

        Returns:
            Optional[str]: Сохранённый ответ, если найден похожий вопрос.
            Возвращает None, если история пуста, индекс не загружен или похожих вопросов нет.
        """
        if self.index is None or len(self.questions) == 0:
            return None

        # Генерация эмбеддинга
        emb = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')

        # Поиск
        distances, indices = self.index.search(emb, self.top_k)
        dist = distances[0][0]
        idx = indices[0][0]

        if dist < self.distance_threshold and 0 <= idx < len(self.answers):
            print("     Похожий запрос найден")
            return self.answers[idx]
        return None


    def add_qa_pair(self, question: str, answer: str):
        """Добавляет новую пару (вопрос, ответ) в историю и обновляет индекс.

        Игнорирует пустые или содержащие только пробелы строки.
        После добавления сохраняет обновлённые данные на диск.

        Args:
            question (str): Вопрос от пользователя.
            answer (str): Ответ, сгенерированный системой или полученный извне.
        """
        if not question.strip() or not answer.strip():
            return

        self.questions.append(question)
        self.answers.append(answer)

        # Генерация эмбеддинга вопроса
        emb = self.embedding_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype('float32')

        # Обновление FAISS-индекса
        if self.index is None:
            dim = emb.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb)

        # Сохранение на диск
        self._save_history()