import os
import json
import hashlib
import faiss
import numpy as np

PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "dataset/rag_lib_official")
FILE_PATH = os.path.join(PROJECT_DIR, "data/dataset.json")


class RAGOfficial:
    """Система извлечения и генерации (RAG) на основе медицинской базы знаний.

    Эта система загружает структурированную медицинскую базу знаний из JSON-файла,
    строит FAISS-индекс по текстовым представлениям записей с использованием
    заданной модели эмбеддингов, кэширует индекс на диске и позволяет извлекать
    релевантные записи по текстовому запросу.

    Атрибуты:
        embedding_model: Модель для генерации эмбеддингов текста (должна иметь метод encode).
        cache_dir (str): Путь к директории для кэширования FAISS-индекса и вспомогательных файлов.
        distance_threshold (float): Максимальное допустимое расстояние L2 для считывания записи релевантной.
        top_k (int): Максимальное количество возвращаемых релевантных записей по умолчанию.
        knowledge_base (list): Загруженная из файла JSON база знаний в виде списка словарей.
        content_hash (str): MD5-хэш содержимого базы знаний для контроля целостности кэша.
        index (faiss.Index): FAISS-индекс для быстрого поиска ближайших соседей.
        kb_texts (list[str]): Текстовые представления записей, использованные для построения индекса.
    """

    def __init__(self, embedding_model, cache_dir: str = DATA_DIR, distance_threshold: float = 1.1, top_k: int = 3):
        """Инициализирует RAG-систему.

        Загружает базу знаний, вычисляет её хэш, строит или загружает кэшированный FAISS-индекс,
        и подготавливает систему к извлечению релевантной информации.

        Args:
            embedding_model: Модель эмбеддингов с методом encode (например, SentenceTransformer).
            cache_dir (str, optional): Директория для хранения кэша индекса. По умолчанию — DATA_DIR.
            distance_threshold (float, optional): Порог расстояния L2 для фильтрации результатов.
                Записи с расстоянием больше этого значения игнорируются. По умолчанию — 1.1.
            top_k (int, optional): Количество возвращаемых релевантных записей при поиске.
                Может быть переопределено в методе _retrieve_relevant_facts. По умолчанию — 3.
        """
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.distance_threshold = distance_threshold
        self.top_k = top_k

        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        print(f"📚 Загружено {len(self.knowledge_base)} записей")
        
        # Генерация "хэша содержимого" — для простоты хэшируем всё как строку
        content_str = json.dumps(self.knowledge_base, sort_keys=True, ensure_ascii=False)
        self.content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
        
        # Создание/загрузка индекса
        self._load_or_build_index()
        print("✅ RAG-система готова к работе!")


    def _get_cache_paths(self):
        """Возвращает пути к файлам кэша индекса.

        Создаёт директорию кэша, если она не существует, и возвращает кортеж из трёх путей:
        - FAISS-индекс (.faiss)
        - Тексты, соответствующие векторам (.json)
        - Хэш содержимого базы знаний (.hash)

        Returns:
            tuple[str, str, str]: Пути к файлам индекса, текстов и хэша соответственно.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        base_name = "kb"
        return (
            os.path.join(self.cache_dir, f"{base_name}.faiss"),
            os.path.join(self.cache_dir, f"{base_name}_texts.json"),
            os.path.join(self.cache_dir, f"{base_name}.hash")
        )


    def _load_or_build_index(self):
        """Загружает FAISS-индекс из кэша или строит его заново.

        Проверяет наличие кэшированных файлов и их соответствие текущему содержимому базы знаний
        по MD5-хэшу. Если кэш устарел или отсутствует — перестраивает индекс.
        """
        index_path, texts_path, hash_path = self._get_cache_paths()
        
        # Проверка кэша
        if os.path.exists(index_path) and os.path.exists(texts_path) and os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                cached_hash = f.read().strip()
            if cached_hash == self.content_hash:
                self.index = faiss.read_index(index_path)
                with open(texts_path, "r", encoding="utf-8") as f:
                    self.kb_texts = json.load(f)
                return
        
        # Строим индекс с нуля
        self._build_faiss_index()
        
        # Сохраняем
        faiss.write_index(self.index, index_path)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(self.kb_texts, f, ensure_ascii=False)
        with open(hash_path, "w") as f:
            f.write(self.content_hash)


    def _build_faiss_index(self):
        """Строит FAISS-индекс на основе текстовых представлений записей базы знаний.

        Для каждой записи объединяет поля 'diagnosis.objectOfRec', 'complaint' и 'keywords'
        в единый текстовый фрагмент. Затем генерирует нормализованные эмбеддинги и добавляет их
        в FAISS-индекс типа IndexFlatL2.
        """
        self.kb_texts = []
        for item in self.knowledge_base:
            diagnosis = item.get("diagnosis", {}).get("objectOfRec", "")
            complaint = item.get("complaint", "")
            keywords = item.get("keywords", "")
            text = f"{diagnosis}. {complaint}. {keywords}"
            self.kb_texts.append(text)

        print("  → Генерация эмбеддингов...")
        embeddings = self.embedding_model.encode(
            self.kb_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"  → Индекс создан. Векторов: {self.index.ntotal}")


    def _retrieve_relevant_facts(self, query: str, top_k: int = 3) -> list:
        """Извлекает релевантные записи из базы знаний по текстовому запросу.

        Генерирует эмбеддинг запроса, выполняет поиск k ближайших соседей в FAISS-индексе
        и фильтрует результаты по порогу расстояния. Возвращает список записей с дополнительным
        полем 'relevance_score'.

        Args:
            query (str): Текстовый запрос пользователя.
            top_k (int, optional): Максимальное количество возвращаемых записей. По умолчанию — 3.

        Returns:
            list[dict]: Список релевантных записей из knowledge_base. Каждая запись дополнена
            ключом 'relevance_score' (float, в диапазоне (0, 1]), рассчитанным как 1/(1 + L2-расстояние).

        Raises:
            RuntimeError: Если FAISS-индекс не был загружен или построен.
        """
        if self.index is None:
            raise RuntimeError("Индекс не загружен.")
        
        query_emb = self.embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        query_emb = np.expand_dims(query_emb, axis=0)

        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist > self.distance_threshold:
                continue
            if 0 <= idx < len(self.knowledge_base):
                result = self.knowledge_base[idx].copy()
                result["relevance_score"] = float(1.0 / (1.0 + dist))
                results.append(result)
        return results