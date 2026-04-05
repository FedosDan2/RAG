import os
import json
import hashlib
import faiss
import numpy as np

PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "dataset/rag_lib_official")
DISEASES_PATH = os.path.join(PROJECT_DIR, "data/diseases.json")
QUESTIONS_PATH = os.path.join(PROJECT_DIR, "data/questions.json")

class RAGOfficial:
    def __init__(self, embedding_model, cache_dir: str = DATA_DIR, 
                 distance_threshold: float = 1.1, top_k: int = 3,
                 weight_complaint: float = 0.4, weight_keywords: float = 0.6):
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        self.weight_complaint = weight_complaint
        self.weight_keywords = weight_keywords

        # Загружаем заболевания и вопросы
        with open(DISEASES_PATH, 'r', encoding='utf-8') as f:
            self.diseases = json.load(f)
        
        with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            self.questions_db = {q["id"]: q for q in questions_data}
        
        # Для каждого заболевания привязываем вопросы по question_ids
        for disease in self.diseases:
            disease["questions"] = [self.questions_db[qid] for qid in disease.get("question_ids", [])]
        
        print(f"📚 Загружено {len(self.diseases)} заболеваний и {len(self.questions_db)} вопросов")
        
        # Кэширование FAISS индексов
        content_str = json.dumps(self.diseases, sort_keys=True, ensure_ascii=False)
        self.content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
        self._load_or_build_index()
        print("✅ RAG-система готова к работе!")

    def _get_cache_paths(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        return {
            "complaint": {
                "index": os.path.join(self.cache_dir, "kb_complaint.faiss"),
                "texts": os.path.join(self.cache_dir, "kb_complaint_texts.json")
            },
            "keywords": {
                "index": os.path.join(self.cache_dir, "kb_keywords.faiss"),
                "texts": os.path.join(self.cache_dir, "kb_keywords_texts.json")
            },
            "hash": os.path.join(self.cache_dir, "kb.hash")
        }

    def _load_or_build_index(self):
        paths = self._get_cache_paths()
        
        if (os.path.exists(paths["complaint"]["index"]) and 
            os.path.exists(paths["keywords"]["index"]) and
            os.path.exists(paths["hash"])):
            with open(paths["hash"], "r") as f:
                cached_hash = f.read().strip()
            if cached_hash == self.content_hash:
                self.index_complaint = faiss.read_index(paths["complaint"]["index"])
                self.index_keywords = faiss.read_index(paths["keywords"]["index"])
                with open(paths["complaint"]["texts"], "r", encoding="utf-8") as f:
                    self.complaint_texts = json.load(f)
                with open(paths["keywords"]["texts"], "r", encoding="utf-8") as f:
                    self.keywords_texts = json.load(f)
                return
        
        self._build_faiss_index()
        faiss.write_index(self.index_complaint, paths["complaint"]["index"])
        faiss.write_index(self.index_keywords, paths["keywords"]["index"])
        with open(paths["complaint"]["texts"], "w", encoding="utf-8") as f:
            json.dump(self.complaint_texts, f, ensure_ascii=False)
        with open(paths["keywords"]["texts"], "w", encoding="utf-8") as f:
            json.dump(self.keywords_texts, f, ensure_ascii=False)
        with open(paths["hash"], "w") as f:
            f.write(self.content_hash)

    def _build_faiss_index(self):
        self.complaint_texts = []
        self.keywords_texts = []
        for disease in self.diseases:
            self.complaint_texts.append(disease.get("complaint", ""))
            self.keywords_texts.append(disease.get("keywords", ""))
        
        print("  → Генерация эмбеддингов для жалоб...")
        complaint_embs = self.embedding_model.encode(
            self.complaint_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        print("  → Генерация эмбеддингов для ключевых слов...")
        keywords_embs = self.embedding_model.encode(
            self.keywords_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        dim = complaint_embs.shape[1]
        self.index_complaint = faiss.IndexFlatL2(dim)
        self.index_keywords = faiss.IndexFlatL2(dim)
        self.index_complaint.add(complaint_embs)
        self.index_keywords.add(keywords_embs)
        print(f"  → Индексы созданы. Векторов: {self.index_complaint.ntotal}")

    def _retrieve_relevant_facts(self, query: str, symptoms: str, top_k: int = 3) -> list:
        query_emb = self.embedding_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        query_emb = np.expand_dims(query_emb, axis=0)
        symptoms_emb = self.embedding_model.encode(symptoms, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        symptoms_emb = np.expand_dims(symptoms_emb, axis=0)
        
        dist_c, idx_c = self.index_complaint.search(query_emb, top_k)
        dist_k, idx_k = self.index_keywords.search(symptoms_emb, top_k)
        
        candidates = {}
        for dist, idx in zip(dist_c[0], idx_c[0]):
            if dist < self.distance_threshold:
                candidates[idx] = [dist, None] if idx not in candidates else [min(candidates[idx][0], dist), candidates[idx][1]]
        for dist, idx in zip(dist_k[0], idx_k[0]):
            if dist < self.distance_threshold:
                if idx not in candidates:
                    candidates[idx] = [None, dist]
                else:
                    candidates[idx][1] = min(candidates[idx][1], dist) if candidates[idx][1] is not None else dist
        
        scored = []
        for idx, (d_c, d_k) in candidates.items():
            if d_c is not None and d_k is not None:
                combined_dist = self.weight_complaint * d_c + self.weight_keywords * d_k
                score = 1.0 / (1.0 + combined_dist)
            elif d_c is not None:
                score = 1.0 / (1.0 + d_c) * 0.8
            elif d_k is not None:
                score = 1.0 / (1.0 + d_k) * 0.8
            else:
                continue
            scored.append((idx, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:top_k]:
            if 0 <= idx < len(self.diseases):
                result = self.diseases[idx].copy()
                result["relevance_score"] = score
                results.append(result)
        return results