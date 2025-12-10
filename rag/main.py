from sentence_transformers import SentenceTransformer
from rag_offical_lib import RAGOfficial
from rag_history_cash_lib import RAGHistoryCache
import requests


class MainAdvise:
    """Основной советчик-адаптер, объединяющий RAG-базу знаний и кэш истории запросов.

    Использует две подсистемы:
      - `RAGOfficial`: для извлечения медицинских рекомендаций из структурированной базы знаний.
      - `RAGHistoryCache`: для кэширования и повторного использования ранее сгенерированных ответов
        на семантически похожие запросы.

    Генерирует промпт на русском языке, отправляет его в локальную модель через Ollama API
    и сохраняет новый ответ в историю для будущего использования.

    Атрибуты:
        rag_official (RAGOfficial): Экземпляр RAG-системы для официальной медицинской базы.
        rag_history (RAGHistoryCache): Экземпляр кэша истории вопросов и ответов.
        embedding_model (SentenceTransformer): Модель эмбеддингов, используемая обеими подсистемами.
    """

    def __init__(self, embedding_model):
        """Инициализирует советчика с заданной моделью эмбеддингов.

        Args:
            embedding_model (SentenceTransformer): Предобученная модель для генерации текстовых эмбеддингов.
                Рекомендуется использовать мультилингвальную модель с поддержкой русского языка.
        """
        # Основная RAG-система
        self.rag_official = RAGOfficial(embedding_model=embedding_model)
        # Кэш истории (передаём ту же модель)
        self.rag_history = RAGHistoryCache(embedding_model=embedding_model)
        self.embedding_model = embedding_model

    def _build_prompt(self, question: str, retrieved_facts: list) -> str:
        """Формирует промпт для генеративной модели на основе запроса и извлечённых фактов.

        Если релевантные факты не найдены, возвращает шаблон с рекомендацией к терапевту.
        В противном случае включает в промпт диагнозы, анализы, врачей и оценку релевантности.

        Args:
            question (str): Исходный запрос пациента (жалоба).
            retrieved_facts (list[dict]): Список релевантных записей из RAG-базы,
                каждая из которых содержит поля: 'diagnosis', 'recommendations',
                'required_tests', 'required_doctors', 'relevance_score'.

        Returns:
            str: Сформированный промпт на русском языке, готовый к передаче в LLM.
        """
        if not retrieved_facts:
            return f"""Отвечай на РУССКОМ ЯЗЫКЕ. Ты — врач.

Пациент жалуется: "{question}".
По вашему запросу ничего не найдено. Рекомендуется обратиться к терапевту для первичного осмотра."""

        advice_items = []
        for fact in retrieved_facts:
            diagnosis = fact["diagnosis"]["objectOfRec"]
            recommendations = ", ".join(fact.get("recommendations", ["не указаны"]))
            tests = ", ".join(fact.get("required_tests", ["не указаны"]))
            doctors = ", ".join(fact.get("required_doctors", ["терапевт"]))
            score = fact.get("relevance_score")
            score_str = f"{score:.2%}" if score is not None else "не указана"
            advice_items.append(
                f"Диагноз: {diagnosis} с вероятностью {score_str}\n"
                f"Рекомендации: {recommendations}\n"
                f"Анализы: {tests}\n"
                f"Врачи: {doctors}"
            )
        joined_advice = "\n\n".join(advice_items)

        prompt = f"""Отвечай на РУССКОМ ЯЗЫКЕ. Ты — дипломированный врач.

Пациент жалуется: "{question}".

На основе релевантных данных рекомендуется:

{joined_advice}

Сформулируй ответ в виде краткого, структурированного совета:
- Перечисли предположительные заболевания и их вероятность
- Перечисли анализы, которые следует сдать больному,
- Перечисли врачей, к которым нужно обратиться больному,
- Дай рекомендации по дальнейшим действиям.
Не используй маркированные списки — оформи как связный текст с эмодзи (🩺, 📋, 👨‍⚕️ и т.д.)."""
        
        return prompt

    def generate_advice(self, symptoms: str, model: str = "llama3:8b", ollama_url: str = "http://localhost:11434/api/generate") -> str:
        """Генерирует медицинский совет по жалобе пациента, используя кэш и RAG + LLM.

        Сначала проверяет, не был ли уже обработан похожий запрос (через `RAGHistoryCache`).
        Если нет — извлекает релевантные медицинские факты, формирует промпт и отправляет его
        в локальную LLM через Ollama API. Полученный ответ сохраняется в историю.

        Args:
            symptoms (str): Описание симптомов или жалоба пациента.
            model (str, optional): Имя модели в Ollama. По умолчанию — "llama3:8b".
            ollama_url (str, optional): URL Ollama API endpoint. По умолчанию — локальный.

        Returns:
            str: Сгенерированный медицинский совет в виде читаемого текста на русском языке.

        Raises:
            RuntimeError: В случае ошибки HTTP-запроса к Ollama (таймаут, недоступность и т.п.).
        """
        # 1. Проверка истории
        cached = self.rag_history.find_similar_answer(symptoms)
        if cached:
            print("🔁 Используем кэшированный ответ из истории.")
            return cached

        # 2. Поиск в основной базе
        retrieved = self.rag_official._retrieve_relevant_facts(symptoms.strip(), top_k=3)
        prompt = self._build_prompt(symptoms.strip(), retrieved)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5, "num_ctx": 2048}
        }

        try:
            response = requests.post(ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            advice = response.json().get("response", "").strip()

            # 3. Сохраняем в историю
            self.rag_history.add_qa_pair(symptoms.strip(), advice)
            return advice

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ошибка при обращении к Ollama: {e}")


def main():
    """Основная точка входа для генерации медицинского совета по примеру.

    Инициализирует эмбеддинг-модель, создаёт экземпляр советчика и генерирует ответ
    на пример запроса о температуре и кашле. Результат сохраняется в файл `out.txt`.
    """
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')
    rag = MainAdvise(embedding_model)
    result = rag.generate_advice(symptoms="У меня второй день держится высокая температура и кашель")

    with open("out.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("✅ Результат сохранён в out.txt")


if __name__ == "__main__":
    main()