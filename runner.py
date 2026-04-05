import os
import sys
import torch
from sentence_transformers import SentenceTransformer

# Импортируем основной класс консультации
from rag.main_advise import MainAdvise

def check_environment() -> bool:
    """Проверяет наличие всех необходимых файлов и директорий."""
    required = [
        "data/diseases.json",
        "data/questions.json",
        "dataset/rag_lib_official"   # папка для кэша FAISS (создастся автоматически)
    ]
    missing = []
    for path in required:
        if not os.path.exists(path) and not path.endswith("rag_lib_official"):
            missing.append(path)
    if missing:
        print("❌ Ошибка: отсутствуют следующие файлы/директории:")
        for m in missing:
            print(f"   - {m}")
        print("\nУбедитесь, что:")
        print("   - папка 'data' содержит diseases.json и questions.json")
        print("   - папка 'dataset/rag_lib_official' существует или будет создана автоматически")
        return False
    return True

def main():
    print("🩺 Запуск медицинской диагностической системы...")
    if not check_environment():
        sys.exit(1)

    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Устройство: {device}")
    print("📥 Загрузка модели эмбеддингов (multilingual-e5-large)...")
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

    # Создаём экземпляр консультанта
    advisor = MainAdvise(embedding_model, device)

    # Запускаем интерактивный режим
    advisor.interactive_advice(certainty_threshold=0.7, max_questions=5)

if __name__ == "__main__":
    main()