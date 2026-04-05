import numpy as np
from typing import Any
from rag.rag_offical_lib import RAGOfficial
from rag.detect_objects.entry_point import get_features
from rag.diagnostic import DiagnosticSession

class MainAdvise:
    def __init__(self, embedding_model, device):
        self.rag = RAGOfficial(embedding_model=embedding_model)
        self.device = device
    
    def _ask_question(self, question: dict) -> Any:
        q_type = question["type"]
        text = question["text"]
        options = question.get("options", [])
        
        if q_type == "yes_no":
            print(f"\n❓ {text} (Да/Нет)")
            while True:
                ans = input("➡️ ").strip().lower()
                if ans in ["да", "yes", "y", "д"]:
                    return "Да"
                elif ans in ["нет", "no", "n", "н"]:
                    return "Нет"
                print("Пожалуйста, ответьте 'Да' или 'Нет'.")
        
        elif q_type == "single_choice":
            print(f"\n❓ {text}")
            for i, opt in enumerate(options, 1):
                print(f"   {i}. {opt}")
            while True:
                try:
                    choice = input("➡️ Введите номер варианта: ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                    print(f"Введите число от 1 до {len(options)}.")
                except ValueError:
                    print("Введите целое число.")
        
        elif q_type == "multi_choice":
            print(f"\n❓ {text} (можно выбрать несколько, через запятую)")
            for i, opt in enumerate(options, 1):
                print(f"   {i}. {opt}")
            print("Пример ввода: 1,3  или 2")
            while True:
                ans = input("➡️ Введите номера через запятую: ").strip()
                if not ans:
                    return []
                parts = ans.replace(",", " ").split()
                selected_nums = []
                for p in parts:
                    try:
                        selected_nums.append(int(p))
                    except:
                        pass
                selected = [options[i-1] for i in selected_nums if 1 <= i <= len(options)]
                return selected
        
        elif q_type == "scale":
            scale_range = question.get("scale_range", [1, 5])
            min_val, max_val = scale_range[0], scale_range[-1]
            print(f"\n❓ {text} (от {min_val} до {max_val})")
            while True:
                ans = input("➡️ ").strip()
                try:
                    val = float(ans)
                    if min_val <= val <= max_val:
                        return str(val)
                    print(f"Введите число от {min_val} до {max_val}.")
                except ValueError:
                    print("Введите число.")
        
        else:
            return input(f"\n❓ {text}\n➡️ ").strip()
    
    def interactive_advice(self, certainty_threshold=0.7, max_questions=10):
        print("\n🩺 Здравствуйте! Я ваш медицинский консультант.")
        print("Пожалуйста, как можно подробнее опишите проблему, которая вас беспокоит.")
        print("Например: 'У меня второй день держится высокая температура и кашель'\n")
        
        complaint = input("Ваши жалобы: ").strip()
        if not complaint:
            print("Вы не ввели жалобы. Завершение работы.")
            return None
        
        # Поиск в RAG
        retrieved = self.rag._retrieve_relevant_facts(complaint, complaint, top_k=5)
        if not retrieved:
            print("\nК сожалению, я не смог найти подходящие диагнозы в моей базе.")
            print("Рекомендую обратиться к врачу очно для детального обследования.")
            return None
        
        # Объединяем диагнозы по id (а не по имени)
        diag_map = {}
        for fact in retrieved:
            diag_id = fact.get("id")          # предполагаем, что в diseases.json есть поле "id"
            diag_name = fact["name"]
            if diag_id not in diag_map:
                diag_map[diag_id] = {
                    "id": diag_id,
                    "name": diag_name,
                    "required_tests": set(fact.get("required_tests", [])),
                    "required_doctors": set(fact.get("required_doctors", [])),
                    "questions": [],
                    "max_score": fact.get("relevance_score", 0.0)
                }
            else:
                diag_map[diag_id]["required_tests"].update(fact.get("required_tests", []))
                diag_map[diag_id]["required_doctors"].update(fact.get("required_doctors", []))
                diag_map[diag_id]["max_score"] = max(diag_map[diag_id]["max_score"], fact.get("relevance_score", 0.0))
            # Собираем вопросы
            for q in fact.get("questions", []):
                if q["id"] not in [q2["id"] for q2 in diag_map[diag_id]["questions"]]:
                    diag_map[diag_id]["questions"].append(q)
        
        diagnoses_objects = []
        initial_scores = []
        questions_db = {}
        for diag_id, data in diag_map.items():
            diagnoses_objects.append({
                "id": diag_id,
                "name": data["name"],
                "required_tests": list(data["required_tests"]),
                "required_doctors": list(data["required_doctors"])
            })
            initial_scores.append(data["max_score"])
            for q in data["questions"]:
                if q["id"] not in questions_db:
                    questions_db[q["id"]] = q
        
        # Нормализуем начальные вероятности
        total_score = sum(initial_scores)
        if total_score == 0:
            probs = [1.0/len(diagnoses_objects)] * len(diagnoses_objects)
        else:
            probs = [s/total_score for s in initial_scores]
        
        session = DiagnosticSession(diagnoses_objects, questions_db, probs)
        
        print("\n📊 Предварительные вероятности диагнозов (на основе ваших жалоб):")
        for i, diag in enumerate(diagnoses_objects):
            print(f"   {i+1}. {diag['name']}: {session.probs[i]:.1%}")
        
        print("\n🔍 Для уточнения диагноза я буду задавать вам вопросы.\n")
        step = 0
        while step < max_questions:
            max_prob = np.max(session.probs)
            if max_prob >= certainty_threshold:
                best_idx = np.argmax(session.probs)
                print(f"\n✅ Достигнута уверенность: {session.diagnoses[best_idx]['name']} с вероятностью {max_prob:.1%}.")
                break
            
            next_qid = session.select_next_question()
            if next_qid is None:
                print("\n⚠️ Больше нет подходящих вопросов.")
                break
            
            question = questions_db[next_qid]
            answer = self._ask_question(question)
            session.update(next_qid, answer)
            
            print("\n📊 Текущие вероятности после ответа:")
            sorted_indices = np.argsort(session.probs)[::-1]
            for idx in sorted_indices[:3]:
                print(f"   {session.diagnoses[idx]['name']}: {session.probs[idx]:.1%}")
            step += 1
        
        best_idx = np.argmax(session.probs)
        best_diag = session.diagnoses[best_idx]
        print(f"\n🏥 Наиболее вероятный диагноз: {best_diag['name']} (вероятность {session.probs[best_idx]:.1%})")
        print(f"📋 Рекомендованные анализы: {', '.join(best_diag.get('required_tests', []))}")
        print(f"👨‍⚕️ Врачи: {', '.join(best_diag.get('required_doctors', []))}")
        
        self._save_consultation(complaint, complaint, session, questions_db)
    
    def _save_consultation(self, complaint, symptoms, session, questions_db):
        with open("consultation_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Жалобы: {complaint}\n")
            f.write(f"Извлечённые симптомы: {', '.join(symptoms)}\n\n")
            f.write("История вопросов и ответов:\n")
            for qid, ans in session.history:
                q_text = questions_db[qid]['text']
                f.write(f"Вопрос: {q_text}\n")
                f.write(f"Ответ: {ans}\n\n")
            f.write("Финальные вероятности диагнозов:\n")
            for diag, prob in zip(session.diagnoses, session.probs):
                f.write(f"{diag['name']}: {prob:.1%}\n")
        print("\n✅ Результат сохранён в 'consultation_result.txt'.")
