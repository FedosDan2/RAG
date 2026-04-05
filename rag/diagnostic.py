import numpy as np
from typing import List, Dict, Optional, Any

class DiagnosticSession:
    """Состояние адаптивного опроса."""
    def __init__(self, diagnoses: List[dict], questions_db: Dict[str, dict], initial_probs: List[float]):
        self.diagnoses = diagnoses          # список объектов заболеваний (с полями name, required_tests, required_doctors)
        self.questions_db = questions_db    # словарь {question_id: question}
        self.probs = np.array(initial_probs, dtype=float)
        self.asked_questions = set()
        self.history = []                   # (question_id, answer)
    
    def entropy(self) -> float:
        return -np.sum(self.probs * np.log2(self.probs + 1e-12))
    
    def update(self, question_id: str, answer: Any):
        q = self.questions_db[question_id]
        q_type = q["type"]
        options = q.get("options", [])
        applicable = q.get("applicable_to", {})
        
        print(f"\n[DEBUG] Обновление по вопросу {question_id}, тип={q_type}, ответ={answer}")
        
        likelihood = []
        for i, diag in enumerate(self.diagnoses):
            diag_name = diag["name"]
            if diag_name in applicable:
                probs_for_diag = applicable[diag_name]
                if q_type == "yes_no":
                    prob = probs_for_diag.get(answer, {}).get("p", 0.5)
                elif q_type == "single_choice":
                    prob = probs_for_diag.get(answer, {}).get("p", 1.0/len(options))
                elif q_type == "multi_choice":
                    prob = 1.0
                    for opt in options:
                        p_present = probs_for_diag.get(opt, {}).get("p_if_present", 0.5)
                        if opt in answer:
                            prob *= p_present
                        else:
                            prob *= (1 - p_present)
                elif q_type == "scale":
                    # Приводим ответ к строке, если нужно
                    ans_key = str(answer)
                    # Пытаемся найти точное совпадение, иначе ищем по интервалам (упрощённо)
                    if ans_key in probs_for_diag:
                        prob = probs_for_diag[ans_key].get("p", 0.0)
                    else:
                        # Для шкалы можно интерполировать, но для простоты возьмём 0
                        prob = 0.0
                else:
                    prob = 0.5
            else:
                # Нейтральное распределение
                if q_type == "yes_no":
                    prob = 0.5
                elif q_type == "single_choice":
                    prob = 1.0 / len(options) if options else 0.5
                elif q_type == "multi_choice":
                    prob = 0.5 ** len(options)
                elif q_type == "scale":
                    prob = 1.0 / len(q.get("scale_range", [1,5]))
                else:
                    prob = 0.5
            likelihood.append(prob)
            print(f"   {diag_name}: P(answer|diag) = {prob:.4f}")
        
        likelihood = np.array(likelihood)
        new_probs = self.probs * likelihood
        total = new_probs.sum()
        if total > 0:
            self.probs = new_probs / total
        else:
            self.probs = np.ones_like(self.probs) / len(self.probs)
        
        self.asked_questions.add(question_id)
        self.history.append((question_id, answer))
        
        print(f"   Нормировочная сумма = {total:.4f}")
        for i, diag in enumerate(self.diagnoses):
            print(f"   {diag['name']}: новая вероятность = {self.probs[i]:.4f}")
    
    def information_gain(self, question_id: str) -> float:
        q = self.questions_db[question_id]
        q_type = q["type"]
        options = q.get("options", [])
        applicable = q.get("applicable_to", {})
        
        if q_type == "yes_no":
            outcomes = ["Да", "Нет"]
        elif q_type == "single_choice":
            outcomes = options
        elif q_type == "multi_choice":
            # Для простоты не используем multi_choice при выборе вопроса
            return 0.0
        elif q_type == "scale":
            # Собираем все возможные значения из applicable_to
            values = set()
            for diag_name, probs in applicable.items():
                values.update(probs.keys())
            outcomes = sorted(values, key=lambda x: float(x) if x.replace('.','',1).isdigit() else 0)
        else:
            return 0.0
        
        current_entropy = self.entropy()
        expected_entropy = 0.0
        for answer in outcomes:
            prob_answer = 0.0
            post_probs = []
            for i, diag in enumerate(self.diagnoses):
                diag_name = diag["name"]
                if diag_name in applicable:
                    if q_type == "yes_no":
                        p_ans = applicable[diag_name].get(answer, {}).get("p", 0.5)
                    elif q_type == "single_choice":
                        p_ans = applicable[diag_name].get(answer, {}).get("p", 1.0/len(options))
                    elif q_type == "scale":
                        p_ans = applicable[diag_name].get(str(answer), {}).get("p", 0.0)
                    else:
                        p_ans = 0.5
                else:
                    if q_type == "yes_no":
                        p_ans = 0.5
                    elif q_type == "single_choice":
                        p_ans = 1.0/len(options)
                    else:
                        p_ans = 1.0/len(outcomes)
                prob_answer += self.probs[i] * p_ans
                post_probs.append(self.probs[i] * p_ans)
            if prob_answer > 0:
                post_probs = np.array(post_probs) / prob_answer
                entropy_post = -np.sum(post_probs * np.log2(post_probs + 1e-12))
                expected_entropy += prob_answer * entropy_post
        return current_entropy - expected_entropy
    
    def select_next_question(self) -> Optional[str]:
        best_qid = None
        best_gain = -1.0
        for qid, q in self.questions_db.items():
            if qid in self.asked_questions:
                continue
            # Пропускаем multi_choice (возвращает 0)
            if q["type"] == "multi_choice":
                continue
            gain = self.information_gain(qid)
            if gain > best_gain:
                best_gain = gain
                best_qid = qid
        return best_qid