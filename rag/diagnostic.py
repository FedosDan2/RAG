import math
import numpy as np
from typing import List, Dict, Optional, Any

class DiagnosticSession:
    def __init__(self, diagnoses: List[dict], questions_db: Dict[str, dict], initial_probs: List[float]):
        """
        diagnoses: список словарей, каждый содержит "id" и "name"
        questions_db: словарь {question_id: question_object}
        initial_probs: начальные вероятности
        """
        self.diagnoses = diagnoses          # список с полями id, name, required_tests, required_doctors
        self.questions_db = questions_db
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
        
        likelihood = []
        for diag in self.diagnoses:
            diag_id = diag["id"]            # используем ID для поиска
            if diag_id in applicable:
                probs_for_diag = applicable[diag_id]
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
                    # Сопоставляем числовой ответ с интервалами
                    val = float(answer)
                    matched_key = None
                    for key in probs_for_diag.keys():
                        if key.startswith("<") and val < float(key[1:]):
                            matched_key = key
                            break
                        elif key.startswith(">") and val > float(key[1:]):
                            matched_key = key
                            break
                        elif "-" in key:
                            low, high = map(float, key.split("-"))
                            if low <= val <= high:
                                matched_key = key
                                break
                    if matched_key:
                        prob = probs_for_diag[matched_key].get("p", 0.0)
                    else:
                        prob = 0.0
                else:
                    prob = 0.5
            else:
                # нейтральное распределение
                if q_type == "yes_no":
                    prob = 0.5
                elif q_type == "single_choice":
                    prob = 1.0 / len(options) if options else 0.5
                elif q_type == "multi_choice":
                    prob = 0.5 ** len(options)
                elif q_type == "scale":
                    # Для шкалы по умолчанию равномерная вероятность по всем интервалам
                    # Возьмём количество интервалов из первого попавшегося диагноза
                    intervals = list(applicable.values())[0].keys() if applicable else []
                    prob = 1.0 / len(intervals) if intervals else 0.5
                else:
                    prob = 0.5
            likelihood.append(prob)
        
        likelihood = np.array(likelihood)
        new_probs = self.probs * likelihood
        total = new_probs.sum()
        if total > 0:
            self.probs = new_probs / total
        else:
            self.probs = np.ones_like(self.probs) / len(self.probs)
        
        self.asked_questions.add(question_id)
        self.history.append((question_id, answer))
    
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
            return 0.0
        elif q_type == "scale":
            # Собираем все возможные интервалы из applicable_to
            intervals = set()
            for probs in applicable.values():
                intervals.update(probs.keys())
            outcomes = sorted(intervals, key=lambda x: float(x.split('-')[0].replace('<','').replace('>','')) if x[0] in '<>' else float(x.split('-')[0]))
        else:
            return 0.0
        
        current_entropy = self.entropy()
        expected_entropy = 0.0
        for answer in outcomes:
            prob_answer = 0.0
            post_probs = []
            for i, diag in enumerate(self.diagnoses):
                diag_id = diag["id"]
                if diag_id in applicable:
                    if q_type == "yes_no":
                        p_ans = applicable[diag_id].get(answer, {}).get("p", 0.5)
                    elif q_type == "single_choice":
                        p_ans = applicable[diag_id].get(answer, {}).get("p", 1.0/len(options))
                    elif q_type == "scale":
                        p_ans = applicable[diag_id].get(answer, {}).get("p", 0.0)
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
            if q["type"] == "multi_choice":
                continue
            gain = self.information_gain(qid)
            if gain > best_gain:
                best_gain = gain
                best_qid = qid
        return best_qid