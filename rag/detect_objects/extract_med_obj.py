from transformers import pipeline
from typing import List, Dict
import re

class Extractor:
    def __init__(self, task="token-classification", model="d4data/biomedical-ner-all", device="cuda"):
        self.model_name = model
        self.task = task
        self.device = device
        self.pipe = pipeline(task=task, model=model, device=device)
    
    def extract_words(self, structure: List[Dict], text: str) -> List[Dict]:
        """
        Извлекает полные слова, удаляет дубликаты и подстроки.
        Возвращает очищенный список сущностей.
        """
        if not structure:
            return structure
        
        # 1. Извлекаем полные слова для каждой сущности
        for entry in structure:
            start = entry['start']
            end = entry['end']
            
            # Расширяем границы влево
            while start > 0 and text[start - 1] not in ' .,;!?\n' and text[start - 1] != '-':
                start -= 1
            # Расширяем границы вправо
            while end < len(text) and text[end] not in ' .,;!?\n' and text[end] != '-':
                end += 1
            
            word = text[start:end]
            entry['full_word'] = word
            entry['clean_start'] = start
            entry['clean_end'] = end
        
        # 2. Удаляем точные дубликаты (одинаковый текст)
        seen_words = set()
        unique_structure = []
        for entry in structure:
            word = entry.get('full_word', '').lower()
            if word not in seen_words:
                seen_words.add(word)
                unique_structure.append(entry)
                
        final_structure = []
        for entry in unique_structure:
            word = entry.get('full_word', '').lower()
            
            # Проверяем, не является ли слово подстрокой уже добавленных
            is_substring = False
            for existing in final_structure:
                existing_word = existing.get('full_word', '').lower()
                if word in existing_word:
                    is_substring = True
                    break
            
            if not is_substring:
                final_structure.append(entry)
        
        return final_structure
    
    def group_related_entities(self, structure: List[Dict], text: str, max_distance=3) -> List[Dict]:
        """Объединяет связанные сущности (симптом + значение, лекарство + дозировка)"""
        if not structure:
            return structure
        
        # Сортируем по позиции
        sorted_entities = sorted(structure, key=lambda x: x['clean_start'])
        grouped = []
        used = set()
        
        for i, entity in enumerate(sorted_entities):
            if i in used:
                continue
            
            current_text = entity.get('full_word', '')
            current_end = entity['clean_end']
            merged_entities = [entity]
            
            for j in range(i + 1, len(sorted_entities)):
                if j in used:
                    continue
                
                next_entity = sorted_entities[j]
                next_start = next_entity['clean_start']
                next_word = next_entity.get('full_word', '')
                
                distance = next_start - current_end
                
                # Проверяем стоит ли объединять
                should_merge = False
                
                # 1. Очень близко
                if 0 <= distance <= max_distance:
                    should_merge = True
                
                # 2. Между ними только разделители
                between_text = text[current_end:next_start].strip()
                if between_text and re.match(r'^[\s,\-\(\)]+$', between_text):
                    should_merge = True
                
                # 3. Следующая сущность - число/измерение
                if re.match(r'^[\d.]+\s*[°%]?[CFcmgkL]+$', next_word, re.IGNORECASE):
                    should_merge = True
                
                if should_merge:
                    between = text[current_end:next_start]
                    current_text = current_text + between + next_word
                    current_end = next_entity['clean_end']
                    merged_entities.append(next_entity)
                    used.add(j)
                else:
                    break
            
            merged_entity = {
                'entity': entity.get('entity', ''),
                'full_word': current_text,
                'start': entity.get('start', 0),
                'end': current_end,
                'clean_start': entity.get('clean_start', 0),
                'clean_end': current_end,
                'merged_count': len(merged_entities)
            }
            grouped.append(merged_entity)
        
        return grouped
    
    def extract_symptom(self, data: List[Dict], no_group=None) -> List[str]:
        """Извлекает симптомы и заболевания"""
        symptoms = []
        for entry in data:
            entity_name = entry.get('entity', '').lower()
            word = entry.get('full_word', '')
            if no_group==None:
                if 'disease' in entity_name or 'symptom' in entity_name:
                    symptoms.append(word.strip())
            else:
                if 'disease' in entity_name or 'symptom' in entity_name:
                    symptoms.append(word.strip())
                    continue

                for not_groupped in no_group:
                    word_no_group = not_groupped.get('full_word', '')
                    entity_name_no_group = not_groupped.get('entity', '').lower()

                    if word_no_group in word and ('disease' in entity_name_no_group or 'symptom' in entity_name_no_group):
                        symptoms.append(word.strip())
                        break
        
        return symptoms
    
    def extract_medication(self, data: List[Dict], no_group=None) -> List[str]:
        """Извлекает лекарства и процедуры"""
        medications = []
        for entry in data:
            entity_name = entry.get('entity', '').lower()
            word = entry.get('full_word', '')
            if no_group==None:
                if 'medication' in entity_name or 'procedure' in entity_name:
                    medications.append(word.strip())
            else:
                if 'medication' in entity_name or 'procedure' in entity_name:
                    medications.append(word.strip())
                    continue

                for not_groupped in no_group:
                    word_no_group = not_groupped.get('full_word', '')
                    entity_name_no_group = not_groupped.get('entity', '').lower()

                    if word_no_group in word and ('medication' in entity_name_no_group or 'procedure' in entity_name_no_group):
                        medications.append(word.strip())
                        break
                    
            
        return medications
    
    def extract_all(self, text: str, group_entities: bool = True) -> Dict:
        """Универсальный метод для извлечения всех сущностей"""
        structure = self.pipe(text)
        structure_no_group = self.extract_words(structure, text)
        
        if group_entities:
            structure = self.group_related_entities(structure_no_group, text)
            structure = self.extract_words(structure, text)
        
        if not structure:
            structure = structure_no_group
            structure_no_group = None

        all_entities = [entry['full_word'] for entry in structure]
        
        
        return {
            'original_text': text,
            'all_entities': all_entities,
            'symptoms': self.extract_symptom(structure, structure_no_group),
            'medication': self.extract_medication(structure, structure_no_group),
            'raw_structure': structure
        }