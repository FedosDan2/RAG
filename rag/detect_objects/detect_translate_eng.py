from langdetect import detect
from deep_translator import GoogleTranslator
from typing import List

# === 2. Переводчик ===
class GoogleFreeTranslator:
    def __init__(self, target_lang='en'):
        self.target_lang = target_lang
        self.separator = ' ||| '
        
    def translate_to_english(self, text: str) -> str:
        self.person_lang = detect(text)
        
        if self.person_lang == 'en':
            return text
        
        try:
            translator = GoogleTranslator(source=self.person_lang, target=self.target_lang)
            result = translator.translate(text)
            return result
        except Exception as e:
            print(f"⚠️ Ошибка перевода: {e}")
            return text
        

    def translate_to_person_lang(self, type: str, text: List[str]) -> List[str]:
        if not text:
            return text
        
        if self.person_lang == self.target_lang:
            return text
        
        add_context = []
        if type == 'symptoms':
            add_context = [f"The patient have symptoms: {term}" for term in text]    
        elif type == 'medication':
            add_context = [f"The patient have medication: {term}" for term in text]    
       
        joined_text = self.separator.join(add_context)
        try:
            translator = GoogleTranslator(source=self.target_lang, target=self.person_lang)
            translated_text = translator.translate(joined_text)
            splited_separator = translated_text.split(self.separator)
            result = []
            for phrase in splited_separator:
                translated_term = phrase.split(': ')[-1].strip()
                result.append(translated_term)
            
            return result
        
        except Exception as e:
            print(f"⚠️ Ошибка перевода: {e}")
            return text
        