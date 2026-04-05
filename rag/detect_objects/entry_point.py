from rag.detect_objects.detect_translate_eng import GoogleFreeTranslator
from rag.detect_objects.extract_med_obj import Extractor

def get_features(text: str, device):
    # для переведа запроса на англ чтобы в ner закинуть
    translator = GoogleFreeTranslator()

    # ner экстрактор 
    extractor = Extractor(device=device)

    result = translator.translate_to_english(text)
    result_group = extractor.extract_all(result, group_entities=True)
    symptoms = translator.translate_to_person_lang(text=result_group['symptoms'], type='symptoms')
    medication = translator.translate_to_person_lang(text=result_group['medication'], type='medication')

    return symptoms, medication


#print(get_features('У меня второй день держится высокая температура и кашель'))