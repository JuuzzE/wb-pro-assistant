# sync_products.py (версия для Supabase)
import os
import requests
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv
import time

load_dotenv()

# --- НАСТРОЙКИ ---
WB_CONTENT_TOKEN = os.environ.get("WB_CONTENT_TOKEN")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
WB_API_URL = "https://content-api.wildberries.ru/content/v2/get/cards/list"

if not all([WB_CONTENT_TOKEN, SUPABASE_URL, SUPABASE_KEY]):
    raise Exception("Ошибка: Не все переменные окружения (.env) настроены!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def fetch_all_wb_products():
    # ... (эта функция остается без изменений из версии 15.1) ...
    headers = {'Authorization': WB_CONTENT_TOKEN}
    all_cards = []
    payload = {"settings": {"cursor": {"limit": 100}, "filter": {"withPhoto": -1}}}
    print("Начинаю полную загрузку товаров с Wildberries...")
    while True:
        try:
            response = requests.post(WB_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            cards = data.get('cards', [])
            if not cards: break
            all_cards.extend(cards)
            print(f"-> Загружено {len(all_cards)} карточек...")
            cursor = data.get('cursor', {})
            payload['settings']['cursor']['updatedAt'] = cursor.get('updatedAt')
            payload['settings']['cursor']['nmID'] = cursor.get('nmID')
            time.sleep(0.7)
        except requests.exceptions.RequestException as e:
            print(f"Критическая ошибка при запросе к WB API: {e}")
            return None
    print(f"Итоговая загрузка завершена. Всего получено: {len(all_cards)} карточек.")
    return all_cards

def update_supabase_database(cards):
    if not cards:
        print("Нет карточек для обновления.")
        return

    print("Очистка старых данных в Supabase...")
    supabase.table('products').delete().gt('id', 0).execute()
    
    print("Подготовка и загрузка новых данных...")
    records_to_insert = []
    for card in cards:
        nm_id = card.get('nmID')
        title = card.get('title', '')
        brand = card.get('brand', '')
        description = card.get('description', '')
        content = f"Название: {title}. Бренд: {brand}. Описание: {description}"
        
        records_to_insert.append({
            "article": nm_id,
            "brand": brand,
            "name": title,
            "content": content,
            "embedding": st_model.encode(content).tolist()
        })
        if len(records_to_insert) >= 50: # Загружаем порциями по 50 штук
            print(f"Загружаю порцию из {len(records_to_insert)} записей...")
            supabase.table('products').insert(records_to_insert).execute()
            records_to_insert = []
    
    if records_to_insert: # Загружаем остаток
        print(f"Загружаю финальную порцию из {len(records_to_insert)} записей...")
        supabase.table('products').insert(records_to_insert).execute()

    print("База знаний в Supabase успешно обновлена!")

if __name__ == "__main__":
    all_products = fetch_all_wb_products()
    if all_products:
        update_supabase_database(all_products)