import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time
import json

# --- НАСТРОЙКИ ---
load_dotenv()
WB_CONTENT_TOKEN = os.environ.get("WB_CONTENT_TOKEN")
WB_API_URL = "https://content-api.wildberries.ru/content/v2/get/cards/list"

DATA_DIR = os.environ.get("RENDER_DISK_PATH", "./db")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- ИНИЦИАЛИЗАЦИЯ ---
print("Инициализация векторной базы данных...")
st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./st_cache')
db_client = chromadb.PersistentClient(path=DATA_DIR)
# Используем новое имя коллекции, чтобы гарантировать чистоту данных
product_collection = db_client.get_or_create_collection(name="wb_products_stable") 
print("Инициализация завершена.")

def fetch_all_wb_products():
    """Получает ВСЕ карточки товаров с WB API, используя надежную пагинацию."""
    if not WB_CONTENT_TOKEN:
        print("ОШИБКА: Токен WB_CONTENT_TOKEN не найден в .env файле.")
        return None
        
    headers = {
        'Authorization': WB_CONTENT_TOKEN,
        'Content-Type': 'application/json'
    }
    all_cards = []
    
    # === РАБОЧИЙ ВАРИАНТ ЗАПРОСА (БЕЗ 'sort') ===
    # Этот вариант успешно прошел проверку и вернул 200 OK
    payload = {
      "settings": {
        "cursor": {
          "limit": 100 
        },
        "filter": {
          "withPhoto": -1
        }
      }
    }

    print("Начинаю полную загрузку товаров с Wildberries (стабильный режим)...")
    total_from_server = 0

    while True:
        try:
            print(f"\nОтправляю запрос на получение следующей порции ({len(all_cards)} уже загружено)...")
            
            response = requests.post(WB_API_URL, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"!!! ПОЛУЧЕНА ОШИБКА: {response.status_code} !!!")
                print(f"Ответ сервера: {response.text}")
                break

            data = response.json()
            
            cards = data.get('cards', [])
            cursor = data.get('cursor', {})
            if 'total' in cursor:
                 total_from_server = cursor.get('total')
            
            if not cards:
                print("Сервер вернул пустой список карточек. Загрузка завершена.")
                break

            all_cards.extend(cards)
            print(f"-> Загружено {len(all_cards)} из (примерно) {total_from_server} карточек...")
            
            if 'updatedAt' not in cursor or 'nmID' not in cursor:
                print("В ответе сервера отсутствует курсор для продолжения. Загрузка остановлена.")
                break

            payload['settings']['cursor']['updatedAt'] = cursor.get('updatedAt')
            payload['settings']['cursor']['nmID'] = cursor.get('nmID')
            
            # Задержка 0.7 секунды, чтобы не превышать лимит в 100 запросов/минуту
            time.sleep(0.7)

        except requests.exceptions.RequestException as e:
            print(f"Критическая ошибка при запросе к WB API: {e}")
            return None

    print(f"\nИтоговая загрузка завершена. Всего получено: {len(all_cards)} карточек.")
    return all_cards

def update_vector_database(cards):
    """Обновляет ChromaDB на основе полученных карточек."""
    if not cards:
        print("Нет карточек для обновления базы данных.")
        return

    print("\nПодготовка данных для векторной базы...")
    documents, metadatas, ids = [], [], []

    for card in cards:
        nm_id = card.get('nmID')
        title = card.get('title', '')
        description = card.get('description', '')
        brand = card.get('brand', '')
        
        characteristics_text = " ".join([f"{char.get('name')}: {char.get('value')}" for char in card.get('characteristics', []) if isinstance(char.get('value'), (str, int, float))])
        doc = f"Название: {title}. Бренд: {brand}. Описание: {description}. Характеристики: {characteristics_text}"
        
        documents.append(doc)
        metadatas.append({'article': nm_id, 'name': title})
        ids.append(str(nm_id))

    print(f"Начинаю обновление {len(ids)} векторов в базе. Это может занять время...")
    
    try:
        db_client.delete_collection(name="wb_products_stable")
        print("Старая коллекция удалена для полного обновления.")
    except Exception:
        print("Старая коллекция не найдена, создается новая.")

    product_collection = db_client.get_or_create_collection(name="wb_products_stable")
    
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end_index = i + batch_size
        print(f"Векторизую и добавляю порцию товаров с {i} по {end_index-1}...")
        batch_embeddings = st_model.encode(documents[i:end_index]).tolist()
        product_collection.add(
            ids=ids[i:end_index],
            embeddings=batch_embeddings,
            documents=documents[i:end_index],
            metadatas=metadatas[i:end_index]
        )

    print("\nБаза знаний по всем товарам успешно обновлена!")

if __name__ == "__main__":
    all_products = fetch_all_wb_products()
    if all_products:
        update_vector_database(all_products)
    else:
        print("\nСинхронизация не удалась. Проверьте сообщения об ошибках выше.")