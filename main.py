import os
import json
import re 
import requests
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. ИНИЦИАЛИЗАЦИЯ ---
load_dotenv()
app = FastAPI(title="WB Pro Assistant API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    perplexity_client = OpenAI(
        api_key=os.environ.get("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
except Exception as e:
    perplexity_client = None

# <<< Промпты остаются те же >>>
PSYCHOTYPES_ANALYSIS_PROMPT = """
Ты — сверхточный эксперт-аналитик. Твоя задача — без ошибок извлечь из отзыва покупателя обуви структурированную информацию.
# Задача:
Проанализируй отзыв и верни ТОЛЬКО JSON-объект. ЗАПОЛНИ ВСЕ ПОЛЯ.
{
  "customer_name": "Имя клиента из отзыва, или 'Покупатель' если имени нет.",
  "product_article": "Артикул WB, на который написан отзыв. Найди его в тексте или метаданных. Если артикула нет, укажи null.",
  "product_type": "ОПРЕДЕЛИ ТИП ОБУВИ одним словом (например: 'сабо', 'босоножки', 'кеды', 'туфли'). Если определить невозможно, укажи null.",
  "brand_mentioned": "МАКСИМАЛЬНО ТОЧНОЕ название бренда из отзыва (например, 'Dino Ricci Select').",
  "sentiment": "Positive/Negative/Neutral",
  "psychotype": "Тестировщик/Импульсивная королева/Заботливая мама/Чеклист-реалист/Колеблющийся",
  "main_problem": "Краткая суть проблемы клиента (например, 'стерлась подошва', 'жесткий ремешок'). Если отзыв позитивный, укажи null."
}
"""
RESPONSE_RULES_PROMPT = """
# Твоя личность и ГЛАВНОЕ ПРАВИЛО:
Твоя основная задача — писать как образованный, интеллигентный и эмпатичный носитель русского языка. Речь должна быть абсолютно грамотной, естественной и человечной. Ошибки в грамматике и стилистике недопустимы. Ты — менеджер премиального бренда обуви.

# ЗАПРЕЩЕННЫЕ ДЕЙСТВИЯ И ФРАЗЫ:
- Никогда не используй обращения "Дорогая", "Уважаемая".
- Никогда не используй фразы "не соответствовала Вашим ожиданиям", "продукт для ваших потребностей".
- Никогда не повторяй дословно проблему клиента.
- Никогда не предлагай пустую помощь ("готовы помочь", "обращайтесь").
- Никогда не упоминай артикул товара, на который написан отзыв.
- Никогда не придумывай артикулы. Используй только те данные, что даны тебе в блоке "рекомендация".

# СТИЛЬ ОТВЕТА:
1.  **Эмпатия и Признание:** Поблагодари за обратную связь. Если отзыв негативный, искренне извинись.
2.  **Рекомендации (ТОЛЬКО ЕСЛИ ДАНЫ В ПРОМПТЕ):**
    *   **Обоснование:** Твоя рекомендация должна логически решать `main_problem` клиента или дополнять его позитивный опыт.
    *   **Формулировка:** "Раз Вам понравилась эта пара, возможно, Вас заинтересует и другая наша модель...", "В качестве альтернативы хотели бы предложить...".
"""


# --- Глобальные переменные для хранения базы в памяти ---
st_model = None
product_collection = None
db_client = None

# --- ЛОГИКА СИНХРОНИЗАЦИИ, КОТОРАЯ ЗАПУСКАЕТСЯ ПРИ СТАРТЕ СЕРВЕРА ---
@app.on_event("startup")
def load_and_sync_database():
    global st_model, product_collection, db_client

    print("Запуск сервера: начинаю загрузку и синхронизацию базы знаний...")
    
    # Инициализируем модели и клиент ChromaDB в памяти
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./st_cache')
    db_client = chromadb.Client() # ephemeral, in-memory
    product_collection = db_client.get_or_create_collection(name="wb_products_in_memory")

    # --- Код для загрузки с WB API (из старого sync_products.py) ---
    WB_CONTENT_TOKEN = os.environ.get("WB_CONTENT_TOKEN")
    WB_API_URL = "https://content-api.wildberries.ru/content/v2/get/cards/list"

    if not WB_CONTENT_TOKEN:
        print("КРИТИЧЕСКАЯ ОШИБКА: Токен WB_CONTENT_TOKEN не найден!")
        return

    headers = {'Authorization': WB_CONTENT_TOKEN, 'Content-Type': 'application/json'}
    all_cards = []
    payload = {"settings": {"cursor": {"limit": 100}, "filter": {"withPhoto": -1}}}

    print("Начинаю загрузку товаров с Wildberries...")
    while True:
        try:
            response = requests.post(WB_API_URL, json=payload, headers=headers)
            if response.status_code != 200: break
            data = response.json()
            cards = data.get('cards', [])
            cursor = data.get('cursor', {})
            total = cursor.get('total', 0)
            if not cards: break
            all_cards.extend(cards)
            print(f"-> Загружено {len(all_cards)} из (примерно) {total} карточек...")
            if len(all_cards) >= total and total > 0: break
            payload['settings']['cursor']['updatedAt'] = cursor.get('updatedAt')
            payload['settings']['cursor']['nmID'] = cursor.get('nmID')
            time.sleep(0.7)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к WB API: {e}")
            break
    
    # --- Заполнение базы в памяти ---
    if all_cards:
        print("Подготовка данных для векторной базы в памяти...")
        documents, metadatas, ids = [], [], []
        for card in all_cards:
            nm_id = card.get('nmID')
            title = card.get('title', '')
            description = card.get('description', '')
            brand = card.get('brand', '')
            characteristics_text = " ".join([f"{char.get('name')}: {char.get('value')}" for char in card.get('characteristics', []) if isinstance(char.get('value'), (str, int, float))])
            doc = f"Название: {title}. Бренд: {brand}. Описание: {description}. Характеристики: {characteristics_text}"
            documents.append(doc)
            metadatas.append({'article': nm_id, 'name': title, 'brand': brand})
            ids.append(str(nm_id))
        
        embeddings = st_model.encode(documents).tolist()
        product_collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print(f"База знаний успешно создана в памяти. {len(ids)} товаров.")
    else:
        print("Не удалось загрузить товары с WB. База знаний будет пустой.")


# --- МОДЕЛИ ДАННЫХ И ФУНКЦИИ ---
class ReviewRequest(BaseModel):
    review_text: str

def get_relevant_products(search_query: str, brand_filter: str = None, exclude_article: str = None, n_results: int = 5):
    # Эта функция теперь работает с глобальными переменными
    if not product_collection or not search_query: return None
    # ... (остальной код функции get_relevant_products остается без изменений) ...

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.5):
    # ... (код функции call_llm остается без изменений) ...

def clean_response(text: str):
    # ... (код функции clean_response остается без изменений) ...

# --- ГЛАВНЫЙ ЭНДПОИНТ API ---
@app.post("/generate-response")
async def generate_response(request: ReviewRequest):
    # ... (весь код эндпоинта generate_response остается без изменений) ...

@app.get("/")
def read_root():
    return {"status": "WB Pro Assistant Backend is running!"}