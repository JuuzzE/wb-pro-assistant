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
import asyncio

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

# --- Глобальные переменные и блокировка для ленивой загрузки ---
st_model = None
product_collection = None
db_client = None
is_db_loaded = False
db_load_lock = asyncio.Lock() # Блокировка, чтобы только один запрос запускал загрузку

# --- Функции, которые теперь будут вызываться при ленивой загрузке ---
def _load_and_sync_database_sync():
    """Синхронная функция, выполняющая всю тяжелую работу по загрузке."""
    global st_model, product_collection, db_client, is_db_loaded

    print("Запуск синхронизации базы знаний...")
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./st_cache')
    db_client = chromadb.Client() # ephemeral, in-memory
    product_collection = db_client.get_or_create_collection(name="wb_products_in_memory")

    WB_CONTENT_TOKEN = os.environ.get("WB_CONTENT_TOKEN")
    if not WB_CONTENT_TOKEN:
        print("КРИТИЧЕСКАЯ ОШИБКА: Токен WB_CONTENT_TOKEN не найден!")
        return

    headers = {'Authorization': WB_CONTENT_TOKEN, 'Content-Type': 'application/json'}
    all_cards = []
    payload = {"settings": {"cursor": {"limit": 100}, "filter": {"withPhoto": -1}}}

    print("Начинаю загрузку товаров с Wildberries...")
    while True:
        try:
            response = requests.post("https://content-api.wildberries.ru/content/v2/get/cards/list", json=payload, headers=headers)
            if response.status_code != 200:
                print(f"Ошибка от WB API: {response.status_code}, {response.text}")
                break
            data = response.json()
            cards = data.get('cards', [])
            cursor = data.get('cursor', {})
            total = cursor.get('total', 0)
            if not cards:
                print("Сервер WB вернул пустой список, загрузка завершена.")
                break
            all_cards.extend(cards)
            print(f"-> Загружено {len(all_cards)} из (примерно) {total} карточек...")
            if 'updatedAt' not in cursor or 'nmID' not in cursor:
                print("В ответе WB отсутствует курсор, загрузка завершена.")
                break
            if len(all_cards) >= total and total > 0:
                print("Загружены все карточки согласно полю 'total'.")
                break
            payload['settings']['cursor']['updatedAt'] = cursor.get('updatedAt')
            payload['settings']['cursor']['nmID'] = cursor.get('nmID')
            time.sleep(0.7)
        except requests.exceptions.RequestException as e:
            print(f"Сетевая ошибка при запросе к WB API: {e}")
            break
    
    if all_cards:
        print("Подготовка данных для векторной базы в памяти...")
        documents, metadatas, ids = [], [], []
        for card in all_cards:
            nm_id = card.get('nmID')
            title = card.get('title', '')
            description = card.get('description', '')
            brand = card.get('brand', '')
            doc = f"Название: {title}. Бренд: {brand}. Описание: {description}."
            documents.append(doc)
            metadatas.append({'article': nm_id, 'name': title, 'brand': brand})
            ids.append(str(nm_id))
        
        embeddings = st_model.encode(documents).tolist()
        product_collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print(f"База знаний успешно создана в памяти. {len(ids)} товаров.")
        is_db_loaded = True
    else:
        print("Не удалось загрузить товары с WB. База знаний будет пустой.")

async def ensure_db_is_loaded():
    """Асинхронная обертка для ленивой загрузки, использующая блокировку."""
    global is_db_loaded
    if not is_db_loaded:
        async with db_load_lock:
            if not is_db_loaded:
                await asyncio.to_thread(_load_and_sync_database_sync)

# --- Остальные функции ---
class ReviewRequest(BaseModel):
    review_text: str

def get_relevant_products(search_query: str, brand_filter: str = None, exclude_article: str = None, n_results: int = 5):
    if not product_collection or not search_query: return None
    print(f"Сформирован поисковый запрос для базы знаний: '{search_query}'")
    try:
        query_embedding = st_model.encode(search_query).tolist()
        where_clause = {}
        if brand_filter and brand_filter.lower() != "null":
             where_clause["brand"] = brand_filter
        results = product_collection.query(query_embeddings=[query_embedding], where=where_clause if where_clause else None, n_results=n_results)
        if results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                 article = str(meta.get('article'))
                 if article != exclude_article:
                     print(f"Найдена релевантная замена: Артикул {article}")
                     return {"article": article, "description": doc}
        print("Релевантных замен не найдено.")
        return None 
    except Exception as e:
        print(f"Ошибка при поиске по базе товаров: {e}")
        return None

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.5):
    if not perplexity_client: return json.dumps({"error": "API-ключ Perplexity не настроен на сервере."})
    try:
        chat_completion = perplexity_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model="sonar-reasoning",
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Ошибка вызова LLM (Perplexity): {e}")
        return json.dumps({"error": f"Ошибка нейросети: {e}"})

def clean_response(text: str):
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

# --- ГЛАВНЫЙ ЭНДПОИНТ API ---
@app.post("/generate-response")
async def generate_response(request: ReviewRequest):
    await ensure_db_is_loaded()
    
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
    1.  **Никогда** не используй обращения "Дорогая", "Уважаемая". Только "Здравствуйте".
    2.  **Никогда** не используй фразы "не соответствовала Вашим ожиданиям", "продукт для ваших потребностей".
    3.  **Никогда** не повторяй дословно проблему клиента.
    4.  **Никогда** не предлагай пустую помощь ("готовы помочь", "обращайтесь").
    5.  **Никогда** не упоминай артикул товара, на который написан отзыв.
    6.  **Никогда** не говори "передали на производство", если отзыв позитивный.
    7.  **Никогда не придумывай артикулы или детали о товарах.** Используй только те данные, что даны тебе в блоке "рекомендация".
    """

    analysis_system_prompt = f"{PSYCHOTYPES_ANALYSIS_PROMPT}\nПроанализируй отзыв и верни ТОЛЬКО JSON-объект, заключенный в ```json ... ```."
    analysis_user_prompt = f"Отзыв: '{request.review_text}'"
    analysis_result_str = call_llm(analysis_system_prompt, analysis_user_prompt, temperature=0.1)
    try:
        json_match = re.search(r'\{.*\}', analysis_result_str, re.DOTALL)
        if json_match:
            analysis_data = json.loads(json_match.group(0))
        else:
            raise json.JSONDecodeError("JSON не найден в ответе", analysis_result_str, 0)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Ошибка парсинга JSON анализа: {e}")
        return {"analysis": {"error": "Ошибка анализа ответа LLM"}, "response": analysis_result_str}

    sentiment = analysis_data.get("sentiment", "Neutral").lower()
    product_type = analysis_data.get("product_type")
    brand_filter = analysis_data.get("brand_mentioned")
    article_to_exclude = analysis_data.get("product_article")
    main_problem = analysis_data.get("main_problem")

    recommendation_product = None
    if product_type and product_type.lower() != 'null':
        search_query = f"{product_type} {main_problem or ''}"
        recommendation_product = get_relevant_products(search_query, brand_filter, article_to_exclude)

    response_system_prompt = f"{RESPONSE_RULES_PROMPT}\nСобери из блоков идеальный ответ. Твой ответ должен быть ТОЛЬКО текстом для клиента, без лишних слов и объяснений."
    brand_for_signature = brand_filter if brand_filter and brand_filter.lower() != 'null' else "нашего бренда"
    
    user_prompt_blocks = {
        "оригинальный_отзыв": request.review_text,
        "анализ_отзыва": analysis_data
    }

    if sentiment == "positive":
        print("Выбран путь А: Позитивный отзыв.")
        user_prompt_blocks["инструкция"] = f"Напиши теплый, благодарственный ответ на позитивный отзыв от имени бренда '{brand_for_signature}'."
        if recommendation_product:
            user_prompt_blocks["рекомендация"] = recommendation_product
            user_prompt_blocks["инструкция"] += " Органично впиши в ответ рекомендацию, объяснив ее пользу."
    elif sentiment == "negative":
        if recommendation_product:
            print("Выбран путь Б: Негативный отзыв с решением.")
            user_prompt_blocks["инструкция"] = f"Напиши эмпатичный ответ на негативный отзыв от имени бренда '{brand_for_signature}'. Предложи конкретное, **проверенное** решение, порекомендовав найденный товар."
            user_prompt_blocks["рекомендация"] = recommendation_product
        else:
            print("Выбран путь В: Негативный отзыв без решения.")
            user_prompt_blocks["инструкция"] = f"Напиши максимально эмпатичный и грамотный ответ на негативный отзыв от имени бренда '{brand_for_signature}'. Искренне извинись. **СТРОГО ЗАПРЕЩЕНО** что-либо рекомендовать. Просто сообщи, что отзыв принят в работу для улучшения качества."
    else: 
        print("Выбран путь для нейтрального отзыва.")
        user_prompt_blocks = { "инструкция": f"Напиши вежливый и сдержанный ответ на нейтральный отзыв от имени бренда '{brand_for_signature}'." }
        if recommendation_product:
            user_prompt_blocks["рекомендация"] = recommendation_product
            user_prompt_blocks["инструкция"] += " Если уместно, впиши рекомендацию."
            
    final_user_prompt = f"""
    Вот данные для работы:
    {json.dumps(user_prompt_blocks, ensure_ascii=False, indent=2)}

    Сгенерируй только текст финального ответа для клиента, и ничего больше.
    """
    final_response_text_raw = call_llm(response_system_prompt, final_user_prompt, temperature=0.7)
    final_response_text = clean_response(final_response_text_raw)

    return {"analysis": analysis_data, "response": final_response_text}

@app.get("/")
def read_root():
    return {"status": "WB Pro Assistant Backend is running!"}