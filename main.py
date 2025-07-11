import os
import json
import re 
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

# <<< Промпты v14.0: Упрощены, т.к. часть логики перенесена в код >>>
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

# --- БАЗА ЗНАНИЙ О ТОВАРАХ ---
DATA_DIR = os.environ.get("RENDER_DISK_PATH", "./db")
try:
    st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='./st_cache')
    db_client = chromadb.PersistentClient(path=DATA_DIR)
    product_collection = db_client.get_collection(name="wb_products_stable")
    print("База знаний по товарам успешно загружена.")
except Exception as e:
    product_collection = None
    print(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке базы знаний: {e}")

# --- МОДЕЛИ ДАННЫХ И ФУНКЦИИ ---
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
            model="sonar-pro",
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
        return {"analysis": analysis_data, "response": analysis_result_str}

    sentiment = analysis_data.get("sentiment", "Neutral").lower()
    product_type = analysis_data.get("product_type")
    brand_filter = analysis_data.get("brand_mentioned")
    article_to_exclude = analysis_data.get("product_article")
    main_problem = analysis_data.get("main_problem")

    recommendation_product = None
    # <<< ИЗМЕНЕНИЕ v14.0: Улучшенная логика поиска для позитивных отзывов >>>
    if product_type and product_type.lower() != 'null':
        search_query = ""
        # Для позитивных и нейтральных отзывов ищем просто похожие товары того же типа
        if sentiment in ["positive", "neutral"]:
            search_query = product_type
        # Для негативных ищем решение проблемы
        elif main_problem and main_problem.lower() != 'null':
            search_query = f"{product_type} {main_problem}"
        
        if search_query:
            recommendation_product = get_relevant_products(search_query, brand_filter, article_to_exclude)

    response_system_prompt = f"{RESPONSE_RULES_PROMPT}\nТвоя задача - написать тело ответа. Не включай приветствие и подпись."
    brand_for_signature = brand_filter if brand_filter and brand_filter.lower() != 'null' else "нашего бренда"
    
    user_prompt_blocks = {
        "оригинальный_отзыв": request.review_text,
        "анализ_отзыва": analysis_data
    }

    if recommendation_product:
        user_prompt_blocks["инструкция"] = f"Напиши текст ответа, органично вписав в него рекомендацию. Обоснуй ее пользу."
        user_prompt_blocks["рекомендация"] = recommendation_product
    else:
        user_prompt_blocks["инструкция"] = f"Напиши текст ответа. СТРОГО ЗАПРЕЩЕНО что-либо рекомендовать."

    final_user_prompt = f"""
    Вот данные для работы:
    {json.dumps(user_prompt_blocks, ensure_ascii=False, indent=2)}

    Сгенерируй только основное тело ответа. БЕЗ "Здравствуйте" и БЕЗ "С уважением...".
    """
    body_response_raw = call_llm(response_system_prompt, final_user_prompt, temperature=0.7)
    body_response = clean_response(body_response_raw)

    # <<< ИЗМЕНЕНИЕ v14.0: Программно собираем финальный ответ >>>
    customer_name = analysis_data.get("customer_name", "Покупатель")
    greeting = f"Здравствуйте, {customer_name}!" if customer_name and customer_name.lower() != 'покупатель' else "Здравствуйте!"
    signature = f"С уважением, команда {brand_for_signature}."
    
    final_response_text = f"{greeting}\n\n{body_response}\n\n{signature}"
    
    return {"analysis": analysis_data, "response": final_response_text}

@app.get("/")
def read_root():
    return {"status": "WB Pro Assistant Backend is running!"}