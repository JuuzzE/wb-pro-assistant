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

# <<< ИЗМЕНЕНИЕ v16.1: Финальный промпт с жесткими запретами и примерами >>>
PSYCHOTYPES_ANALYSIS_PROMPT = """
Ты — эксперт-психолог и data-аналитик с глубоким пониманием клиентов e-commerce. Твоя задача — провести всесторонний анализ отзыва покупателя обуви.
# Карта психотипов (модель DISC + адаптация):
1.  **Доминантный (D - Dominant): "Тестировщик", "Лидер"**
    *   **Мотивация:** Контроль, результат, качество, эффективность.
    *   **Маркеры в тексте:** Короткие, ясные фразы; факты, цифры; сравнение с другими; требование конкретики ("где сертификат?", "сколько см стелька?"). Тон часто требовательный или констатирующий.
2.  **Влияющий (I - Influential): "Энтузиаст", "Импульсивная королева"**
    *   **Мотивация:** Эмоции, признание, новизна, общение, внешний вид.
    *   **Маркеры в тексте:** Эмоциональные слова ("влюбилась!", "шикарные", "восторг"); суперлативы; много восклицательных знаков; фото; рассказ о том, как на них отреагировали другие.
3.  **Постоянный (S - Steady): "Заботливая мама", "Гармонизатор"**
    *   **Мотивация:** Безопасность, надежность, комфорт, предсказуемость, забота о близких.
    *   **Маркеры в тексте:** Вопросы об уходе, гарантиях, материалах; упоминание детей/семьи ("брала сыну"); фокус на комфорте и безопасности ("не скользят?"). Тон спокойный, ищет поддержки.
4.  **Добросовестный (C - Conscientious): "Аналитик", "Чеклист-реалист"**
    *   **Мотивация:** Точность, детали, правила, логика, соответствие заявленному.
    *   **Маркеры в тексте:** Структурированный отзыв (плюсы/минусы); перечисление характеристик; точные вопросы о деталях (вес, состав подошвы); сравнение с описанием в карточке.

# Задача:
Проанализируй отзыв и верни ТОЛЬКО JSON-объект. ЗАПОЛНИ ВСЕ ПОЛЯ.
{
  "customer_name": "Имя клиента из отзыва, или 'Покупатель' если имени нет.",
  "product_article": "Артикул WB, на который написан отзыв. Если его нет, укажи null.",
  "product_type": "ОПРЕДЕЛИ ТИП ОБУВИ одним словом (например: 'тапочки', 'сабо', 'угги', 'туфли').",
  "product_season": "ОПРЕДЕЛИ СЕЗОН обуви ('лето', 'зима', 'демисезон'). Если неясно, укажи 'всесезонный'.",
  "brand_mentioned": "МАКСИМАЛЬНО ТОЧНОЕ название бренда из отзыва.",
  "sentiment": "Positive/Negative/Neutral",
  "customer_emotion": "КЛЮЧЕВАЯ ЭМОЦИЯ КЛИЕНТА (например: 'Радость и удовлетворение', 'Разочарование в качестве', 'Сомнение в выборе', 'Гнев из-за брака').",
  "psychotype": "Определи доминирующий психотип из карты: 'Доминантный', 'Влияющий', 'Постоянный', 'Добросовестный'.",
  "main_problem": "Краткая суть проблемы клиента. Если отзыв позитивный, укажи null."
}
"""
RESPONSE_RULES_PROMPT = """
# Твоя личность и ГЛАВНОЕ ПРАВИЛО:
Твоя основная задача — писать как чуткий, образованный и интеллигентный менеджер премиального бренда обуви. Твоя речь абсолютно грамотна и естественна. Твой главный приоритет — клиент и его чувства.

# ЗАПРЕЩЕННЫЕ ДЕЙСТВИЯ:
1.  **НИКОГДА** не упоминай в ответе название психотипа ("добросовестный", "влияющий" и т.д.). Эта информация дана тебе только для выбора правильного тона.
2.  **НИКОГДА** не используй обращения "Дорогая", "Уважаемая".
3.  **НИКОГДА** не используй канцелярит ("продукт", "ваши потребности", "соответствовало ожиданиям").
4.  **НИКОГДА** не вставляй в текст сноски, цифры в квадратных скобках `[1]` или любые другие артефакты. Ответ должен быть чистым текстом.
5.  **НИКОГДА** не придумывай артикулы.

# СТИЛЬ И СТРУКТУРА ОТВЕТА (ОБУЧАЮЩИЕ ПРИМЕРЫ):

**Пример для психотипа "Добросовестный" (C):**
*   **НЕПРАВИЛЬНО:** "Мы рады, что такие добросовестные люди, как Вы, нас выбирают[1][2]."
*   **ПРАВИЛЬНО:** "Большое спасибо за такой подробный и взвешенный отзыв. Нам особенно приятно получать обратную связь от людей, которые так внимательны к деталям." (Реагируем на СУТЬ психотипа, а не на его НАЗВАНИЕ).

**Пример для психотипа "Влияющий" (I):**
*   **НЕПРАВИЛЬНО:** "Вы, как влияющий человек, сделали отличный выбор!"
*   **ПРАВИЛЬНО:** "Мы в восторге от Ваших фотографий! Эти сабо действительно выглядят на Вас потрясающе." (Реагируем на ЭМОЦИЮ).

# ТВОЯ ЗАДАЧА:
Напиши основной текст ответа. Он должен быть сфокусирован на реакции на отзыв клиента и его эмоции. Рекомендацию давай только если она дана тебе в промпте, и делай это кратко и ненавязчиво.
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

def get_relevant_products(product_type: str, season: str, search_context: str, brand_filter: str = None, exclude_article: str = None, n_results: int = 5):
    if not all([product_collection, product_type, season, search_context]): return None
    contextual_search_query = f"{season} {product_type} {search_context}"
    print(f"Сформирован поисковый запрос для базы знаний: '{contextual_search_query}'")
    try:
        query_embedding = st_model.encode(contextual_search_query).tolist()
        where_clause = {}
        if brand_filter and brand_filter.lower() != "null":
            where_clause["brand"] = brand_filter
        results = product_collection.query(query_embeddings=[query_embedding], where=where_clause if where_clause else None, n_results=n_results)
        if results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                 article = str(meta.get('article'))
                 doc_lower = doc.lower()
                 if season in doc_lower and product_type in doc_lower and article != exclude_article:
                     print(f"Найдена релевантная замена: Артикул {article}")
                     return {"article": article, "description": doc}
        print("Релевантных замен, прошедших фильтр, не найдено.")
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
    # Удаляем блок <think> и любые цифры в квадратных скобках
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

# --- ГЛАВНЫЙ ЭНДПОИНТ API ---
@app.post("/generate-response")
async def generate_response(request: ReviewRequest):
    analysis_system_prompt = f"{PSYCHOTYPES_ANALYSIS_PROMPT}"
    analysis_user_prompt = f"Отзыв: '{request.review_text}'"
    analysis_result_str = call_llm(analysis_system_prompt, analysis_user_prompt, temperature=0.1)
    try:
        json_match = re.search(r'\{.*\}', analysis_result_str, re.DOTALL)
        analysis_data = json.loads(json_match.group(0)) if json_match else {}
    except (json.JSONDecodeError, TypeError) as e:
        return {"analysis": {"error": "Ошибка анализа ответа LLM"}, "response": analysis_result_str}

    sentiment = analysis_data.get("sentiment", "Neutral").lower()
    product_type = analysis_data.get("product_type")
    product_season = analysis_data.get("product_season")
    brand_filter = analysis_data.get("brand_mentioned")
    article_to_exclude = analysis_data.get("product_article")
    main_problem = analysis_data.get("main_problem")

    recommendation_product = None
    if product_type and product_season:
        search_context = main_problem or request.review_text
        recommendation_product = get_relevant_products(product_type, product_season, search_context, brand_filter, article_to_exclude)

    response_system_prompt = f"{RESPONSE_RULES_PROMPT}\nТвоя задача - написать тело ответа. Не включай приветствие и подпись."
    brand_for_signature = brand_filter if brand_filter and brand_filter.lower() != 'null' else "нашего бренда"
    
    user_prompt_blocks = { "анализ_отзыва": analysis_data }
    
    instruction = f"Напиши основной текст ответа от имени бренда '{brand_for_signature}', основываясь на анализе психотипа и эмоции клиента. "
    if recommendation_product:
        instruction += "Кратко и ненавязчиво порекомендуй найденный товар, объяснив его пользу."
        user_prompt_blocks["рекомендация"] = recommendation_product
    else:
        instruction += "Так как подходящей замены не найдено, рекомендацию давать **ЗАПРЕЩЕНО**."

    user_prompt_blocks["инструкция"] = instruction
    
    final_user_prompt = f"Данные для работы:\n{json.dumps(user_prompt_blocks, ensure_ascii=False, indent=2)}\n\nСгенерируй только основное тело ответа. БЕЗ 'Здравствуйте' и БЕЗ 'С уважением...'."
    body_response_raw = call_llm(response_system_prompt, final_user_prompt, temperature=0.7)
    body_response = clean_response(body_response_raw)

    customer_name = analysis_data.get("customer_name", "Покупатель")
    greeting = f"Здравствуйте, {customer_name}!" if customer_name and customer_name.lower() != 'покупатель' else "Здравствуйте!"
    signature = f"С уважением, команда {brand_for_signature}."
    
    if not body_response:
        if sentiment == 'positive':
            body_response = "Большое спасибо за Ваш отзыв и высокую оценку! Мы очень рады, что Вы остались довольны покупкой."
        else:
            body_response = "Благодарим за обратную связь. Нам искренне жаль, что Вы столкнулись с этой ситуацией. Информация уже передана в отдел качества."

    final_response_text = f"{greeting}\n\n{body_response}\n\n{signature}"
    
    return {"analysis": analysis_data, "response": final_response_text}

@app.get("/")
def read_root():
    return {"status": "WB Pro Assistant Backend is running!"}