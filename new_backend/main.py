from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from algorithm import ProfileVectorizer, StudentRecommender, load_student_data, FEATURE_WEIGHTS

import random
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_cors():
    origins = os.getenv("CORS_ALLOW_ORIGINS", "").split(";")

    # Очищаем и проверяем origins
    cleaned_origins = [origin.strip() for origin in origins if origin.strip()]

    if not cleaned_origins:
        cleaned_origins = [
            "http://localhost",
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000"
        ]
        logger.warning("Using default CORS origins for local development")

    allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

    allow_methods = os.getenv("CORS_ALLOW_METHODS", "GET;POST;PUT;OPTIONS").split(";")
    allow_headers = os.getenv("CORS_ALLOW_HEADERS", "Content-Type;Authorization").split(";")

    logger.info(f"CORS Configuration: "
                f"origins={cleaned_origins}, "
                f"credentials={allow_credentials}, "
                f"methods={allow_methods}, "
                f"headers={allow_headers}")
    return {
        "allow_origins": cleaned_origins,
        "allow_credentials": allow_credentials,
        "allow_methods": [m.strip() for m in allow_methods if m.strip()],
        "allow_headers": [h.strip() for h in allow_headers if h.strip()],
    }

# Применяем CORS middleware
app.add_middleware(
    CORSMiddleware,
    **configure_cors()
)

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()

    # Пропускаем запросы к документации
    if request.url.path in ["/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)

    logger.info(f"Request: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        raise

    process_time = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"Response: {response.status_code} "
                f"(took {process_time:.2f} ms)")

    return response

recommender = StudentRecommender()
vectorizer = ProfileVectorizer(FEATURE_WEIGHTS)

class RecommendationRequest(BaseModel):
    available_places: int
    max_capacity: int
    questionnaires_ids: Optional[List[int]] = None
    gender: Optional[str] = None
    foreigner: Optional[bool] = None

@app.get("/check_questionnaires_data")
def check_student_data():
    all_df = load_student_data()  # Загрузка данных
    student_count = len(all_df)  # Количество анкет
    student_ids = all_df['id'].tolist()  # Список ID анкет

    return {
        "questionnaires_count": student_count,
        "questionnaires_ids": student_ids
    }

@app.post("/recommend", response_model=List[int])
def recommend_profiles(data: RecommendationRequest):
    n = data.available_places

    if not (1 <= n <= 5):
        raise HTTPException(status_code=400, detail="available_places должен быть от 1 до 4")

    # Очищаем кэш рекомендаций перед каждым запросом, чтобы учесть актуальные данные о заселении
    recommender.cache = {}

    # Загружаем и векторизуем все анкеты
    all_df = load_student_data()
    vectorizer.fit(all_df)

    # Убираем уже выданные анкеты
    unissued_mask = ~all_df['id'].isin(recommender.issued_ids)
    df_unissued = all_df[unissued_mask].reset_index(drop=True)

    # Получаем список валидных ID из data
    requested_ids = set(data.questionnaires_ids or [])
    available_ids = set(all_df['id'].values)
    valid_ids = list(requested_ids & available_ids)

    if valid_ids:
        # Удаляем анкеты из пула, чтобы не рекомендовать их снова
        df_unissued = df_unissued[~df_unissued['id'].isin(valid_ids)].reset_index(drop=True)

    # Ограничиваем df_unissued только первокурсниками
    if 'course' in df_unissued.columns:
        df_unissued = df_unissued[df_unissued['course'] == 1].reset_index(drop=True)

    vectors_unissued = vectorizer.transform(df_unissued)

    if len(df_unissued) < n:
        raise HTTPException(status_code=400, detail="Недостаточно анкет для подбора")

    # Если переданы текущие ID анкет в комнате
    if valid_ids:
        df_existing = all_df[all_df['id'].isin(valid_ids)].reset_index(drop=True)
        existing_vectors = vectorizer.transform(df_existing)
        # Проверяем, не пустые ли вектора
        if existing_vectors.shape[0] == 0:
            raise HTTPException(status_code=404, detail="Не удалось векторизовать анкеты по переданным ID")

        sex = df_existing['sex'].mode()[0] if 'sex' in df_existing else None
        is_foreigner = df_existing['is_foreigner'].mode()[0] if 'is_foreigner' in df_existing else None

        # Используем хэш от переданных ID как seed для воспроизводимости результатов
        seed = abs(hash(str(sorted(valid_ids))))
        np.random.seed(seed)

        # Находим похожие профили с детерминированным рандомом
        ids = recommender.find_similar_profiles(
            input_vectors=existing_vectors,
            pool_df=df_unissued,
            pool_vectors=vectors_unissued,
            n=n,
            sex=sex,
            is_foreigner=is_foreigner,
            input_ids=valid_ids
        )
    else:
        # Если нет входных анкет, используем специальные критерии для отбора
        # Вместо случайных выборов из всего пула, отберем на основе gender и foreigner параметров
        filtered_mask = pd.Series([True] * len(df_unissued))

        if data.gender is not None:
            filtered_mask &= df_unissued['sex'] == data.gender

        if data.foreigner is not None:
            filtered_mask &= df_unissued['is_foreigner'] == data.foreigner

        filtered_df = df_unissued[filtered_mask].reset_index(drop=True)

        if len(filtered_df) < n:
            raise HTTPException(status_code=400, detail="Недостаточно подходящих анкет для подбора")

        # Используем хэш от gender и foreigner как seed, чтобы получать одинаковые результаты
        seed = abs(hash(f"{data.gender}_{data.foreigner}")) % 2**32
        np.random.seed(seed)
        selected_indices = np.random.choice(
            len(filtered_df), size=min(n, len(filtered_df)), replace=False
        )

        ids = filtered_df.iloc[selected_indices]['id'].tolist()

    if not ids:
        raise HTTPException(status_code=404, detail="Не удалось подобрать подходящие анкеты")

    # Обновляем выданные анкеты
    recommender.issued_ids.update(ids)
    recommender.save_issued_ids()

    return ids
