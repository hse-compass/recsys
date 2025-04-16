from fastapi import FastAPI, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import logging
import sys

MOCK_RECOMMENDATIONS = [1, 2, 3, 4, 5, 6]

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student Recommendation API",
    description="API для получения рекомендаций по подбору соседей",
    version="1.0.0"
)

class RecommendationRequest(BaseModel):
    available_places: int
    max_capacity: int
    questionnaires_ids: Optional[List[int]] = None

@app.post("/recommend")  # Изменено с GET на POST
async def get_recommendations(request: RecommendationRequest):
    """
    Заглушка для тестирования фронтенда.
    Всегда возвращает статический список рекомендаций.
    """
    try:
        logger.info(f"Получен запрос: available_places={request.available_places}, max_capacity={request.max_capacity}, questionnaires_ids={request.questionnaires_ids}")
        
        # Проверяем корректность входных данных
        if request.available_places < 1 or request.available_places > 5:
            error_msg = "Количество мест должно быть от 1 до 5"
            logger.error(f"Ошибка валидации: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
            
        if request.max_capacity < request.available_places:
            error_msg = "Максимальная вместимость не может быть меньше количества доступных мест"
            logger.error(f"Ошибка валидации: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
            
        return MOCK_RECOMMENDATIONS[:request.available_places]
        
    except Exception as e:
        error_msg = f"Произошла ошибка: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.on_event("startup")
async def startup_event():
    logger.info("API сервер запущен")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API сервер остановлен")