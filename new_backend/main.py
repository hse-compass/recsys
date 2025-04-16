from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from algorithm import ProfileVectorizer, StudentRecommender, load_student_data, FEATURE_WEIGHTS
import random

app = FastAPI()
recommender = StudentRecommender()
vectorizer = ProfileVectorizer(FEATURE_WEIGHTS)

class RecommendationRequest(BaseModel):
    available_places: int
    max_capacity: int
    questionnaires_ids: Optional[List[int]] = None

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


        ids = recommender.find_similar_profiles(
            input_vectors=existing_vectors,
            pool_df=df_unissued,
            pool_vectors=vectors_unissued,
            n=n,
            sex=sex
        )
    else:
        random_sex = random.choice(['Male', 'Female'])

        ids = recommender.find_similar_profiles(
            input_vectors=vectors_unissued,
            pool_df=df_unissued,
            pool_vectors=vectors_unissued,
            n=n,
            sex=random_sex
        )

    if not ids:
        raise HTTPException(status_code=404, detail="Не удалось подобрать подходящие анкеты")

    # Обновляем выданные анкеты
    recommender.issued_ids.update(ids)
    recommender.save_issued_ids()

    return ids