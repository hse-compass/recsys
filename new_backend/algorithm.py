import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from directus_sdk_py import DirectusClient
from pathlib import Path
from config import DIRECTUS_URL, DIRECTUS_TOKEN


# Подключение к Директусу
directus = DirectusClient(DIRECTUS_URL, token=DIRECTUS_TOKEN)

# Веса признаков
FEATURE_WEIGHTS = {
    'profile_text': 1.0,
    'roomstyle': 0.8,
    'russianproficiency': 0.5,
    'englishproficiency': 0.5,
    'preferredfloor': 0.2,
    'boardgames': 0.4,
    'doSmoke': 0.8,
    'earlyBird': 0.6,
    'haschronicdiseases': 0.6,
    'needsbenefitplacement': 0.2,
    'needsspecialconditions': 0.1
}

def load_student_data():
    # Параметры для пагинации
    page_size = 25  
    page = 1  

    all_students = []

    while True:
        # Загружаем анкету с пагинацией
        students = directus.get_items(
            "questionnaires", 
            params={"page": page, "page_size": page_size}
        )
        
        if not students:
            break  
        
        # Добавляем данные в список
        all_students.extend(students)
        
        page += 1  

    # Преобразуем в DataFrame
    df = pd.DataFrame(all_students)
    df = df.dropna(subset=['sex'])  
    return df

# Векторизация профилей
class ProfileVectorizer:
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights
        self.tfidf = TfidfVectorizer()
        self.ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.scaler = StandardScaler()

    def fit(self, df):
        df = df.copy()
        df['profile_text'] = df[['hobbies', 'sportstype', 'city', 'roomstyle']].fillna('').apply(lambda row: ' '.join(map(str, row)).lower(), axis=1)

        self.tfidf.fit(df['profile_text'])
        self.ohe.fit(df[['roomstyle', 'russianproficiency', 'englishproficiency']].fillna(''))
        self.scaler.fit(df[['preferredfloor', 'boardgames', 'doSmoke', 'earlyBird',
                            'haschronicdiseases', 'needsbenefitplacement', 'needsspecialconditions']].fillna(0))

    def transform(self, df):
        df = df.copy()
        df['profile_text'] = df[['hobbies', 'sportstype', 'city', 'roomstyle']].fillna('').apply(lambda row: ' '.join(map(str, row)).lower(), axis=1)
        profile_vecs = self.tfidf.transform(df['profile_text']) * self.feature_weights['profile_text']

        cat_encoded = self.ohe.transform(df[['roomstyle', 'russianproficiency', 'englishproficiency']].fillna(''))
        feature_names = self.ohe.get_feature_names_out()
        weights_map = {
            'roomstyle': self.feature_weights['roomstyle'],
            'russianproficiency': self.feature_weights['russianproficiency'],
            'englishproficiency': self.feature_weights['englishproficiency']
        }
        cat_weights = np.array([weights_map[col.split('_')[0]] for col in feature_names])
        cat_vecs = cat_encoded.multiply(cat_weights)

        num_cols = ['preferredfloor', 'boardgames', 'doSmoke', 'earlyBird',
                    'haschronicdiseases', 'needsbenefitplacement', 'needsspecialconditions']
        num_features = df[num_cols].fillna(0)
        num_scaled = self.scaler.transform(num_features)
        for i, col in enumerate(num_cols):
            num_scaled[:, i] *= self.feature_weights[col]

        return np.hstack((profile_vecs.toarray(), cat_vecs.toarray(), num_scaled))


# Алгоритм рекомендаций
class StudentRecommender:
    def __init__(self, issued_path='issued_ids.json'):
        self.issued_path = Path(issued_path)
        self.issued_ids = self.load_issued_ids()

    def load_issued_ids(self):
        if self.issued_path.exists():
            with open(self.issued_path, 'r') as f:
                return set(json.load(f))
        return set()

    def save_issued_ids(self):
        with open(self.issued_path, 'w') as f:
            #json.dump(list(self.issued_ids), f)
            json.dump([int(i) for i in self.issued_ids], f)

    def find_similar_profiles(self, input_vectors, pool_df, pool_vectors, n, sex=None, distance_threshold=None):

        # Фильтрация по полу
        if sex is not None:
            mask = pool_df['sex'] == sex
            pool_df = pool_df[mask].reset_index(drop=True)
            pool_vectors = pool_vectors[mask.values]

        if len(pool_df) < n:
            return []

        # Подбор ближайших соседей
        knn = NearestNeighbors(n_neighbors=min(n * 5, len(pool_df)), metric='cosine')
        knn.fit(pool_vectors)

        centroid = np.mean(input_vectors, axis=0).reshape(1, -1)
        distances, indices = knn.kneighbors(centroid)

        selected_ids = []
        for dist, idx in zip(distances[0], indices[0]):
            student_id = pool_df.iloc[idx]['id']
            if student_id not in self.issued_ids and (distance_threshold is None or dist <= distance_threshold):
                selected_ids.append(student_id)
            if len(selected_ids) == n:
                break

        return selected_ids