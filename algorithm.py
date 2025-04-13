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

# Подгружаем данные из Директуса
def load_student_data():
    students = directus.get_items("questionnaires")
    df = pd.DataFrame(students)
    df = df.dropna(subset=['sex'])
    return df

# Векторизуем признаки
def vectorize_profiles(df):
    df['profile_text'] = df[['hobbies', 'sportstype', 'city', 'roomstyle']].fillna('').apply(lambda row: ' '.join(map(str, row)).lower(), axis=1)

    tfidf = TfidfVectorizer()
    profile_vecs = tfidf.fit_transform(df['profile_text']) * FEATURE_WEIGHTS['profile_text']

    cat_cols = ['roomstyle', 'russianproficiency', 'englishproficiency']
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[cat_cols].fillna(''))

    feature_names = ohe.get_feature_names_out(cat_cols)
    weights_map = {
        'roomstyle': FEATURE_WEIGHTS['roomstyle'],
        'russianproficiency': FEATURE_WEIGHTS['russianproficiency'],
        'englishproficiency': FEATURE_WEIGHTS['englishproficiency']
    }
    cat_weights = np.array([weights_map[col.split('_')[0]] for col in feature_names])
    cat_vecs = cat_encoded.multiply(cat_weights)

    num_cols = ['preferredfloor', 'boardgames', 'doSmoke', 'earlyBird',
                'haschronicdiseases', 'needsbenefitplacement', 'needsspecialconditions']
    num_features = df[num_cols].fillna(0)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_features)
    for i, col in enumerate(num_cols):
        num_scaled[:, i] *= FEATURE_WEIGHTS[col]

    full_vector = np.hstack((profile_vecs.toarray(), cat_vecs.toarray(), num_scaled))
    return df, full_vector

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
            json.dump(list(self.issued_ids), f)

    def recommend(self, n, distance_threshold=None):
        df = load_student_data()
        df, features = vectorize_profiles(df)
        
        mask = ~df['id'].isin(self.issued_ids)
        df_unissued = df[mask].reset_index(drop=True)
        features_unissued = features[mask]

        if len(df_unissued) < n:
            raise ValueError("Недостаточно анкет для подбора")

        result_ids = []

        for sex in ['Male', 'Female']:
            group_df = df_unissued[df_unissued['sex'] == sex].reset_index(drop=True)
            if len(group_df) >= n:
                group_features = features_unissued[df_unissued['sex'] == sex]

                # KNN
                knn = NearestNeighbors(n_neighbors=n, metric='cosine')
                knn.fit(group_features)
                centroid = np.mean(group_features, axis=0).reshape(1, -1)
                distances, indices = knn.kneighbors(centroid)
                selected_indices = []
                for dist, idx in zip(distances[0], indices[0]):
                    if distance_threshold is None or dist < distance_threshold:
                        selected_indices.append(idx)
                    if len(selected_indices) == n:
                        break

                if len(selected_indices) < n:
                    continue 

                selected_ids = group_df.iloc[selected_indices]['id'].tolist()
                result_ids.extend(selected_ids)
                break

        self.issued_ids.update(result_ids)
        self.save_issued_ids()

        return df[df['id'].isin(result_ids)]

# Тест алгоритма
if __name__ == '__main__':
    recommender = StudentRecommender()
    recommendations = recommender.recommend(n=2, distance_threshold=0.5)
    print(recommendations[['firstname', 'lastname', 'sex', 'roomstyle', 'sportstype', 'hobbies', 'city']])
