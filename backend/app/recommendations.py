import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_recommendations(students):
    df = pd.DataFrame([s.__dict__ for s in students])

    def age_to_category(age):
        age = int(age)
        if age <= 20:
            return "18_20"
        elif 21 <= age <= 25:
            return "21_25"
        elif 26 <= age <= 30:
            return "26_30"
        else:
            return "30_plus"

    df["Age_Category"] = df["age"].apply(age_to_category)

    df["Profile"] = (
        df["interests"] + " " + df["level_of_education"] + " " +
        df["languages"] + " " + df["gender"] + " " + df["Age_Category"]
    )

    vectorizer = TfidfVectorizer()
    student_vectors = vectorizer.fit_transform(df["Profile"])
    similarity_matrix = cosine_similarity(student_vectors)

    return df, similarity_matrix


def recommend_students(student_id, students, similarity_matrix, df, top_n=3):
    idx = df[df["id"] == student_id].index[0]
    target_gender = df.loc[idx, "gender"]

    same_gender_indices = df[df["gender"] == target_gender].index.tolist()

    sim_scores = [(i, similarity_matrix[idx][i]) for i in same_gender_indices if i != idx]

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

    recommended_students = [students[i[0]] for i in sim_scores]
    return recommended_students
