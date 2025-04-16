from typing import Dict, List

# Моки для разных сценариев
MOCK_RECOMMENDATIONS: List[int] = [1, 2, 3, 4, 5]  


# Моки данных студентов
MOCK_STUDENT_DATA = {
    101: {
        "firstname": "Иван",
        "lastname": "Иванов",
        "sex": "Male",
        "roomstyle": "Спокойный",
        "sportstype": "Футбол",
        "hobbies": "Чтение, музыка",
        "city": "Москва"
    },
    102: {
        "firstname": "Мария",
        "lastname": "Петрова",
        "sex": "Female",
        "roomstyle": "Активный",
        "sportstype": "Теннис",
        "hobbies": "Рисование, танцы",
        "city": "Санкт-Петербург"
    },
    103: {
        "firstname": "Алексей",
        "lastname": "Сидоров",
        "sex": "Male",
        "roomstyle": "Спокойный",
        "sportstype": "Плавание",
        "hobbies": "Программирование, шахматы",
        "city": "Казань"
    },
    104: {
        "firstname": "Елена",
        "lastname": "Козлова",
        "sex": "Female",
        "roomstyle": "Активный",
        "sportstype": "Йога",
        "hobbies": "Фотография, путешествия",
        "city": "Новосибирск"
    }
}

# Статические моки для тестирования фронтенда
STATIC_RECOMMENDATIONS = [
    {
        "firstname": "Иван",
        "lastname": "Иванов",
        "sex": "Male",
        "roomstyle": "Спокойный",
        "sportstype": "Футбол",
        "hobbies": "Чтение, музыка",
        "city": "Москва"
    },
    {
        "firstname": "Мария",
        "lastname": "Петрова",
        "sex": "Female",
        "roomstyle": "Активный",
        "sportstype": "Теннис",
        "hobbies": "Рисование, танцы",
        "city": "Санкт-Петербург"
    },
    {
        "firstname": "Алексей",
        "lastname": "Сидоров",
        "sex": "Male",
        "roomstyle": "Спокойный",
        "sportstype": "Плавание",
        "hobbies": "Программирование, шахматы",
        "city": "Казань"
    },
    {
        "firstname": "Елена",
        "lastname": "Козлова",
        "sex": "Female",
        "roomstyle": "Активный",
        "sportstype": "Йога",
        "hobbies": "Фотография, путешествия",
        "city": "Новосибирск"
    }
] 