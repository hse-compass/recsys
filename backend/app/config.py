import os
from dotenv import load_dotenv

# Загружаем переменные из .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("postgresql://postgres:@localhost:5432/dormitory")