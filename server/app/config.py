import os
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALG = os.getenv("JWT_ALG")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_URL = os.getenv("DB_URL")
REDIS_URL = os.getenv("REDIS_URL")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")