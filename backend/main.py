from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# Инициализация приложения
app = FastAPI()

# Разрешение CORS (для взаимодействия с фронтендом)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Можно указать "http://localhost:3000" для безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели и векторизатора
MODEL_PATH = os.path.join("saved", "model.pkl")
VECTORIZER_PATH = os.path.join("saved", "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# Модель данных
class TextInput(BaseModel):
    text: str

# Эндпоинт для предсказания
@app.post("/predict")
def predict_sentiment(data: TextInput):
    X = vectorizer.transform([data.text])
    prediction = model.predict(X)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}