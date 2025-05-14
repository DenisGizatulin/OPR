from fastapi import FastAPI
from pydantic import BaseModel
import torch, joblib
from model.sentiment_model import SentimentModel
from utils.preprocessing import clean_text

class Review(BaseModel):
    text: str

app = FastAPI()

vectorizer = joblib.load('saved/vectorizer.pkl')
model = SentimentModel(input_dim=5000)
model.load_state_dict(torch.load('saved/model.pt'))
model.eval()

@app.post("/predict")
def predict_sentiment(review: Review):
    clean = clean_text(review.text)
    vec = vectorizer.transform([clean]).toarray()
    with torch.no_grad():
        pred = model(torch.tensor(vec).float())
    return {"prediction": "positive" if pred.item() > 0.5 else "negative"}