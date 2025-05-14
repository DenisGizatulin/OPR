import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Параметры
CSV_PATH = "reviews_large.csv"
MODEL_PATH = "saved/model.pt"
VECTORIZER_PATH = "saved/vectorizer.pkl"

# Создание директории, если не существует
os.makedirs("saved", exist_ok=True)

# Загрузка данных
df = pd.read_csv(CSV_PATH)
df["review"] = df["review"].str.lower()

# Векторизация
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["review"]).toarray()
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Модель
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier(X.shape[1]).to(device)

# Обучение
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
batch_size = 64

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        x_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Тестирование
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = model(X_test_tensor).argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {acc:.2f}")

# Сохранение модели и TF-IDF
torch.save(model.state_dict(), MODEL_PATH)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
