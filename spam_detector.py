import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import os

# Load Dataset
csv_file = os.path.join(
    os.path.dirname(__file__),
    "emails.csv"
)

data = pd.read_csv(csv_file)

# Features and Labels
X = data["text"]
y = data["label"]

# Text Vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, pred) * 100, 2),
      "%")

print("Model Trained Successfully!")

# ===========================
# Prediction Loop
# ===========================

while True:

    message = input(
        "\nEnter Email Message (or type exit): "
    )

    if message.lower() == "exit":
        break

    msg_vector = vectorizer.transform(
        [message]
    )

    prediction = model.predict(
        msg_vector
    )[0]

    probability = model.predict_proba(
        msg_vector
    )

    confidence = round(
        max(probability[0]) * 100,
        2
    )

    print("\nResult:", prediction)
    print("Confidence:", confidence, "%")