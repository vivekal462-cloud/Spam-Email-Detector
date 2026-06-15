import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load Dataset

csv_file = os.path.join(
    os.path.dirname(__file__),
    "emails.csv"
)

data = pd.read_csv(csv_file)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

model = MultinomialNB()

model.fit(X, y)

# Prediction Function

def check_email():

    message = text_box.get(
        "1.0",
        tk.END
    ).strip()

    if not message:
        messagebox.showwarning(
            "Warning",
            "Enter Email Text"
        )
        return

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

    result_label.config(
        text=f"Result: {prediction.upper()}\nConfidence: {confidence}%"
    )

# GUI

root = tk.Tk()

root.title("Spam Email Detector")

root.geometry("600x400")

title = tk.Label(
    root,
    text="Spam Email Detector",
    font=("Arial", 18, "bold")
)

title.pack(pady=10)

text_box = tk.Text(
    root,
    height=8,
    width=60
)

text_box.pack(pady=10)

btn = tk.Button(
    root,
    text="Check Email",
    command=check_email,
    width=20
)

btn.pack(pady=10)

result_label = tk.Label(
    root,
    text="Result:",
    font=("Arial", 14)
)

result_label.pack(pady=10)

root.mainloop()