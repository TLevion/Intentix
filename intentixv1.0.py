# ==============================
# INTENT DETECTION USING ONLINE DATASET
# ==============================

# Install libraries
!pip install pandas scikit-learn datasets

# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import pickle

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 2. LOAD DATASET FROM HUGGINGFACE
# ==============================
dataset = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1")

# Convert to pandas
df = pd.DataFrame(dataset['train'])

# Rename columns if needed
df.columns = ["text", "intent"]

print("Dataset sample:")
print(df.head())

# ==============================
# 3. ENCODE LABELS
# ==============================
le = LabelEncoder()
df["intent_encoded"] = le.fit_transform(df["intent"])

# ==============================
# 4. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["intent_encoded"], test_size=0.2, random_state=42
)

# ==============================
# 5. TF-IDF VECTORIZATION
# ==============================
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 6. TRAIN MODEL
# ==============================
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ==============================
# 7. EVALUATION
# ==============================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 8. SAVE MODEL (OPTIONAL)
# ==============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\nModel saved successfully!")

# ==============================
# 9. PREDICTION FUNCTION
# ==============================
def predict_intent(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)
    intent = le.inverse_transform(pred)
    return intent[0]

# ==============================
# 10. INTERACTIVE TESTING
# ==============================
print("\n=== Intent Detection System ===")

while True:
    user_input = input("\nEnter command (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    result = predict_intent(user_input)
    print("Predicted Intent:", result)