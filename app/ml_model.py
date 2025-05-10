import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd

model = load_model("app/models/neural_model.h5")

with open("app/models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

symptom_data = pd.read_csv("dataset/train_disease.csv")  
symptom_data["prognosis"] = symptom_data["yellow_crust_ooze"]
symptom_data.drop(columns=["yellow_crust_ooze"], inplace=True)
all_symptoms = list(symptom_data.columns[:-1])

# def get_disease(symptoms: list):
#     encoded_input = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
#     encoded_input = np.array(encoded_input).reshape(1, -1)

#     predictions = model.predict(encoded_input)
#     predicted_disease = le.inverse_transform([np.argmax(predictions)])
#     confidence = float(np.max(predictions))

#     return {"disease": predicted_disease[0], "confidence": confidence}
def get_disease(symptoms: list):
    # Convert symptom list into one-hot vector and predict multiple possible diseases
    encoded_input = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    encoded_input = np.array(encoded_input).reshape(1, -1)

    predictions = model.predict(encoded_input)

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    diseases = le.inverse_transform(top_indices)
    confidences = predictions[0][top_indices]

    return {
        "top_predictions": [
            {"disease": diseases[i], "confidence": float(confidences[i])}
            for i in range(3)
        ]
    }
