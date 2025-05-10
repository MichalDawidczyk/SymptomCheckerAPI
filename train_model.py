import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

train_data = pd.read_csv("dataset/train_disease.csv")
test_data = pd.read_csv("dataset/test_disease.csv")

# data wrongly assigned, assign predictions from yellow_crust_ooze then remove yellow_crust_ooze
train_data["prognosis"] = train_data["yellow_crust_ooze"]
train_data.drop(columns=["yellow_crust_ooze"], inplace=True)
test_data.drop(columns=["yellow_crust_ooze"], inplace=True)

X_train = train_data.iloc[:, :-1]
y_train = train_data["prognosis"]
X_test = test_data.iloc[:, :-1]
y_test = test_data["prognosis"]

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

with open("app/models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

model.save("app/models/neural_model.h5")
print("Neural Network model trained & saved successfully!")