#!/usr/bin/python

import pandas as pd
import numpy as np
import sys
import os
import cloudpickle

# Cargar el modelo multiclase
with open("model/lgbm_simple_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Nombres de clases sin prefijo
genre_columns = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
    'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
    'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western'
]

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa los datos: asegura tipos correctos y construye la columna 'text'.
    """
    required_columns = ['year', 'title', 'plot']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    df['year'] = pd.to_numeric(df['year'], errors='raise')
    df['text'] = df['title'] + ' ' + df['plot']
    return df[['year', 'text']]

def predict_genre(raw_features: dict) -> dict:
    """
    Retorna las probabilidades por género, ordenadas de mayor a menor.
    """
    df = pd.DataFrame([raw_features])
    processed_df = preprocess_features(df)

    # Obtener las probabilidades
    probs = model.predict_proba(processed_df)[0]  # Una sola fila

    # Mapear sin el prefijo y ordenar
    genre_probs = dict(sorted(zip(genre_columns, probs), key=lambda x: x[1], reverse=True))

    return genre_probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python p2_app.py '{\"year\": 2010, \"title\": \"Inception\", \"plot\": \"...\"}'")
        sys.exit(1)

    try:
        raw_input = json.loads(sys.argv[1])
        genre_probabilities = predict_genre(raw_input)
        print("Probabilidades por género (ordenadas):")
        for genre, prob in genre_probabilities.items():
            print(f"{genre}: {prob:.4f}")
    except Exception as e:
        print("Error:", str(e))

