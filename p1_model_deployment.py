#!/usr/bin/python

import pandas as pd
import numpy as np
import joblib
import sys
import os
# carga del modelo
model = joblib.load("model/lgbm_simple_model.pkl")

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesamiento de las características de entrada.
    """
    # 1. Drop columns not needed for prediction
    columns_to_drop = ['track_id', 'track_name', 'explicit', 'time_signature', 'key', 'mode']
    df = df.drop(columns=columns_to_drop)

    # 2. conversion de variableas a tipo categorico
    cat_cols = ['artists', 'album_name', 'track_genre']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df
    

def predict_popularity(raw_features: dict) -> float:
    """
    Predice la popularidad de una canción utilizando el modelo cargado.
    Args:
        raw_features (dict): Input data como diccionario
    Returns:
        float:  popularity score
    """
   
    # Convert input dict to DataFrame
    df = pd.DataFrame([raw_features])

    # Preprocess the input
    processed_df = preprocess_features(df)

    # Predict
    prediction = model.predict(processed_df)

    return float(prediction[0])

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python p1_model_deployment.py '{...json...}'")
        sys.exit(1)

    try:
        # Parse raw JSON input from command line
        raw_input = json.loads(sys.argv[1])
        prediction = predict_popularity(raw_input)
        print("Predicted popularity:", prediction)
    except Exception as e:
        print("Error:", str(e))

        