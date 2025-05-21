import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model(file_path):
    #cargar datos
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    #verificar que exista la columna objetivo
    if 'maltrato' not in df.columns:
        raise ValueError("La columna 'maltrato' no se encuentra en el archivo.")
    
    df = pd.get_dummies(df, drop_first=True)
    
    #separar variables
    X = df.drop(columns=['maltrato'])
    y = df['maltrato']
    
    #Preprocesamiento (escalar)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #Entrenar modelo
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    #Guardar modelo y scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'models/logistic_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return "Modelo entrenado y guardado con éxito"