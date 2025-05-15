import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

MODEL_PATH = 'model/classifier.pkl'

def train_model(file):
    df = pd.read_csv(file)

    # columna'target' (obligatorio)
    if 'target' not in df.columns:
        raise Exception("El archivo debe contener una columna llamada 'target'")

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    return {
        "message": "Modelo entrenado exitosamente",
        "metrics": report
    }

def predict_from_file(file):
    if not os.path.exists(MODEL_PATH):
        raise Exception("El modelo no ha sido entrenado aún. Entrénalo primero con /train.")
    
    df = pd.read_csv(file)
    
    # Eliminar columnas sin nombre que puedan haber aparecido
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    clf = joblib.load(MODEL_PATH)
    predictions = clf.predict(df)
    
    df['prediction'] = predictions
    return {
        "message": "Predicción realizada con éxito",
        "predictions": df['prediction'].tolist()
    }
