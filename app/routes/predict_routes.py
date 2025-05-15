from flask import Blueprint, request, jsonify, render_template
import os
import joblib
import pandas as pd
from app.services.model_service import train_model, predict_from_file

predict_bp = Blueprint('predict', __name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model', 'model.joblib')
FEATURES = ['agresividad', 'aislamiento', 'ansiedad', 'calificaciones']

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400

    file = request.files['file']
    try:
        result = predict_from_file(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@predict_bp.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400

    file = request.files['file']
    try:
        result = train_model(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@predict_bp.route('/predict_view', methods=['GET', 'POST'])
def predict_view():
    predictions = None
    error = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df[FEATURES]  # aseguramos columnas y orden
                
                clf = joblib.load(MODEL_PATH)
                preds = clf.predict(df)
                predictions = preds.tolist()
            except Exception as e:
                error = str(e)
        else:
            error = "Por favor, sube un archivo CSV válido."
    
    return render_template('predict.html', predictions=predictions, error=error)