from flask import Blueprint, request, jsonify
from app.services.model_service import train_model, predict_from_file

predict_bp = Blueprint('predict', __name__)

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