from flask import Flask, request, render_template, send_file
from model import train_model
from rl_agent import rewards_per_episode
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import base64
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            try:
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                
                preview_html = df.head(10).to_html(classes= 'table table-bordered', index=False)
                
                message = train_model(file_path)
                return render_template("index.html", message=message, preview=preview_html)
            except Exception as e:
                return render_template("index.html", error=str(e))
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')

@app.route('/refuerzo')
def refuerzo():
    try:
        with open('rewards.json', 'r') as f:
            rewards = json.load(f)
    except FileNotFoundError:
        rewards = {
            "rewards_per_episode": [],
            "total_recompensas": 0,
            "episodio_optimo": None
        }

    return render_template('refuerzo.html', rewards=rewards)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)