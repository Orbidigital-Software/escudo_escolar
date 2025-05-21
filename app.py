from flask import Flask, request, render_template, redirect, url_for
from model import train_model
import pandas as pd
import os

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

@app.route('/')
def index():
    return "¡Hola, Flask está funcionando!"

if __name__ == '__main__':
    app.run(debug=True)