from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    
    app.config.from_object('app.config.Config')
    
    CORS(app)
    
    from app.routes.predict_routes import predict_bp
    app.register_blueprint(predict_bp, url_prefix='/api')
    
    return app