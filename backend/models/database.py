from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.String(50))
    amount = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100))
    transaction_type = db.Column(db.String(50))
    time = db.Column(db.String(50))
    device = db.Column(db.String(100))
    status = db.Column(db.String(20), default='Pending') # Fraud/Safe
    risk_score = db.Column(db.Integer, default=0)
    prediction_confidence = db.Column(db.Float, default=0.0)
    explanation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        # Seed an admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', password='password123') # Basic for local setup
            db.session.add(admin)
            db.session.commit()
