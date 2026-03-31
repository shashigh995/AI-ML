from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from models.database import db, init_db, Transaction, User
from models.model_engine import FraudDetectionModel
import uuid

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-ai-finance'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
init_db(app)

model_engine = FraudDetectionModel()

# -- Authentication Middlewares --
def is_logged_in():
    return 'user' in session

# -- Routes --

@app.route('/')
def index():
    if is_logged_in():
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username, password=password).first()
    if user:
        session['user'] = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not is_logged_in(): return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/transactions-page')
def transactions_page():
    if not is_logged_in(): return redirect(url_for('index'))
    return render_template('transactions.html')

@app.route('/upload-page')
def upload_page():
    if not is_logged_in(): return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/data/<path:filename>')
def download_file(filename):
    return send_from_directory('data', filename)

@app.route('/api/stats')
def get_stats():
    # Summarized stats for dashboard
    total = Transaction.query.count()
    fraud = Transaction.query.filter_by(status='Fraud').count()
    safe = Transaction.query.filter_by(status='Safe').count()
    
    # Calculate Risk Distribution
    low_risk = Transaction.query.filter(Transaction.risk_score <= 30).count()
    med_risk = Transaction.query.filter((Transaction.risk_score > 30) & (Transaction.risk_score <= 70)).count()
    high_risk = Transaction.query.filter(Transaction.risk_score > 70).count()
    
    return jsonify({
        'total': total,
        'fraud_count': fraud,
        'safe_count': safe,
        'risk_distribution': [low_risk, med_risk, high_risk],
        'recent_trends': [10, 15, 8, 20, 25, 12, 18] # Dummy trend for now
    })

@app.route('/api/transactions')
def get_transactions():
    transactions = Transaction.query.order_by(Transaction.created_at.desc()).all()
    results = []
    for t in transactions:
        results.append({
            'id': t.id,
            'transaction_id': t.transaction_id,
            'amount': t.amount,
            'location': t.location,
            'status': t.status,
            'risk_score': t.risk_score,
            'prediction_confidence': f"{t.prediction_confidence:.2f}%"
        })
    return jsonify(results)

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        df = pd.read_csv(file)
        
        # Save to DB (mocking processing)
        for _, row in df.iterrows():
            # For demonstration, we'll try to predict if the model is loaded/exists
            # Or we'll just store the data
            t_id = row.get('Transaction_ID', str(uuid.uuid4())[:8])
            
            # Predict
            data_point = {
                'Amount': row.get('Amount'),
                'Location': row.get('Location'),
                'Transaction_Type': row.get('Transaction_Type'),
                'Time': row.get('Time'),
                'Device': row.get('Device')
            }
            
            # We will use this to "train" initially with some dummy data if the DB is empty
            new_t = Transaction(
                transaction_id=t_id,
                user_id=row.get('User_ID'),
                amount=row.get('Amount'),
                location=row.get('Location'),
                transaction_type=row.get('Transaction_Type'),
                time=row.get('Time'),
                device=row.get('Device'),
                status=row.get('Status', 'Pending')
            )
            db.session.add(new_t)
        
        db.session.commit()
        return jsonify({'success': True, 'count': len(df)})

@app.route('/api/train', methods=['POST'])
def train_model():
    # Load all current transactions from DB to train
    transactions = Transaction.query.all()
    if not transactions:
        return jsonify({'error': 'No data in database to train on'})
    
    data = []
    for t in transactions:
        data.append({
            'Amount': t.amount,
            'Location': t.location,
            'Transaction_Type': t.transaction_type,
            'Time': t.time,
            'Device': t.device,
            'Status': t.status
        })
    
    df = pd.DataFrame(data)
    # Ensure there are both classes
    if len(df['Status'].unique()) < 2:
        return jsonify({'error': 'Dataset must contain both Fraud and Safe examples for training'})
    
    results = model_engine.train(df)
    
    # After training, update all "Pending" transactions with predictions
    pending = Transaction.query.all()
    for t in pending:
        data_point = {
            'Amount': t.amount, 'Location': t.location, 
            'Transaction_Type': t.transaction_type, 'Time': t.time, 'Device': t.device
        }
        res = model_engine.predict(data_point)
        t.status = 'Fraud' if res['is_fraud'] else 'Safe'
        t.risk_score = res['risk_score']
        t.prediction_confidence = res['confidence']
    
    db.session.commit()
    
    return jsonify({'success': True, 'accuracies': results})

@app.route('/api/explain/<int:id>')
def explain_transaction(id):
    t = Transaction.query.get(id)
    if not t: return jsonify({'error': 'Not found'})
    
    data_point = {
        'Amount': t.amount, 'Location': t.location, 
        'Transaction_Type': t.transaction_type, 'Time': t.time, 'Device': t.device
    }
    
    explanation = model_engine.explain(data_point)
    if not explanation:
        return jsonify({'error': 'Model not trained yet'})
        
    return jsonify({
        'transaction_id': t.transaction_id,
        'amount': t.amount,
        'location': t.location,
        'status': t.status,
        'risk_score': t.risk_score,
        'explanation': explanation
    })

if __name__ == '__main__':
    # Ensure templates and static folders are set correctly if needed
    # (By default Flask looks in current directory)
    app.run(debug=True, port=5000)
