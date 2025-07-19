from flask import Flask, request, jsonify
from quant_engine import run_quant_for_ticker
import os
from dotenv import load_dotenv
from flask import render_template
from flask_cors import CORS
CORS(app)  # Esto permite que el frontend llame al backend

# @app.route('/')
# def index():
#     return render_template('index.html')

load_dotenv("tiingo.env")

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    ticker = data.get('ticker')
    start_date = data.get('start_date', '2020-01-01')
    end_date = data.get('end_date', None)
    
    if not ticker:
        return jsonify({'status': 'error', 'message': 'Ticker is required'}), 400
    
    try:
        results = run_quant_for_ticker(ticker, start_date, end_date)
        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)