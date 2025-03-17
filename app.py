from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model_and_scaler, predict_stock

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model and scaler
model, scaler = load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')
    n_days = data.get('days', 30)

    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400

    try:
        result = predict_stock(model, scaler, ticker, n_days)
        predictions = [{'date': f'2025-03-{18 + i}', 'price': price} for i, price in enumerate(result)]
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
