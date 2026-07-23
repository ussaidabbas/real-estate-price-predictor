-from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'artifacts.pkl')

model = joblib.load(MODEL_PATH)
artifacts = joblib.load(ARTIFACTS_PATH)

LOCATION_FREQ_MAP = artifacts['location_freq_map']
FEATURE_COLUMNS = artifacts['feature_columns']
CITY_CATEGORIES = artifacts['city_categories']
PROPERTY_TYPE_CATEGORIES = artifacts['property_type_categories']

DEFAULT_LOCATION_FREQ = min(LOCATION_FREQ_MAP.values()) if LOCATION_FREQ_MAP else 0.0001


def build_feature_row(payload):
    row = {col: 0 for col in FEATURE_COLUMNS}

    row['bedrooms'] = float(payload.get('bedrooms', 0))
    row['baths'] = float(payload.get('baths', 0))
    row['area'] = float(payload.get('area', 0))

    location = str(payload.get('location', '')).strip().title()
    row['location_freq'] = LOCATION_FREQ_MAP.get(location, DEFAULT_LOCATION_FREQ)

    city = str(payload.get('city', '')).strip().title()
    city_col = f'city_{city}'
    if city_col in row:
        row[city_col] = 1

    property_type = str(payload.get('property_type', '')).strip().title()
    pt_col = f'property_type_{property_type}'
    if pt_col in row:
        row[pt_col] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.route('/')
def home():
    return render_template('index.html', cities=CITY_CATEGORIES, property_types=PROPERTY_TYPE_CATEGORIES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        required = ['city', 'location', 'property_type', 'bedrooms', 'baths', 'area']
        missing = [f for f in required if f not in payload]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        X = build_feature_row(payload)
        log_pred = model.predict(X)[0]
        price_pred = float(np.expm1(log_pred))
        confidence = estimate_confidence(model, X)

        return jsonify({
            'predicted_price_pkr': round(price_pred, 2),
            'predicted_price_formatted': f"PKR {price_pred:,.0f}",
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def estimate_confidence(model, X):
    try:
        if hasattr(model, 'estimators_'):
            preds = np.array([est.predict(X)[0] for est in model.estimators_])
            spread = preds.std()
            confidence = float(max(0.5, min(0.99, 1 - (spread / (abs(preds.mean()) + 1e-6)))))
            return round(confidence, 2)
    except Exception:
        pass
    return 0.85


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)