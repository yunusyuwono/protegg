import os
import numpy as np
from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from utils import extract_features, generate_yellow_focus_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FOCUS_FOLDER'] = 'static/focus'

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FOCUS_FOLDER'], exist_ok=True)

# Load model
model = load_model("model/bpnn_model.keras")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Fokus kuning
        focus_path = os.path.join(app.config['FOCUS_FOLDER'], filename)
        generate_yellow_focus_image(filepath, focus_path)

        # Prediksi
        features_raw = extract_features(filepath)
        features = np.array([features_raw])
        prediction = model.predict(features)[0][0]
        density = 1.03
        percent = (prediction / (density * 1000)) * 100
        result = explain_feature_influence(model, features)

        return jsonify({
            "original_image": url_for("static", filename="uploads/" + filename),
            "focus_image": url_for("static", filename="focus/" + filename),
            "features": [float(val) for val in features_raw],
            "mgml": float(round(prediction, 2)),
            "percent": float(round(percent, 2)),
            "explanation": result
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500

def explain_feature_influence(model, features):
    base_pred = model.predict(features)[0][0]
    influence = {}
    deltas = [10, 10, 10, 0.05]
    labels = ['H', 'S', 'V', 'Yellow Ratio']
    for i in range(4):
        mod_feat = features.copy()
        mod_feat[0][i] += deltas[i]
        mod_pred = model.predict(mod_feat)[0][0]
        impact = mod_pred - base_pred
        influence[labels[i]] = round(impact, 4)

    sorted_infl = sorted(influence.items(), key=lambda x: abs(x[1]), reverse=True)
    result = "🔍 Pengaruh Fitur Terhadap Prediksi:\n"
    for label, impact in sorted_infl:
        result += f"• {label}: {impact:+.2f} mg/mL\n"
    return result

if __name__ == '__main__':
    app.run(debug=True)
