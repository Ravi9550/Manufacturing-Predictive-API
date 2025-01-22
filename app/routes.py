from flask import Blueprint, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
main = Blueprint('main', __name__)

TEMP_DIR = './temp'
os.makedirs(TEMP_DIR, exist_ok=True)

@main.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed"}), 400
    
    filepath = os.path.join(TEMP_DIR, file.filename)
    file.save(filepath)
    
    try:
        data = pd.read_csv(filepath)
        if 'Machine_ID' not in data.columns or 'Temperature' not in data.columns or 'Run_Time' not in data.columns:
            return jsonify({"error": "Invalid file format"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 500
    
    return jsonify({"message": "File uploaded and validated successfully"}), 200

trained_model = None
@main.route('/train', methods=['POST'])
def train_model():
    global trained_model

    dataset_path = os.path.join(TEMP_DIR, 'sample_dataset.csv')

    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset not found. Please upload a dataset first."}), 400

    try:
        data = pd.read_csv(dataset_path)

        required_columns = ['Temperature', 'Run_Time', 'Downtime_Flag']
        if not all(col in data.columns for col in required_columns):
            return jsonify({"error": f"Dataset must contain the following columns: {required_columns}"}), 400

        X = data[['Temperature', 'Run_Time']]
        y = data['Downtime_Flag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        trained_model = model

        return jsonify({
            "message": "Model trained successfully.",
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4)
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred while training the model: {str(e)}"}), 500

@main.route('/predict', methods=['POST'])
def predict():
    global trained_model

    if trained_model is None:
        return jsonify({"error": "Model not trained yet. Please train the model first."}), 400

    try:
        input_data = request.get_json()

        if 'Temperature' not in input_data or 'Run_Time' not in input_data:
            return jsonify({"error": "Missing required fields. Please provide 'Temperature' and 'Run_Time'."}), 400

        temperature = input_data['Temperature']
        run_time = input_data['Run_Time']

        prediction = trained_model.predict([[temperature, run_time]])
        prediction_prob = trained_model.predict_proba([[temperature, run_time]])[0][1]  # Probability of Downtime: Yes

        return jsonify({
            "Downtime": "Yes" if prediction[0] == 1 else "No",
            "Confidence": round(prediction_prob, 4)
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred while making the prediction: {str(e)}"}), 500
