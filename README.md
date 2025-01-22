# Manufacturing Predictive API

A Flask-based API to predict downtime in manufacturing based on temperature and runtime.

---

## Features
- Upload a dataset for training.
- Train a machine learning model.
- Predict downtime using the trained model.

---

## Setup Instructions
1. Clone the repository
   ```bash
   git clone https://github.com/Ravi9550/Manufacturing-Predictive-API.git
   
2. Go to Folder
   ```bash
   cd ManufacturingPredictiveAPI
   
3. Install required libraries
   ```bash
   pip install -r requirements.txt
   
4. Run the Flask application
   ```bash
   python run.py
5. Use the following API endpoints:

  - Upload Dataset: /upload
    POST with form-data including file.
  - Train Model: /train
    POST to train the ML model.
  - Predict Downtime: /predict
    POST with JSON body
    ```bash
    {
    "Temperature": 80,
    "Run_Time": 120
     }

## Example cURL Commands

1. Upload Dataset
   ```bash
   curl -X POST -F "file=@data/sample_dataset.csv" http://127.0.0.1:5000/upload
2.Train Model
   ```bash
   curl -X POST http://127.0.0.1:5000/train
   ```
3.Predict Downtime
  ```bash
curl -X POST -H "Content-Type: application/json" -d "{\"Temperature\": 80, \"Run_Time\": 120}" http://127.0.0.1:5000/predict

