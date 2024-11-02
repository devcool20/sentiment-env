from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('best_logistic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    # Assume preprocessing and vectorization here
    prediction = model.predict(df)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
