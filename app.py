import os
from flask import Flask, request, jsonify
import torch
from preprocessing.preprocess import preprocess_text
from model.LSTMModel import LSTMModel
import torch.nn.functional as F

app = Flask(__name__)

# Correct the path to refer to the models directory at the root level
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')


# Load the trained model
model = LSTMModel(input_dim=68811, embedding_dim=100, hidden_dim=256, output_dim=1)
model.load_state_dict(torch.load(os.path.join(model_dir, 'sentiment_lstm.pth')))
model.eval()


# Flask route for prediction
@app.route('/')
def home():
    return "Welcome to the LSTM Predictor. Use the /predict endpoint."

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Suppress 404 for favicon requests

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the review text from the POST request
    review_text = data['review']

    # Preprocess the review text
    tokenized_review = preprocess_text(review_text)
    review_tensor = torch.tensor([tokenized_review])

    # Get the model prediction
    with torch.no_grad():
        output = model(review_tensor)
        prediction = torch.round(torch.sigmoid(output)).item()

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=False)
