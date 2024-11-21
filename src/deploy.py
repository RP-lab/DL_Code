import os
import torch
from PIL import Image  # Ensure Image from Pillow is imported
from torchvision import transforms  # Ensure transforms is imported
import torch.nn.functional as F
from flask import Flask, request, jsonify
from model import CNNModel  # Import your CNN model from the model script

app = Flask(__name__)

# Model configuration
model_path = os.path.join(os.path.dirname(__file__), '../models/defect_detection_cnn.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = CNNModel(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale or single-channel normalization
])

# Define the home route (optional, to handle root URL)
@app.route('/')
def home():
    return "Welcome to the defect detection API! Use POST /predict to upload an image for prediction."

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "File upload failed"}), 400

    try:
        # Load and transform the image
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Map the prediction to class labels
        class_labels = {0: "Non-Defective", 1: "Defective"}
        response = {
            "prediction": class_labels[predicted_class],
            "confidence": probabilities.squeeze().tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
