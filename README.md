# Graded Assignment: Deploying a Defect Detection Model with Flask

## Overview
This project demonstrates how to deploy a trained defect detection model as a web service using Flask. The model is a Convolutional Neural Network (CNN) that classifies images into two categories: Defective or Non-Defective. The web service allows users to upload images and receive predictions about the defect status of the uploaded image.

## Key Features:
* A pre-trained defect detection model based on a CNN.
* Flask API to serve the model and handle incoming image requests.
* Image classification: Detect whether an image is "Defective" or "Non-Defective."
* Returns prediction results with confidence scores for each class.
---

## Project Structure

project-directory
│
├── /models                # Directory for saving and loading model weights
│   └── defect_detection_cnn.pth    # Pre-trained model
│
├── /src                   # Source code
│   ├── deploy.py          # Flask app to handle predictions
│   ├── model.py           # CNN model definition
│   └── data_preparation.py  # (Optional) Data preprocessing (if required)
│
└── requirements.txt       # Python dependencies for the project


---

## Prerequisites
- **Python 3.10** or higher
- **CUDA (Optional)**: If you want to leverage GPU acceleration.
- **Flask**: Web framework to deploy the model.
- **Pillow**: Image processing library to handle image inputs.
- **Torch**: Deep learning library for loading the model and making predictions.
- **Torchvision**: For image transformations.
---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd project
```

### 2. Clone the Repository
```
python -m venv .venv
source .venv/bin/activate # For Linux/macOS
.\.venv\Scripts\activate  # For Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
---

## Dataset Generation, Train and Evaluate
### Dataset Generation
```angular2html
python src/generate_dataset.py
```
### Model, Training, Evaluation
```angular2html
python src/train.py
python src/evaluate.py
```

---

# Setting Up the Flask Application
## 1. Load the Pre-Trained Model
The pre-trained model is loaded from the file ../models/defect_detection_cnn.pth. Ensure the model is saved in the correct location or modify the path accordingly in the deploy.py script.
## 2. Starting the Flask Server
To start the Flask server, execute the following command:
```angular2html
python src/deploy.py
```

---

## License
This project is licensed under the MIT License.
This README covers all essential details and is formatted for markdown editors. You can copy and paste it directly.
