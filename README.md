# Graded Assignment: GAN-Based Image Generation

## Overview
This project demonstrates the training and deployment of a Generative Adversarial Network (GAN) for generating synthetic images. The GAN leverages the FashionMNIST dataset to produce realistic-looking images.

The project is structured into:
- **Training Module**: For model training.
- **Application Module**: For deploying and generating images using the pre-trained model.

---

## Features
- **GAN Architecture**:
  - **Generator**: Synthesizes images.
  - **Discriminator**: Classifies real and fake images.
- **REST API**: Flask-based API for on-demand image generation.
- **Dockerized Deployment**: Simplified container-based execution.

---

## File Structure



---

## Prerequisites
- **Python 3.10** or higher
- **CUDA (Optional)**: If you want to leverage GPU acceleration.

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
## Usage
### Training the GAN
Run the following command to train the generator and discriminator:
```angular2html
python src/train.py
```
Trained models are saved in the models/ directory.
### Generating Images
Run the Flask application to use the pre-trained generator model:
```angular2html
python src/app.py
```
The API will be available at http://127.0.0.1:5001.

## Docker Setup
### Build the Docker Image
```angular2html
docker build -t gan-app .
```
### Run the Docker Container
```angular2html
docker run -p 5001:5001 gan-app
```
The Flask API will be accessible at http://127.0.0.1:5001.

## Output
Generated images will be saved in the output/ directory.

## License
This project is licensed under the MIT License.
This README covers all essential details and is formatted for markdown editors. You can copy and paste it directly.
