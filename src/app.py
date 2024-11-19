import os
from flask import Flask, request, jsonify, send_file
import torch
from generator import Generator
from utils import save_images

app = Flask(__name__)

# Path to the models directory
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

latent_dim = 100
generator = Generator(latent_dim)
generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
generator.eval()

@app.route('/')
def home():
    return "Welcome to the GAN Image Generator API. Use the /generate endpoint to generate images."

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Suppress 404 for favicon requests

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    count = data.get('count', 1)
    noise = torch.randn(count, latent_dim, 1, 1)
    fake_images = generator(noise)
    save_images(fake_images, './output/generated.png')
    return send_file('./output/generated.png', mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=False, port=5001)
