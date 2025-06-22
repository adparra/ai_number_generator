from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=16, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

device = torch.device('cpu')
model = ConditionalVAE().to(device)

try:
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval()
    print("Conditional VAE loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")
    model = None

def to_one_hot(digit, num_classes=10):
    return F.one_hot(torch.tensor([digit]), num_classes=num_classes).float()

def generate_digit_images(digit, num_images=5):
    if model is None:
        raise Exception("Model not loaded")
        
    images = []
    one_hot = to_one_hot(digit).to(device)
    
    with torch.no_grad():
        for i in range(num_images):
            # Sample from standard normal
            z = torch.randn(1, 16).to(device)
            
            # Generate image
            generated = model.decode(z, one_hot)
            img_array = generated.view(28, 28).cpu().numpy()
            
            # Convert to PIL Image
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images.append(img_str)
    
    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if request.is_json:
            digit = int(request.json.get('digit', 0))
        else:
            digit = int(request.form.get('digit', 0))
        
        if digit < 0 or digit > 9:
            return jsonify({'error': 'Digit must be between 0 and 9'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        images = generate_digit_images(digit)
        return jsonify({'images': images, 'digit': digit})
    
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
