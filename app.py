from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# VAE Model (same as training)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load model
device = torch.device('cpu')  # Use CPU for deployment
model = VAE().to(device)

# Load trained weights
try:
    model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
    model.eval()
    print("Model loaded successfully")
except:
    print("Model file not found. Please upload vae_mnist.pth")

# Store digit prototypes for conditional generation
digit_prototypes = {}

def get_digit_prototype(digit):
    """Get latent representation for a specific digit"""
    if digit not in digit_prototypes:
        # Generate random latent vectors biased toward specific patterns
        # This is a simple approach - in practice you'd use a conditional VAE
        np.random.seed(digit * 42)  # Consistent seed per digit
        base_vector = np.random.randn(20) * 0.5
        
        # Add digit-specific bias
        if digit == 0:
            base_vector[0:5] = [-1, 0, 1, 0, -1]
        elif digit == 1:
            base_vector[0:5] = [0, 1, 1, 0, 0]
        elif digit == 2:
            base_vector[0:5] = [1, -1, 0, 1, -1]
        elif digit == 3:
            base_vector[0:5] = [1, 0, -1, 1, 0]
        elif digit == 4:
            base_vector[0:5] = [-1, 1, 1, -1, 0]
        elif digit == 5:
            base_vector[0:5] = [0, -1, 1, 0, 1]
        elif digit == 6:
            base_vector[0:5] = [-1, -1, 0, 1, 1]
        elif digit == 7:
            base_vector[0:5] = [1, 1, 0, 0, -1]
        elif digit == 8:
            base_vector[0:5] = [0, 0, -1, -1, 1]
        elif digit == 9:
            base_vector[0:5] = [1, 0, 1, -1, -1]
        
        digit_prototypes[digit] = base_vector
    
    return digit_prototypes[digit]

def generate_digit_images(digit, num_images=5):
    """Generate images for a specific digit"""
    images = []
    prototype = get_digit_prototype(digit)
    
    with torch.no_grad():
        for i in range(num_images):
            # Add noise to prototype for variation
            noise = np.random.randn(20) * 0.3
            latent_vector = prototype + noise
            latent_tensor = torch.FloatTensor(latent_vector).unsqueeze(0).to(device)
            
            # Generate image
            generated = model.decode(latent_tensor)
            img_array = generated.view(28, 28).cpu().numpy()
            
            # Convert to PIL Image
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            # Convert to base64 for web display
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
        digit = int(request.json.get('digit', 0))
        if digit < 0 or digit > 9:
            return jsonify({'error': 'Digit must be between 0 and 9'}), 400
        
        images = generate_digit_images(digit)
        return jsonify({'images': images, 'digit': digit})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)