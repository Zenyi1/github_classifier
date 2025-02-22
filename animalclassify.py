import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO
import numpy as np
import requests
from flask import Flask, request, jsonify
from tf_keras.models import Sequential
import os
from tqdm import tqdm

def preprocess_image(image):
    """Preprocess image for model input"""
    image = np.array(image)
    image_resized = tf.image.resize(image, (224, 224))
    image_resized = tf.cast(image_resized, tf.float32)
    image_resized = (image_resized - 127.5) / 127.5
    return tf.expand_dims(image_resized, 0).numpy()

def load_image_from_url(url):
    """Load and preprocess image from URL"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image

class AnimalClassifier:
    def __init__(self):
        # Load the model and labels during initialization
        self.model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"  # Updated model URL
        self.classification_model = Sequential([
            hub.KerasLayer(self.model_url, trainable=False)  # Added trainable=False
        ])
        
        # Download and load ImageNet labels
        self.download_labels()
        with open("ilsvrc2012_wordnet_lemmas.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]  # Changed rstrip() to strip()
    
    def download_labels(self):
        """Download ImageNet labels if not present"""
        try:
            with open("ilsvrc2012_wordnet_lemmas.txt", "r") as f:
                pass
        except FileNotFoundError:
            url = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
            response = requests.get(url)
            with open("ilsvrc2012_wordnet_lemmas.txt", "w") as f:  # Changed "wb" to "w"
                f.write(response.text)  # Changed content to text
    
    def predict(self, image_url):
        try:
            # Load and preprocess image
            image = load_image_from_url(image_url)
            
            # Make prediction
            predictions = self.classification_model.predict(image)
            predicted_label = self.labels[int(np.argmax(predictions))]
            
            return {"species": predicted_label, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

# Initialize Flask app
app = Flask(__name__)
classifier = AnimalClassifier()

@app.route('/predict', methods=['POST'])
def predict_species():
    if 'image_url' not in request.json:
        return jsonify({"error": "No image URL provided", "status": "error"}), 400
    
    image_url = request.json['image_url']
    result = classifier.predict(image_url)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)