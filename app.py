from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define age groups
AGE_GROUPS = ['teenager', 'child', 'senior', 'adult', 'young']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Resize image to match model's expected sizing
    img = image.resize((200, 200))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Read and preprocess the image
            image = Image.open(file)
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image)
            
            # Get the class with highest probability
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            return jsonify({
                'class': AGE_GROUPS[predicted_class],
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

# For local development
if __name__ == '__main__':
    app.run(debug=True)
