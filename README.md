# Image Classification Web Application

This is a Flask-based web application that allows users to upload images and get classification results using a pre-trained TensorFlow model.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Place your trained model file (`model.h5`) in the root directory of the project.

## Running the Application

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click on the upload area or drag and drop an image file
2. The application will process the image and display the classification result
3. The result includes the predicted class and confidence score

## Supported Image Formats

- JPG/JPEG
- PNG

## Note

Make sure your `model.h5` file is compatible with the input preprocessing in the application. The current implementation expects:

- Input image size: 224x224 pixels
- Input normalization: pixel values divided by 255.0
