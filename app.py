"""
This script sets up a web-based user interface for stone classification using a pre-trained neural network model.
It leverages Flask for the web server and Google Drive API for file storage. The main functionalities include:

1. **Import Libraries and Modules**:
    Import necessary libraries such as Flask, PyTorch, OpenCV, and Google Drive API, as well as local modules for the
    model and utilities.

2. **Google Drive API Initialization**:
    Set up credentials and initialize the Google Drive API client for uploading images.

3. **Initialize Flask Application**:
    Set up the Flask web application, configure the upload folder, and create necessary directories.

4. **Load the Pre-trained Model**:
    Load the stone classification model, move it to the appropriate device (GPU or CPU), and set it to evaluation mode.

5. **Define Image Transformations and Class Labels**:
    Set up image transformations for preprocessing and define the class labels for classification.

6. **Helper Functions**:
    Define utility functions for uploading files to Google Drive and predicting the class of input images using the
    pre-trained model.

7. **Flask Routes**:
    Define the main routes for the web application:
    - `/`: Handles the upload and processing of images, performs classification, and renders the results.
    - `/media/<path:path>`: Serves static files from the media directory.
    - `/upload`: Handles file uploads to Google Drive.

8. **Main Execution**:
    Run the Flask application in debug mode.

Dependencies:
    - flask
    - googleapiclient
    - torch
    - torchvision
    - PIL
    - cv2

Usage:
    Run this script to start the Flask web server and access the web interface for stone classification.

Author: SeyyedReza Moslemi
Date: Jun 15, 2024
"""


import os
import glob
from time import time
import cv2
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
# import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload
from google.oauth2.credentials import Credentials

# Local imports
from model import neural_net
from utils import image_cropper

# Google Drive API credentials
CLIENT_ID = '142857969907-2f0geqqrfn8ius4lk9bv2jvlmbgrrki6.apps.googleusercontent.com'
CLIENT_SECRET = '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'
REFRESH_TOKEN = \
    '1//04371Rfl2TnF4CgYIARAAGAQSNwF-L9IrXo-EHSSC3pe_FzJRQgGtP48-jUcJ8TFj6B8wDlz1soBKPMJJjSQshCI8UX2ZzXICaQs'

# Load credentials from refresh token
creds = Credentials(
    None,
    refresh_token=REFRESH_TOKEN,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    token_uri='https://oauth2.googleapis.com/token'
)

# Initialize the Drive API client
drive_service = build('drive', 'v3', credentials=creds)

# Initialize Flask app
app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = neural_net.Net()
checkpoint = torch.load("stone_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Define class labels
classes = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor()
])

# Folder to save uploaded images
UPLOAD_FOLDER = 'media'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def upload_file_to_drive(file_path):
    """
    Uploads a file to Google Drive.

    Args:
        file_path (str): The path to the file to be uploaded.

    Returns:
        None
    """
    try:
        file_metadata = {'name': os.path.basename(file_path)}
        media = MediaFileUpload(file_path, mimetype='image/jpg')
        response = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print('File ID:', response.get('id'))
    except Exception as error:
        print(f'An error occurred: {error}')


def predict(image_path):
    """
    Predicts the class of an image using the pre-trained model.

    Arguments:
        image_path (str): Path to the image file.

    Returns:
        tuple: Predicted class label and accuracy.
    """
    img_ = Image.open(image_path)
    img = transform(img_).to(device).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)

    probability = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()
    accuracy = probability.cpu().numpy()[0][predicted_class] * 100
    accuracy = float(f"{accuracy:.2f}")

    return classes[predicted_class], accuracy


@app.route("/", methods=["GET", "POST"])
def upload_image():
    """
    Handles the upload and processing of an image.

    Returns:
        str: Rendered HTML template with the result.
    """
    files_ = glob.glob('media/images/*')
    if request.method == "POST":
        for f in files_:
            ti = os.path.getatime(f)
            tc = time()
            if tc - ti >= 300.0:  # after 5 minutes of upload an image
                os.remove(f)

        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            filename = file.filename
            processes_filename = f"{filename.split('.')[0]}_processed.jpg"

            # Save the input image in a temporary path
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "images", filename)
            file.save(filepath)

            # Process the image
            selected_image = cv2.imread(filepath)
            processed_image = image_cropper(selected_image)
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "images", processes_filename)
            cv2.imwrite(new_file_path, processed_image)  # Save the processed image

            # Predict the class of input image
            h, w = processed_image.shape[:2]
            predicted_class, accuracy = predict(new_file_path)

            # Resize to have a better demonstration
            h_, w_ = selected_image.shape[:2]
            if h_ > 600:
                w_ //= h_ // 600
                h_ = 600
            selected_image = cv2.resize(selected_image, (w_, h_))
            processed_image = cv2.resize(processed_image, ((w_ * h) // h_, h_))
            cv2.imwrite(new_file_path, processed_image)  # Save the processed image
            cv2.imwrite(filepath, selected_image)

            # Render template
            return render_template("result.html", image1=f"images/{processes_filename}", image2=f"images/{filename}",
                                   prediction=predicted_class, accuracy=accuracy)

    return render_template("index.html")


@app.route("/media/<path:path>")
def serve_file(path):
    """
    Serves files from the media directory.

    Args:
        path (str): Path to the file.

    Returns:
        flask.Response: The requested file.
    """
    return send_from_directory(directory=app.config["UPLOAD_FOLDER"], path=path)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Uploads a file to Google Drive.

    Returns:
        flask.Response: Rendered HTML template or JSON response with an error message.
    """
    if 'file' not in request.files:
        return jsonify({'error': "No file part"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({'error': "No selected file"}), 400

    if file.filename[-3:].lower() not in ("jpg", "png"):
        return jsonify({'error': "File is not a supported image"}), 400

    if file:
        try:
            class_name = request.form['classes']

            # Upload a file
            file_metadata = {
                'name': f'{class_name}_{file.filename}',
                'parents': ['1p9Pu90baLPX0qEs2jkeT6YC07iHIQ3JW']
            }

            buffer = io.BytesIO()
            buffer.name = file.filename
            file.save(buffer)
            buffer.seek(0)

            media_content = MediaIoBaseUpload(buffer, mimetype='image/jpg')

            response = drive_service.files().create(
                body=file_metadata,
                media_body=media_content,
                fields='id'
            ).execute()
            print('File ID:', response.get('id'))

            return render_template("greeting.html")

        except Exception as error:
            print(f'An error occurred: {error}')
            return jsonify({'error': 'Something went wrong'}), 500

    return jsonify({'error': 'Something went wrong'}), 500


if __name__ == "__main__":
    app.run(debug=True)
