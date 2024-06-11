import os
import glob
from time import time
import cv2
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# locals
from model import neural_net
from utils import image_cropper, grad_cam


CLIENT_ID = '142857969907-2f0geqqrfn8ius4lk9bv2jvlmbgrrki6.apps.googleusercontent.com'
CLIENT_SECRET = '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'
REFRESH_TOKEN = '1//04371Rfl2TnF4CgYIARAAGAQSNwF-L9IrXo-EHSSC3pe_FzJRQgGtP48-jUcJ8TFj6B8wDlz1soBKPMJJjSQshCI8UX2ZzXICaQs'

# Load credentials from refresh token
creds = Credentials(None,
                    refresh_token=REFRESH_TOKEN,
                    client_id=CLIENT_ID,
                    client_secret=CLIENT_SECRET,
                    token_uri='https://oauth2.googleapis.com/token')

# Initialize the Drive API client
drive_service = build('drive', 'v3', credentials=creds)

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


def upload_file(file_path):
    try:
        file_metadata = {'name': 'image.jpg'}
        media = MediaFileUpload(file_path, mimetype='image/jpg')
        response = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print('File ID:', response.get('id'))
    except Exception as error:
        print(f'An error occurred: {error}')


def predict(image_path, w, h):
    img_ = Image.open(image_path)
    img = transform(img_).to(device).unsqueeze(0)
    # # Grad-CAM of the input image
    # grad_cam(model, image_path=image_path, transform=transform, device=device)

    with torch.no_grad():
        logits = model(img)

    probability = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()
    accuracy = probability.cpu().numpy()[0][torch.argmax(probability, dim=1).item()]
    accuracy *= 100
    accuracy = float("{:.2f}".format(accuracy))

    return classes[predicted_class], accuracy


# Route to upload image and predict
@app.route("/", methods=["GET", "POST"])
def upload_image():
    files_ = glob.glob('media/images/*')
    if request.method == "POST":
        for f in files_:
            ti = os.path.getatime(f)
            tc = time()
            if tc - ti >= 300.0:  # after 5 min of upload an image
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
            filepath = os.path.join(UPLOAD_FOLDER + "/images/", filename)
            file.save(filepath)

            # Process the image
            selected_image = cv2.imread(filepath)
            processed_image = image_cropper(selected_image)
            new_file_path = os.path.join(UPLOAD_FOLDER + "/images/", processes_filename)
            cv2.imwrite(new_file_path, processed_image)  # Save the processed image

            # Predict the class of input image
            h, w = processed_image.shape[:2]
            predicted_class, accuracy = predict(new_file_path, w, h)

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
                                   prediction=predicted_class,
                                   accuracy=accuracy)

    return render_template("index.html")


@app.get("/media/<path:path>")
def serve_file(path):
    return send_from_directory(
        directory=app.config["UPLOAD_FOLDER"], path=path
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': "Not file part"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({'error': "Not selected file"}), 400

    if file.filename[-3:] not in ("jpg", "png"):
        return jsonify({'error': "File is not supported image"}), 400

    if file:
        try:
            class_name = request.form['classes']

            # Upload a file
            file_metadata = {'name': f'{class_name}_{file.filename}',
                             'parents': ['1p9Pu90baLPX0qEs2jkeT6YC07iHIQ3JW']
                             }

            buffer = io.BytesIO()
            buffer.name = file.filename
            file.save(buffer)

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

    return jsonify({'error': 'something went wrong'}), 500


if __name__ == "__main__":
    app.run(debug=True)
