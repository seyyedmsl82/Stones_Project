import os, glob
import cv2
import pickle
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
import googleapiclient
from googleapiclient.discovery import build
import google
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# locals
from model import neural_net
from utils import image_cropper

# # Google Drive API setup
# CLIENT_SECRET_FILE = "client_secret.json"
# API_NAME = 'drive'
# API_VERSION = 'v3'
# SCOPES = ['https://www.googleapis.com/auth/drive']
# drive = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
#
# creds = None
# if os.path.exists('token.pickle'):
#     with open('token.pickle', 'rb') as token:
#         creds = pickle.load(token)
#
# # If there are no (valid) credentials available, let the user log in.
# if not creds or not creds.valid:
#     if creds and creds.expired and creds.refresh_token:
#         creds.refresh(Request())
#     else:
#         flow = InstalledAppFlow.from_client_secrets_file(
#             CLIENT_SECRET_FILE, SCOPES)
#         creds = flow.run_local_server()
#
#     # Save the credentials for the next run
#     with open('token.pickle', 'wb') as token:
#         pickle.dump(creds, token)
#
# service = build('drive', 'v3', credentials=creds)

CLIENT_ID = '142857969907-2f0geqqrfn8ius4lk9bv2jvlmbgrrki6.apps.googleusercontent.com'
CLIENT_SECRET = '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'
REFRESH_TOKEN = '1//04fecKoDh9AQvCgYIARAAGAQSNwF-L9Iry5hDY_hMzfXaqsb9-PqbMwqCABrAELOi87dkHzBCXAY0Wn2T8j0dlCxYk4Eme9YlWQA'

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
# Unfreeze the desired layer
model.unfreeze_layer('layer4.1.conv2.weight')
model.unfreeze_layer('layer4.1.conv2.bias')

checkpoint = torch.load("stone_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Set the target layer for grad-cam
target_layers = [model.base_model.layer4[-1]]
# cam = GradCAM(model=model, target_layers=target_layers)

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


def save_to_local(file, class_type):
    if not os.path.exists(app.config['UPLOAD_FOLDER_INPUTS']):
        os.makedirs(app.config['UPLOAD_FOLDER_INPUTS'])

    file_path = os.path.join(app.config['UPLOAD_FOLDER_INPUTS'], f"{class_type}_{file.filename}")
    file.save(file_path)
    return file_path


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
    # grad_cam(img, w, h)

    with torch.no_grad():
        logits = model(img)

    probability = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()
    accuracy = probability.cpu().numpy()[0][torch.argmax(probability, dim=1).item()]
    accuracy = float("{:.4f}".format(accuracy))
    print(float(accuracy))

    return classes[predicted_class], accuracy


# def grad_cam(image, w, h):
#     grayscale_cam = cam(input_tensor=image)
#
#     # In this example grayscale_cam has only one image in the batch:
#     grayscale_cam = grayscale_cam[0, :]
#
#     # Overlay the heatmap on the original image
#     plt.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))
#
#     # Remove axis numbers
#     plt.xticks([])
#     plt.yticks([])
#
#     # Overlay the heatmap with transparency
#     plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
#
#     # # Add a colorbar to show the intensity scale of the heatmap
#     # plt.colorbar()
#
#     plt.savefig("heatmap.png")
#     plt.show()
#
#     img = cv2.imread("heatmap.png")
#     img = cv2.resize(img, (int(w * 3 / 2), int(h * 3 / 2)))
#     cv2.imshow("heatmap", img)
#     cv2.waitKey(0)


# Route to upload image and predict
@app.route("/", methods=["GET", "POST"])
def upload_image():
    files_ = glob.glob('media/images/*')
    if request.method == "POST":
        for f in files_:
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
            processed_image = cv2.resize(processed_image, (w // 2, h // 2))
            cv2.imwrite(new_file_path, processed_image)  # Save the processed image

            # Render template
            return render_template("result.html", image=f"images/{processes_filename}", prediction=predicted_class,
                                   accuracy=accuracy * 100)

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
