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
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseUpload
# locals
from model import neural_net
from utils import image_cropper
from Google import Create_Service


# Google Drive API setup
CLIENT_SECRET_FILE = "client_secret.json"
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']
drive = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server()

    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('drive', 'v3', credentials=creds)

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
cam = GradCAM(model=model, target_layers=target_layers)

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


def predict(image_path, w, h):
    img_ = Image.open(image_path)
    img = transform(img_).to(device).unsqueeze(0)
    # # Grad-CAM of the input image
    # grad_cam(img, w, h)

    with torch.no_grad():
        logits = model(img)

    probability = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()

    return classes[predicted_class], probability.cpu().numpy()[0][torch.argmax(probability, dim=1).item()]


def grad_cam(image, w, h):
    grayscale_cam = cam(input_tensor=image)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # Overlay the heatmap on the original image
    plt.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))

    # Remove axis numbers
    plt.xticks([])
    plt.yticks([])

    # Overlay the heatmap with transparency
    plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)

    # # Add a colorbar to show the intensity scale of the heatmap
    # plt.colorbar()

    plt.savefig("heatmap.png")
    plt.show()

    img = cv2.imread("heatmap.png")
    img = cv2.resize(img, (int(w * 3 / 2), int(h * 3 / 2)))
    cv2.imshow("heatmap", img)
    cv2.waitKey(0)


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
        class_name = request.form['classes']
        # Upload a file
        file_metadata = {'name': f'{class_name}_{file.filename}',
                         'parents': ['1p9Pu90baLPX0qEs2jkeT6YC07iHIQ3JW']
                         }

        buffer = io.BytesIO()
        buffer.name = file.filename
        file.save(buffer)

        media_content = MediaIoBaseUpload(buffer, mimetype='image/jpg')

        service.files().create(
            body=file_metadata,
            media_body=media_content,
        ).execute()

        return render_template("greeting.html")

    return jsonify({'error': 'something went wrong'}), 500


if __name__ == "__main__":
    app.run(debug=True)

# import os
# import flask
# import httplib2
# from apiclient import discovery
# from apiclient.http import MediaIoBaseDownload, MediaFileUpload
# from oauth2client import client
# from oauth2client import tools
# from oauth2client.file import Storage
#
# app = flask.Flask(__name__)
#
#
# @app.route('/')
# def index():
#     credentials = get_credentials()
#     if credentials == False:
#         return flask.redirect(flask.url_for('oauth2callback'))
#     elif credentials.access_token_expired:
#         return flask.redirect(flask.url_for('oauth2callback'))
#     else:
#         print('now calling fetch')
#         all_files = fetch("'root' in parents and mimeType = 'application/vnd.google-apps.folder'",
#                           sort='modifiedTime desc')
#         s = ""
#         for file in all_files:
#             s += "%s, %s<br>" % (file['name'], file['id'])
#         return s
#
#
# @app.route('/oauth2callback')
# def oauth2callback():
#     flow = client.flow_from_clientsecrets('client_id.json',
#                                           scope='https://www.googleapis.com/auth/drive',
#                                           redirect_uri=flask.url_for('oauth2callback',
#                                                                      _external=True))  # access drive api using developer credentials
#     flow.params['include_granted_scopes'] = 'true'
#     if 'code' not in flask.request.args:
#         auth_uri = flow.step1_get_authorize_url()
#         return flask.redirect(auth_uri)
#     else:
#         auth_code = flask.request.args.get('code')
#         credentials = flow.step2_exchange(auth_code)
#         open('credentials.json', 'w').write(credentials.to_json())  # write access token to credentials.json locally
#         return flask.redirect(flask.url_for('index'))
#
#
# def get_credentials():
#     credential_path = 'credentials.json'
#
#     store = Storage(credential_path)
#     credentials = store.get()
#     if not credentials or credentials.invalid:
#         print("Credentials not found.")
#         return False
#     else:
#         print("Credentials fetched successfully.")
#         return credentials
#
#
# def fetch(query, sort='modifiedTime desc'):
#     credentials = get_credentials()
#     http = credentials.authorize(httplib2.Http())
#     service = discovery.build('drive', 'v3', http=http)
#     results = service.files().list(
#         q=query, orderBy=sort, pageSize=10, fields="nextPageToken, files(id, name)").execute()
#     items = results.get('files', [])
#     return items
#
#
# def download_file(file_id, output_file):
#     credentials = get_credentials()
#     http = credentials.authorize(httplib2.Http())
#     service = discovery.build('drive', 'v3', http=http)
#     # file_id = '0BwwA4oUTeiV1UVNwOHItT0xfa2M'
#     request = service.files().export_media(fileId=file_id,
#                                            mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
#     # request = service.files().get_media(fileId=file_id)
#
#     fh = open(output_file, 'wb')  # io.BytesIO()
#     downloader = MediaIoBaseDownload(fh, request)
#     done = False
#     while done is False:
#         status, done = downloader.next_chunk()
#     # print ("Download %d%%." % int(status.progress() * 100))
#     fh.close()
#
#
# # return fh
#
# def update_file(file_id, local_file):
#     credentials = get_credentials()
#     http = credentials.authorize(httplib2.Http())
#     service = discovery.build('drive', 'v3', http=http)
#     # First retrieve the file from the API.
#     file = service.files().get(fileId=file_id).execute()
#     # File's new content.
#     media_body = MediaFileUpload(local_file, resumable=True)
#     # Send the request to the API.
#     updated_file = service.files().update(
#         fileId=file_id,
#         # body=file,
#         # newRevision=True,
#         media_body=media_body).execute()
#
#
# if __name__ == '__main__':
#     if os.path.exists('client_id.json') == False:
#         print('Client secrets file (client_id.json) not found in the app path.')
#         exit()
#     import uuid
#
#     app.secret_key = str(uuid.uuid4())
#     app.run(debug=True)