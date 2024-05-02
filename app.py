import os, glob
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
from model import neural_net
from utils import image_cropper

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
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Folder to save uploaded images
UPLOAD_FOLDER = 'media'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def predict(image_path):
    img_ = Image.open(image_path)
    img = transform(img_).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)

    probability = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()

    return classes[predicted_class]


files = glob.glob('media/images/*')
# Route to upload image and predict
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        for f in files:
            os.remove(f)
            
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            filename = file.filename
            processes_filename = f"{filename.split('.')[0]}_processed.jpg"
            filepath = os.path.join(UPLOAD_FOLDER + "/images/", filename)
            print(filepath)
            file.save(filepath)
            selected_image = cv2.imread(filepath)
            processed_image = image_cropper(selected_image)
            new_file_path = os.path.join(UPLOAD_FOLDER + "/images/", processes_filename)
            # print(processed_image)
            cv2.imwrite(new_file_path, processed_image)

            predicted_class = predict(new_file_path)
            return render_template("result.html", image=f"images/{processes_filename}", prediction=predicted_class)

    return render_template("index.html")


@app.get("/media/<path:path>")
def serve_file(path):
    return send_from_directory(
        directory=app.config["UPLOAD_FOLDER"], path=path
    )


if __name__ == "__main__":
    app.run(debug=True)
