import os

from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
from model import neural_net

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = neural_net.Net()
checkpoint = torch.load("model/Stone_model.pth", map_location=device)
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
UPLOAD_FOLDER = 'images'
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


# Route to upload image and predict
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            predicted_class = predict(filepath)
            return render_template("result.html", image=filename, prediction=predicted_class)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
