import os, glob
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
# locals
from model import neural_net
from utils import image_cropper

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


UPLOAD_FOLDER_INPUTS = "uploads"
app.config['UPLOAD_FOLDER_INPUTS'] = UPLOAD_FOLDER_INPUTS


def save_to_local(file):
    if not os.path.exists(app.config['UPLOAD_FOLDER_INPUTS']):
        os.makedirs(app.config['UPLOAD_FOLDER_INPUTS'])
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER_INPUTS'], file.filename)
    file.save(file_path)
    return file_path


def predict(image_path, w, h):
    img_ = Image.open(image_path)
    img = transform(img_).to(device).unsqueeze(0)
    # Grad-CAM of the input image
    gradcam(img, w, h)

    with torch.no_grad():
        logits = model(img)

    probability = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()

    return classes[predicted_class], probability.cpu().numpy()[0][torch.argmax(probability, dim=1).item()]


def gradcam(image,w , h):
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
    img = cv2.resize(img, (w, h))
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
            cv2.imwrite(new_file_path, processed_image) # Save the procesed image

            # Predict the class of input image
            h, w = processed_image.shape[:2]
            predicted_class, accuracy = predict(new_file_path, w, h)

            # Resize to have a better demonstration
            processed_image = cv2.resize(processed_image, (w//2, h//2))
            cv2.imwrite(new_file_path, processed_image) # Save the procesed image

            # Render template
            return render_template("result.html", image=f"images/{processes_filename}", prediction=predicted_class, accuracy=accuracy*100)

    return render_template("index.html")


@app.get("/media/<path:path>")
def serve_file(path):
    return send_from_directory(
        directory=app.config["UPLOAD_FOLDER"], path=path
    )


@app.route('/upload', method=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error':"Not file part"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({'error':"Not selected file"}), 400

    if file.filename[-3:] not in ("jpg", "png"):
        return jsonify({'error':"File is not supported image"}), 400

    if file:
        local_path = save_to_local(file)

        return jsonify({
            'local_path': local_path,
        }), 200

    return jsonify ({'error': 'something went wrong'}), 500


if __name__ == "__main__":
    app.run(debug=True)
