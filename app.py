from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image
import io

app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
num_features = model.last_channel
num_classes = 2  # "indian" and "nonIndian"

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(num_features, num_classes)
)

model.load_state_dict(torch.load('mobilenet_epoch_9.pth', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# Face detection using MTCNN
mtcnn = MTCNN(image_size=224, margin=20)

# Define image transformations
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def detect_face(img):
    """Detect and crop the face from an image using MTCNN."""
    face = mtcnn(img)
    if face is not None:
        return transforms.ToPILImage()(face)
    else:
        return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if 'file' is in the request
        if 'file' not in request.files:
            print("No file found in the request")
            return jsonify({'error': 'No file provided'}), 400
        
        # Read the file
        file = request.files['file']
        print(f"Received file: {file.filename}")

        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Detect face and preprocess the image
        img = detect_face(img)
        img = test_transforms(img).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        # Map prediction to class label
        class_names = ["indian", "nonIndian"]
        result = class_names[predicted.item()]
        confidence_score = confidence.item()

        return jsonify({'result': result, 'confidence': confidence_score}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
