import os
from flask import Flask, render_template, request, redirect, jsonify
from PIL import Image
from uuid import uuid4
import torch
import torchvision.transforms as transforms

model = torch.load('training_data.pt')

model = model.to('cpu')
model.eval()

classes = ['강아지', '고양이']
app = Flask(__name__)

# Set a directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    # 이미지 크기 조정 및 정규화
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정 (필요에 따라 조정)
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # ImageNet 평균 및 표준 편차
    ])
    return transform(image)

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for handling image upload and processing
@app.route('/', methods=['POST'])
def upload_and_process():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return redirect('/')

        # Get the uploaded image file
        image_file = request.files['image']

        # Check if the file is an image
        if not image_file.filename.endswith('.jpg') and not image_file.filename.endswith('.png'):
            return redirect('/')

        # Generate a unique filename for the image
        filename = str(uuid4()) + '.' + image_file.filename.split('.')[-1]

        # Save the image file to the uploads directory
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the image (resize, convert, etc.) using Pillow
        processed_image = preprocess_image(Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))

        try:
            # 이미지 분류 수행
            predicted_class, probability = predict_class(processed_image)

            return render_template('result.html', prediction=predicted_class, probability=probability, filename=filename)
        except Exception as e:
            return render_template("result.html", msg=str(e))
@app.route('/predict', methods=['POST'])
def predict_class(processed_image):

    prediction = model(processed_image.unsqueeze(0))

    prediction_class_index = prediction.argmax().item()
    prediction_class = classes[prediction_class_index]

    probability = torch.nn.functional.softmax(prediction, dim=1)[0, prediction_class_index].item()

    # JSON 응답 반환
    return prediction_class, probability



if __name__ == '__main__':
    app.run(debug=True)