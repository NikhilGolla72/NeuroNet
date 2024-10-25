import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from train_model import SimpleNN  # Import the model


# Function to preprocess the MRI scans
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64,64)),  # Adjust to your model's input size
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict disorder using the trained SimpleNN model
def predict_disorder(mri_scan_filenames):
    # **Loading the pre-trained model safely**
    model = SimpleNN()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_model.pt')
        
        # Check if the model exists
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model and map it to CPU for inference
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    predictions = []
    for scan in mri_scan_filenames:
        image_tensor = preprocess_image(os.path.join('data/temp', scan))  # Preprocess image
        with torch.no_grad():
            output = model(image_tensor)  # Forward pass through the model
            _, predicted = torch.max(output.data, 1)  # Get the predicted class
            predictions.append(predicted.item())

    # Disorder labels based on predicted output
    disorder_labels = {
        0: "Mild Dementia", 
        1: "Moderate Dementia", 
        2: "Non Demented", 
        3: "very mild Dementia"
    }

    # **Return the most severe prediction**
    most_severe_prediction = max(predictions)
    disorder_prediction = disorder_labels[most_severe_prediction]
    
    print(f"Predicted Disorder: {disorder_prediction}")

    return disorder_prediction
