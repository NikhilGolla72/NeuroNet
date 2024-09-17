import openai
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Set up OpenAI API key (replace with your actual key)
openai.api_key = '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'

from train_model import SimpleNN  # Ensure train_model.py is in the same directory or add it to your PYTHONPATH

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def extract_features_from_mri(images):
    # Define the model architecture before loading the saved state
    model = SimpleNN()
    model.load_state_dict(torch.load('models/trained_model.pt'))  # Load the trained weights
    model.eval()

    features = []
    with torch.no_grad():
        for img in images:
            output = model(img)
            features.append(output.mean().item())  # Example: mean of predictions as feature
    return features

def generate_report_with_gpt(patient_data, features):
    # Prepare the prompt with patient data and MRI features
    prompt = f"""
    Given the following patient data and MRI scan features, generate a prediction and diagnostic report.
    
    Patient Data:
    Age: {patient_data['age']}
    Gender: {patient_data['gender']}
    Family History: {', '.join(patient_data['family_history'])}
    Symptoms: {', '.join(patient_data['symptoms'])}
    
    MRI Scan Features:
    {features}
    
    Generate the following:
    - A prediction of brain disorders such as Alzheimer's and Parkinson's.
    - A detailed diagnostic report including MRI analysis and recommendations.
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    
    result = response.choices[0].text.strip()
    return result

def predict_disorder(data):
    # Preprocess MRI images and extract features
    mri_images = [preprocess_image(os.path.join("data/train", img)) for img in data["MRI_scans"]]
    features = extract_features_from_mri(mri_images)
    
    # Generate the GPT report based on patient data and extracted MRI features
    gpt_response = generate_report_with_gpt(data["patient_data"], features)
    
    # Split the GPT response into prediction and diagnostic report
    split_index = gpt_response.find('Diagnostic Report:')
    prediction = gpt_response[:split_index].strip()  # Extract prediction part
    diagnostic_report = gpt_response[split_index:].strip()  # Extract diagnostic report

    visualized_results = "Link to detailed MRI scan analysis and prediction charts"

    return prediction, diagnostic_report, visualized_results
