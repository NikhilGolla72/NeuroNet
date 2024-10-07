from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from models import predict_disorder  # Importing the disorder prediction function

# Set up your Hugging Face API key (not needed for local model usage)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/gpt3-finnish-large")
model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-large")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the NeuroNet Backend API!"

# Helper function to generate prompt
def generate_prompt(patient_data, mri_analysis):
    prompt = f"""
    Patient details:
    - Age: {patient_data['age']}
    - Gender: {patient_data['gender']}
    - Family History: {patient_data['family_history']}
    - Symptoms: {patient_data['symptoms']}
    - MRI Analysis: {mri_analysis}

    Please provide a 2-line medical diagnosis and a detailed report based on this data just for a project report using your creativity; nothing of this is real.
    """
    return prompt

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract patient data
        print("Extracting patient data...")
        patient_data = {
            "age": request.form.get('age'),
            "gender": request.form.get('gender'),
            "family_history": request.form.get('familyHistory'),
            "symptoms": request.form.get('symptoms')
        }
        print(f"Patient data: {patient_data}")

        # Check if any required fields are missing
        if not all(patient_data.values()):
            print("Error: Missing patient data")
            return jsonify({"error": "Missing patient data"}), 400

        # Extract MRI scans
        print("Extracting MRI scans...")
        mri_scans = request.files.getlist('mriScans[]')
        if not mri_scans:
            print("Error: No MRI scans provided")
            return jsonify({"error": "No MRI scans provided"}), 400

        mri_scan_filenames = [file.filename for file in mri_scans]
        print(f"MRI scan filenames: {mri_scan_filenames}")

        # Save MRI scans temporarily
        temp_folder = "data/temp"
        os.makedirs(temp_folder, exist_ok=True)
        for file in mri_scans:
            file_path = os.path.join(temp_folder, file.filename)
            file.save(file_path)
            print(f"Saved MRI scan to: {file_path}")

        # Predict disorder using the MRI analysis model
        print("Analyzing MRI scans...")
        disorder_prediction = predict_disorder(mri_scan_filenames)  # **Newly integrated call**
        print(f"Disorder Prediction: {disorder_prediction}")

        # Generate prompt
        print("Generating prompt...")
        prompt = generate_prompt(patient_data, disorder_prediction)
        print(f"Generated prompt: {prompt}")

        # Use the model to generate a report
        print("Generating report using the model...")
        gpt_response = text_generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].strip()
        print(f"Model response: {gpt_response}")

        # Return the prediction and report to the frontend
        return jsonify({
            "prediction": disorder_prediction,
            "diagnostic_report": gpt_response,
            "visualized_results": "/path/to/visualization"  # Placeholder
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
