from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from models import predict_disorder  # Importing the disorder prediction function

# Set up your OpenAI API key
openai.api_key='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'
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the NeuroNet Backend API!"

# Helper function to generate GPT prompt
def generate_prompt(patient_data, mri_analysis):
    prompt = f"""
    Patient details:
    - Age: {patient_data['age']}
    - Gender: {patient_data['gender']}
    - Family History: {patient_data['family_history']}
    - Symptoms: {patient_data['symptoms']}
    - MRI Analysis: {mri_analysis}

    Please provide a medical diagnosis and detailed report based on this data.
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

        # Check if any of the required fields are missing
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

        # Generate GPT prompt with MRI analysis
        print("Generating GPT prompt...")
        prompt = generate_prompt(patient_data, disorder_prediction)
        print(f"Generated prompt: {prompt}")

        # Call GPT API
        print("Sending request to GPT API...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or gpt-4 if you're using GPT-4
            messages=[
                {"role": "system", "content": "You are a medical expert diagnosing brain disorders based on MRI scans."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Adjust as needed
            temperature=0.7
        )
        print(f"Received GPT API response: {response}")

        # Extract GPT's response
        gpt_response = response['choices'][0]['message']['content'].strip()
        print(f"GPT response: {gpt_response}")

        # Return the GPT response to the frontend
        return jsonify({
            "prediction": disorder_prediction,
            "diagnostic_report": gpt_response,
            "visualized_results": "/path/to/visualization"  # Placeholder
        })

    except Exception as e:
        # Handle all other errors
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
