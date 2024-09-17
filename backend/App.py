from flask import Flask, request, jsonify
from models import predict_disorder  # Ensure 'predict_disorder' is in 'models.py'
import os
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract patient data from request
        patient_data = {
            "age": request.form['age'],
            "gender": request.form['gender'],
            "family_history": request.form['family_history'].split(','),
            "symptoms": request.form['symptoms'].split(',')
        }
        
        # Get the MRI scan files
        mri_scans = request.files.getlist('mri_scans')
        mri_scan_filenames = [file.filename for file in mri_scans]
        
        # Save files temporarily to process them
        for file in mri_scans:
            file.save(os.path.join("data/train", file.filename))

        # Prepare the data for prediction
        data = {
            "patient_data": patient_data,
            "MRI_scans": mri_scan_filenames
        }
        
        # Call the prediction function
        prediction, diagnostic_report, visualized_results = predict_disorder(data)
        
        # Return the results as JSON
        return jsonify({
            "prediction": prediction,
            "diagnostic_report": diagnostic_report,
            "visualized_results": visualized_results
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

