import React, { useState } from 'react';
import axios from 'axios';
import './App.css';  // Import CSS file for styling
import logo from '/Users/bhanuprakash/Desktop/Neuro/NeuroNet/src/logoNeuro.webp';  // Use relative path for portability

function App() {
    const [formData, setFormData] = useState({
        age: '',
        gender: '',
        familyHistory: '',
        symptoms: '',
        mriScans: []
    });
    const [response, setResponse] = useState(null);
    const [errorMessage, setErrorMessage] = useState(null);
    const [loading, setLoading] = useState(false);  // For showing loading state

    // Handle input change for text fields
    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    // Handle file input change for MRI scans
    const handleFileChange = (e) => {
        setFormData(prevState => ({
            ...prevState,
            mriScans: Array.from(e.target.files)  // Allow multiple files
        }));
    };

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);  // Show loading while form is submitted

        const formDataToSend = new FormData();
        formDataToSend.append('age', formData.age);
        formDataToSend.append('gender', formData.gender);
        formDataToSend.append('familyHistory', formData.familyHistory);
        formDataToSend.append('symptoms', formData.symptoms);

        // Append MRI scan files to FormData object
        formData.mriScans.forEach((file) => {
            formDataToSend.append('mriScans[]', file);
        });

        try {
            // Send POST request to backend
            const response = await axios.post('http://localhost:5000/predict', formDataToSend, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setResponse(response.data);  // Set the API response
            setErrorMessage(null);  // Clear any previous errors
        } catch (error) {
            console.error('Prediction request failed:', error);
            setErrorMessage(error.response?.data?.error || 'An unexpected error occurred.');
        } finally {
            setLoading(false);  // Stop loading after request completes
        }
    };

    return (
        <div className="App">
            {/* Display the logo */}
            <img src={logo} className="App-logo" alt="NeuroNet Logo" />
            
            <h1>NeuroNet Diagnosis Form</h1>

            <form className="diagnosis-form" onSubmit={handleSubmit}>
                <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                    placeholder="Enter your Age"
                    min="0"
                    required
                />
                <input
                    type="text"
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    placeholder="Enter your Gender"
                    required
                />
                <input
                    type="text"
                    name="familyHistory"
                    value={formData.familyHistory}
                    onChange={handleChange}
                    placeholder="Family History of Disorders"
                    required
                />
                <input
                    type="text"
                    name="symptoms"
                    value={formData.symptoms}
                    onChange={handleChange}
                    placeholder="Describe Symptoms"
                    required
                />
                <input
                    type="file"
                    multiple
                    onChange={handleFileChange}
                    accept=".jpg,.jpeg,.png,.webp"
                    required
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Submitting...' : 'Submit'}
                </button>
            </form>

            {errorMessage && <p className="error-message">{errorMessage}</p>}

            {response && (
                <div className="results">
                    <h2>Prediction Results</h2>
                    <p><strong>Prediction:</strong> {response.prediction}</p>
                    <p><strong>Diagnostic Report:</strong> {response.diagnostic_report}</p>
                </div>
            )}
        </div>
    );
}

export default App;
