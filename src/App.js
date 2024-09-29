import React, { useState } from 'react';
import axios from 'axios';

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

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    const handleFileChange = (e) => {
        setFormData(prevState => ({
            ...prevState,
            mriScans: Array.from(e.target.files)  // Handle multiple MRI scan files
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formDataToSend = new FormData();
        formDataToSend.append('age', formData.age);
        formDataToSend.append('gender', formData.gender);
        formDataToSend.append('familyHistory', formData.familyHistory);
        formDataToSend.append('symptoms', formData.symptoms);

        // Append MRI scan files to the FormData object
        formData.mriScans.forEach((file) => {
            formDataToSend.append('mriScans[]', file);  // File array must be appended this way
        });

        try {
            const response = await axios.post('http://localhost:5000/predict', formDataToSend, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setResponse(response.data);
            setErrorMessage(null);  // Clear any error message on success
        } catch (error) {
            console.error('Prediction request failed:', error);
            setErrorMessage(error.response?.data?.error || 'An unexpected error occurred.');
        }
    };

    return (
        <div className="App">
            <h1>NeuroNet Diagnosis Form</h1>

            <form onSubmit={handleSubmit}>
                <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                    placeholder="Age"
                    required
                />
                <input
                    type="text"
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    placeholder="Gender"
                    required
                />
                <input
                    type="text"
                    name="familyHistory"
                    value={formData.familyHistory}
                    onChange={handleChange}
                    placeholder="Family History"
                    required
                />
                <input
                    type="text"
                    name="symptoms"
                    value={formData.symptoms}
                    onChange={handleChange}
                    placeholder="Symptoms"
                    required
                />
                <input
                    type="file"
                    multiple
                    onChange={handleFileChange}
                    required
                />
                <button type="submit">Submit</button>
            </form>

            {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}

            {response && (
                <div>
                    <h2>Prediction Results</h2>
                    <p>Prediction: {response.prediction}</p>
                    <p>Diagnostic Report: {response.diagnostic_report}</p>
                </div>
            )}
        </div>
    );
}

export default App;
