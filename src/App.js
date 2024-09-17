import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [formData, setFormData] = useState({
        age: '',
        gender: '',
        familyHistory: '',
        symptoms: '',
        mriScans: [],
        medicalRecords: []
    });
    const [response, setResponse] = useState(null);

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
            mriScans: Array.from(e.target.files)
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formDataToSend = new FormData();
        formDataToSend.append('age', formData.age);
        formDataToSend.append('gender', formData.gender);
        formDataToSend.append('familyHistory', formData.familyHistory);
        formDataToSend.append('symptoms', formData.symptoms);
        formDataToSend.append('medicalRecords', formData.medicalRecords);

        formData.mriScans.forEach((file, index) => {
            formDataToSend.append(`mriScans[${index}]`, file);
        });

        try {
            const response = await axios.post('http://localhost:5000/predict', formDataToSend, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setResponse(response.data);
        } catch (error) {
            console.error('There was an issue with the prediction request.', error);
        }
    };

    return (
        <div className="App">
            <form onSubmit={handleSubmit}>
                <input type="number" name="age" value={formData.age} onChange={handleChange} placeholder="Age" required />
                <input type="text" name="gender" value={formData.gender} onChange={handleChange} placeholder="Gender" required />
                <input type="text" name="familyHistory" value={formData.familyHistory} onChange={handleChange} placeholder="Family History" required />
                <input type="text" name="symptoms" value={formData.symptoms} onChange={handleChange} placeholder="Symptoms" required />
                <input type="file" multiple onChange={handleFileChange} />
                <textarea name="medicalRecords" value={formData.medicalRecords} onChange={handleChange} placeholder="Medical Records" required></textarea>
                <button type="submit">Submit</button>
            </form>
            {response && (
                <div>
                    <h2>Prediction Results:</h2>
                    <pre>{JSON.stringify(response, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}

export default App;
