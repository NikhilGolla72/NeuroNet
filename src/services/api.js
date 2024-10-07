import axios from 'axios';

// Check if the environment is production (GitHub Pages) or development (localhost)
const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://your-live-api.com/predict'  // Replace with your live API endpoint for production
  : 'http://localhost:5000/predict';      // Use localhost for development

export const postPatientData = async (data) => {
  try {
    const response = await axios.post(API_URL, data);
    return response.data;
  } catch (error) {
    console.error('Error posting patient data:', error);
    throw error;
  }
};
