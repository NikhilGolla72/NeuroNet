import axios from 'axios';

const API_URL = 'http://localhost:5000/predict';

export const postPatientData = async (data) => {
  return await axios.post(API_URL, data);
};
