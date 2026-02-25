import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
});

export const analyzeVideo = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/predict/video', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export const analyzeImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/predict/image', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export const checkHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

export default api;
