import axios from 'axios'

const API_BASE_URL = 'http://localhost:8001'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const askQuestion = async (question) => {
  try {
    const response = await apiClient.post('/ask', { question })
    return response.data
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error')
    } else if (error.request) {
      throw new Error('No response from server. Make sure the backend is running on http://localhost:8001')
    } else {
      throw new Error(error.message)
    }
  }
}

export default apiClient
