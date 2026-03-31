// Configuration for Vercel Frontend to communicate with Render Backend
const API_URL = "https://ai-ml-1rec.onrender.com/api";

// Helper to handle API requests
async function apiFetch(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        return await response.json();
    } catch (err) {
        console.error("API Error:", err);
        return { error: "Network error" };
    }
}
