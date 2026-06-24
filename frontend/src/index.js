import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import axios from 'axios';

// Use relative URLs so all API calls go through the CRA dev-server proxy
// (see src/setupProxy.js), which forwards to the backend and injects the
// X-API-Key header. Keeping requests same-origin also avoids CORS.
axios.defaults.baseURL = process.env.REACT_APP_API_URL || '';

// Attach the API key (if configured) to every request so the backend's
// X-API-Key auth middleware accepts calls from the SPA.
if (process.env.REACT_APP_API_KEY) {
  axios.defaults.headers.common['X-API-Key'] = process.env.REACT_APP_API_KEY;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

reportWebVitals();