// src/main.jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { SyncProvider } from './context/SyncContext'   // ✅ add provider

// Tailwind styles (make sure you have index.css imported somewhere)
import './index.css'

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/sw.js')
      .then(() => console.log('✅ Service Worker registered'))
      .catch((err) => console.error('SW registration failed:', err))
  })
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <SyncProvider>
      <App />
    </SyncProvider>
  </React.StrictMode>
)
