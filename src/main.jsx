import ReactDOM from 'react-dom/client'; 
import { StrictMode ,React} from 'react';
import App from './App';

// Create a root and render the App
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <StrictMode>
    <App />
  </StrictMode>
);