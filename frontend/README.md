# Adaptive RAG Frontend (React + Vite)

A modern React frontend for the Adaptive RAG (Retrieval-Augmented Generation) system built with Vite.

## Project Structure

```
frontend-react/
├── index.html              # Entry HTML file
├── package.json            # Dependencies
├── vite.config.js          # Vite configuration
├── src/
│   ├── main.jsx           # React entry point
│   ├── App.jsx            # Main app component
│   ├── App.module.css     # App styles
│   ├── components/        # Reusable components
│   │   ├── QuestionInput.jsx
│   │   ├── QuestionInput.module.css
│   │   ├── AnswerBox.jsx
│   │   ├── AnswerBox.module.css
│   │   ├── LoadingSpinner.jsx
│   │   ├── LoadingSpinner.module.css
│   │   ├── ErrorBox.jsx
│   │   └── ErrorBox.module.css
│   ├── utils/             # Utilities
│   │   └── api.js         # API client (axios)
│   └── styles/
│       └── global.css     # Global styles
└── .gitignore
```

## Setup

### 1. Install Dependencies

```bash
cd frontend-react
npm install
```

### 2. Configure Backend URL

By default, the frontend connects to `http://localhost:8001`. If your backend runs on a different URL, update it in [src/utils/api.js](src/utils/api.js#L3).

### 3. Start Development Server

```bash
npm run dev
```

The frontend will open automatically at `http://localhost:5173`.

## Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

## Features

- **Component-based architecture** - Modular, reusable components
- **CSS Modules** - Scoped styling to prevent conflicts
- **Fast development** - Vite provides instant HMR (Hot Module Replacement)
- **Axios integration** - Simplified API calls with error handling
- **Responsive design** - Works on all device sizes
- **Modern React patterns** - Hooks, useCallback for optimization

## Component Overview

### QuestionInput
Handles user input for questions with Enter key support.

### AnswerBox
Displays the answer from the RAG system with source type information.

### LoadingSpinner
Shows loading animation while fetching answers.

### ErrorBox
Displays error messages in a user-friendly format.

## Making API Calls

All API calls are centralized in [src/utils/api.js](src/utils/api.js). Use the `askQuestion()` function:

```javascript
import { askQuestion } from './utils/api'

const response = await askQuestion('Your question here')
console.log(response.answer)
```

## Building for Production

```bash
npm run build
```

This creates an optimized `dist/` folder ready for deployment.

## Deployment

### Option 1: Serve with Python
```bash
cd dist
python -m http.server 8080
```

### Option 2: Deploy to Netlify/Vercel
Push the `dist/` folder to your hosting service.

### Option 3: Docker
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY . .
RUN npm install && npm run build

FROM node:18-alpine
RUN npm install -g serve
COPY --from=build /app/dist /app/dist
EXPOSE 3000
CMD ["serve", "-s", "/app/dist", "-l", "3000"]
```

## Environment Variables

Create a `.env.local` file for local development:

```env
VITE_API_URL=http://localhost:8001
```

Then use it in your code:

```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'
```

## Troubleshooting

### Backend connection error
- Ensure the Python backend is running on `http://localhost:8001`
- Check CORS is enabled in the backend (it should be)
- Open browser DevTools (F12) to see network errors

### Port already in use
Change the port in `vite.config.js`:
```javascript
server: {
  port: 5174,  // Change this
}
```

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **CSS Modules** - Scoped styling

## License

MIT
