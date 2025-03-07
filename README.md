# Age Predictor Application

This repository contains a full-stack application for predicting age based on health and demographic features. The application consists of a Next.js frontend and a FastAPI backend.

## Project Structure

```
├── frontend/                # Next.js frontend application
│   ├── pages/               # Next.js pages
│   ├── styles/              # CSS styles
│   ├── package.json         # Frontend dependencies
│   └── ...
├── endpoint.py              # FastAPI backend server
├── model.py                 # Neural network model definition
├── age_predictor.pth        # Trained model weights
├── train.py                 # Model training script
├── inference.py             # Model inference utilities
├── requirements.txt         # Python dependencies
└── demo.py                  # Demo application
```

## Backend (FastAPI)

The backend is built with FastAPI, providing a RESTful API for age prediction.

### Features

- RESTful API for age prediction
- CORS middleware for cross-origin requests
- Health check endpoint
- Neural network model for age prediction

### API Endpoints

- `POST /predict`: Predicts age based on input features
- `GET /health`: Health check endpoint

### Technologies Used

- FastAPI
- PyTorch
- Pandas
- NumPy
- Uvicorn (ASGI server)

### Running the Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn endpoint:app --reload
```

## Frontend (Next.js)

The frontend is built with Next.js, providing a user-friendly interface for interacting with the age prediction model.

### Features

- Form for inputting health and demographic data
- Display of prediction results
- Responsive design with Tailwind CSS

### Technologies Used

- Next.js
- React
- Axios for API requests
- Tailwind CSS for styling

### Running the Frontend

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The application will be available at http://localhost:3000.

## Model Information

The age prediction model is a neural network trained on health and demographic data. The model is defined in `model.py` and the trained weights are stored in `age_predictor.pth`.

## Development

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Setup

1. Clone the repository
2. Set up the backend:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```

## License

[MIT License](LICENSE)
