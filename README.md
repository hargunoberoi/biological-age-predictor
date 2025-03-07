# Age Predictor Application

This repository contains a full-stack application for predicting age based on health and demographic features. The application consists of a Next.js frontend and a FastAPI backend.

## Project Structure

```
├── app/                    # Containerized application
│   ├── frontend/           # Next.js frontend application
│   ├── backend/            # FastAPI backend server
│   └── docker-compose.yml  # Docker Compose configuration
├── frontend/               # Development version of Next.js frontend
├── endpoint.py             # Development version of FastAPI backend
├── model.py                # Neural network model definition
├── age_predictor.pth       # Trained model weights
├── train.py                # Model training script
├── inference.py            # Model inference utilities
├── requirements.txt        # Python dependencies
└── demo.py                 # Demo application
```

## Docker Deployment (Recommended)

The easiest way to run the application is using Docker Compose, which will set up both the frontend and backend services in containers.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Running with Docker Compose

1. Navigate to the app directory:

   ```bash
   cd app
   ```

2. Start the application using Docker Compose:

   ```bash
   docker-compose up -d
   ```

3. Access the application:

   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

4. To stop the application:
   ```bash
   docker-compose down
   ```

### Docker Container Details

The application consists of two Docker containers:

1. **Backend (age-predictor-backend)**

   - FastAPI server running on port 8000
   - Provides the prediction API
   - Health check endpoint at /health

2. **Frontend (age-predictor-frontend)**
   - Next.js application running on port 3000
   - User interface for the age prediction service
   - Communicates with the backend API

## Local Development Setup

If you prefer to run the application without Docker for development purposes, follow these instructions:

### Backend (FastAPI)

The backend is built with FastAPI, providing a RESTful API for age prediction.

#### Features

- RESTful API for age prediction
- CORS middleware for cross-origin requests
- Health check endpoint
- Neural network model for age prediction

#### API Endpoints

- `POST /predict`: Predicts age based on input features
- `GET /health`: Health check endpoint

#### Technologies Used

- FastAPI
- PyTorch
- Pandas
- NumPy
- Uvicorn (ASGI server)

#### Running the Backend

There are two ways to run the backend:

1. **Using the root endpoint.py file**:

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Start the server from the project root
   uvicorn endpoint:app --reload
   ```

2. **Using the app/backend/endpoint.py file**:

   ```bash
   # Navigate to the backend directory
   cd app/backend

   # Install dependencies
   pip install -r requirements.txt

   # Start the server
   uvicorn endpoint:app --reload
   ```

The API will be available at http://localhost:8000 and the API documentation at http://localhost:8000/docs.

### Frontend (Next.js)

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
