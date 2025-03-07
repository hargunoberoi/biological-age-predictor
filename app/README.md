# Age Predictor Dockerized Application

This is a containerized application consisting of:

- A FastAPI backend
- A Next.js frontend

## Directory Structure

```
app/
├── docker-compose.yml       # Main Docker Compose file
├── README.md                # This file
├── backend/                 # Backend directory
│   ├── Dockerfile           # Backend Dockerfile
│   ├── .dockerignore        # Backend .dockerignore
│   ├── endpoint.py          # FastAPI application
│   ├── model.py             # ML model definition
│   ├── age_predictor.pth    # Trained model
│   └── requirements.txt     # Python dependencies
└── frontend/                # Frontend directory
    ├── Dockerfile           # Frontend Dockerfile
    ├── .dockerignore        # Frontend .dockerignore
    ├── package.json         # Node.js dependencies
    ├── pages/               # Next.js pages
    ├── styles/              # CSS styles
    └── ...                  # Other Next.js files
```

## Local Development

### Prerequisites

- Docker and Docker Compose installed on your system

### Building and Running the Application

1. Navigate to the app directory:

   ```bash
   cd app
   ```

2. Build and start the application using Docker Compose:

   ```bash
   docker compose up -d
   ```

3. To rebuild the images (useful after code changes):
   ```bash
   docker compose up -d --build
   ```

### Accessing the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Stopping the Application

To stop and remove the containers:

```bash
docker compose down
```

## Deploying to DigitalOcean App Platform

1. Push your code to GitHub/GitLab repository

2. Create a new App in DigitalOcean App Platform:

   - Connect to your repository
   - DigitalOcean will detect your docker-compose.yml
   - Configure environment variables if needed
   - Set up domain settings
   - Deploy!

3. App Platform will automatically:
   - Build both containers
   - Set up networking between them
   - Provide HTTPS
   - Handle scaling

## Production Considerations

For production deployment:

1. Update the environment variable in the frontend service:

   ```yaml
   environment:
     - NEXT_PUBLIC_API_URL=${API_URL}
   ```

   Set `API_URL` in your App Platform environment variables to the actual backend URL

2. Consider adding resource limits to control costs

3. Set up custom domains through the App Platform UI
