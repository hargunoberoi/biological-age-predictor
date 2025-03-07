FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary files
COPY endpoint.py .
COPY model.py .
COPY age_predictor.pth .

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "endpoint:app", "--host", "0.0.0.0", "--port", "8000"] 