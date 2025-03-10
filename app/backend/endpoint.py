from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import uvicorn
from model import AgePredictionNN
from typing import List, Dict, Any
import base64
import os
import io
from PIL import Image
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Age Predictor API", 
              description="API for predicting age based on health and demographic features",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
MODEL_PATH = "age_predictor.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = AgePredictionNN()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model = model.to(device)
model.eval()

# Define input request model
class PredictionInput(BaseModel):
    features: Dict[str, Any]

# Define output response model
class PredictionOutput(BaseModel):
    predicted_age: float

# Get OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def preprocess_data(data: Dict[str, Any]) -> torch.Tensor:
    """
    Preprocess incoming data to match the expected model input format,
    following the same approach as in demo.py
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Process blood pressure
        if "Blood Pressure (s/d)" in df.columns:
            df[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = df["Blood Pressure (s/d)"].str.split('/', expand=True)
            df['Systolic Blood Pressure'] = df['Systolic Blood Pressure'].astype(int)
            df['Diastolic Blood Pressure'] = df['Diastolic Blood Pressure'].astype(int)
            df.pop("Blood Pressure (s/d)")
        
        # Define all possible categorical values based on the training data
        # These are the exact same values as in demo.py
        categorical_columns = {
            "Physical Activity Level": ["Low", "Moderate", "High"],
            "Chronic Diseases": ["None", "Hypertension", "Diabetes", "Heart Disease"],
            "Medication Use": ["None", "Occasional", "Regular"],
            "Education Level": ["None", "High School", "Undergraduate", "Postgraduate"]
        }
        
        # One-hot encode categorical variables
        for cat_col, possible_values in categorical_columns.items():
            if cat_col in df.columns:
                value = df[cat_col].iloc[0]
                
                # Create one-hot encoded columns for all possible values
                for possible_value in possible_values:
                    col_name = f"{cat_col}_{possible_value}"
                    # Set 1 if it matches the selected value, 0 otherwise
                    df[col_name] = 1 if value == possible_value else 0
                
                # Remove the original categorical column
                df.pop(cat_col)
            else:
                # If categorical column is missing, add dummy columns with 0s
                for possible_value in possible_values:
                    col_name = f"{cat_col}_{possible_value}"
                    df[col_name] = 0
                print(f"Warning: Missing categorical feature '{cat_col}'. Adding zero-filled dummy columns.")
        
        # Check if we have sufficient numeric features
        required_numeric = [
            'Height (cm)', 'Weight (kg)', 'Cholesterol Level (mg/dL)', 
            'BMI', 'Blood Glucose Level (mg/dL)', 'Bone Density (g/cm²)', 
            'Vision Sharpness', 'Hearing Ability (dB)', 'Cognitive Function',
            'Systolic Blood Pressure', 'Diastolic Blood Pressure'
        ]
        
        for col in required_numeric:
            if col not in df.columns:
                print(f"Warning: Missing numeric feature '{col}'. Adding with default value 0.")
                df[col] = 0
        
        # Convert to numpy array and then tensor
        x_numpy = df.to_numpy().astype(np.float32)
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        
        print(f"Final tensor shape: {x_tensor.shape}")
        
        return x_tensor
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise Exception(f"Failed to preprocess data: {str(e)}")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Preprocess the input data
        input_tensor = preprocess_data(input_data.features)
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Convert prediction to float
        predicted_age = float(prediction.cpu().numpy().flatten()[0])
        
        return {"predicted_age": predicted_age}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "age_predictor.pth"}

def encode_image(image_bytes):
    """Encode image bytes to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')

@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """
    Analyze image using OpenAI O1 model to extract health metrics.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured on server")
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the uploaded image
        contents = await image.read()
        
        # Convert to base64
        base64_image = encode_image(contents)
        
        # Prepare the prompt for OpenAI
        prompt = """
        This image may contain health-related information. 
        Please extract the following health metrics if visible in the image:
        - Height (in cm)
        - Weight (in kg)
        - Blood Pressure (systolic/diastolic)
        - Cholesterol Level (in mg/dL)
        - BMI
        - Blood Glucose Level (in mg/dL)
        - Bone Density (in g/cm²)
        - Vision Sharpness (0.1-1.0)
        - Hearing Ability (in dB)
        - Activity Level (sedentary, light, moderate, active, very active)
        - Sleep Duration (in hours)
        - Smoking Status (yes, no)
        - Alcohol Consumption (none, light, moderate, heavy)
        - Medication Count (number)
        - Heart Rate (BPM)
        
        Return only the values you can confidently extract in a JSON format with the following keys:
        height, weight, bloodPressure, cholesterol, bmi, glucose, boneDensity, vision, hearing, activity, sleepDuration, smokingStatus, alcoholConsumption, medicationCount, heartRate
        
        Include only the fields that you can extract from the image. If you cannot determine a value, do not include that field in the response.
        """
        
        # Set up the messages for the API call
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ]
        
        # Make API call using the OpenAI client
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            content = response.choices[0].message.content
            
            # Parse JSON from the response
            try:
                # Try to extract JSON if it's wrapped in markdown code blocks
                if "```json" in content and "```" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    extracted_data = json.loads(json_str)
                else:
                    # Otherwise try to parse the whole response as JSON
                    extracted_data = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, attempt to extract key-value pairs from text
                extracted_data = {}
                # Simple extraction using common patterns
                if "height" in content.lower():
                    # Attempt to extract height value
                    pass
                # Add similar extraction for other fields if needed
            
            return extracted_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run("endpoint:app", host="127.0.0.1", port=8000, reload=True) 