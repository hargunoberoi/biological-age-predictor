import gradio as gr
import torch
import numpy as np
import pandas as pd
from model import AgePredictionNN

# Load the pretrained model
def load_model(model_path='age_predictor.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgePredictionNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess the user input to match the expected model input format
def preprocess_input(
    height, weight, systolic_bp, diastolic_bp, cholesterol, bmi, 
    glucose, bone_density, vision, hearing, activity_level,
    chronic_disease, medication_use, cognitive_function, education_level
):
    # Create a dictionary to hold all values
    input_data = {
        'Height (cm)': height,
        'Weight (kg)': weight,
        'Blood Pressure (s/d)': f"{systolic_bp}/{diastolic_bp}",
        'Cholesterol Level (mg/dL)': cholesterol,
        'BMI': bmi,
        'Blood Glucose Level (mg/dL)': glucose,
        'Bone Density (g/cm²)': bone_density,
        'Vision Sharpness': vision,
        'Hearing Ability (dB)': hearing,
        'Physical Activity Level': activity_level,
        'Chronic Diseases': chronic_disease,
        'Medication Use': medication_use,
        'Cognitive Function': cognitive_function,
        'Education Level': education_level
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Process blood pressure
    df[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = df["Blood Pressure (s/d)"].str.split('/', expand=True)
    df['Systolic Blood Pressure'] = df['Systolic Blood Pressure'].astype(int)
    df['Diastolic Blood Pressure'] = df['Diastolic Blood Pressure'].astype(int)
    df.pop("Blood Pressure (s/d)")
    
    # Define all possible categorical values based on the training data
    categorical_columns = {
        "Physical Activity Level": ["Low", "Moderate", "High"],
        "Chronic Diseases": ["None", "Hypertension", "Diabetes", "Heart Disease"],
        "Medication Use": ["None", "Occasional", "Regular"],
        "Education Level": ["None", "High School", "Undergraduate", "Postgraduate"]
    }
    
    # One-hot encode categorical variables
    for cat_col, possible_values in categorical_columns.items():
        value = df[cat_col].iloc[0]
        
        # Create one-hot encoded columns for all possible values
        for possible_value in possible_values:
            col_name = f"{cat_col}_{possible_value}"
            # Set 1 if it matches the selected value, 0 otherwise
            df[col_name] = 1 if value == possible_value else 0
        
        # Remove the original categorical column
        df.pop(cat_col)
    
    # Convert to numpy array and then tensor
    x_numpy = df.to_numpy().astype(np.float32)
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    
    return x_tensor

# Make a prediction using the preprocessed input and the model
def predict_age(
    height, weight, systolic_bp, diastolic_bp, cholesterol, bmi, 
    glucose, bone_density, vision, hearing, activity_level,
    chronic_disease, medication_use, cognitive_function, education_level
):
    # Load the model
    model = load_model()
    device = next(model.parameters()).device
    
    # Preprocess the input
    input_tensor = preprocess_input(
        height, weight, systolic_bp, diastolic_bp, cholesterol, bmi, 
        glucose, bone_density, vision, hearing, activity_level,
        chronic_disease, medication_use, cognitive_function, education_level
    )
    
    # Move tensor to the same device as the model
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Convert to integer
    predicted_age = int(round(prediction.item()))
    
    return f"Predicted Age: {predicted_age} years"

# Create the Gradio interface
def create_interface():
    # Define categorical options based on the full dataset analysis
    activity_options = ["Low", "Moderate", "High"]
    disease_options = ["None", "Hypertension", "Diabetes", "Heart Disease"]
    medication_options = ["None", "Occasional", "Regular"]
    education_options = ["None", "High School", "Undergraduate", "Postgraduate"]
    
    # Create the interface with all required inputs
    interface = gr.Interface(
        fn=predict_age,
        inputs=[
            gr.Number(label="Height (cm)", value=170),
            gr.Number(label="Weight (kg)", value=70),
            gr.Number(label="Systolic Blood Pressure", value=120),
            gr.Number(label="Diastolic Blood Pressure", value=80),
            gr.Number(label="Cholesterol Level (mg/dL)", value=200),
            gr.Number(label="BMI", value=24),
            gr.Number(label="Blood Glucose Level (mg/dL)", value=100),
            gr.Number(label="Bone Density (g/cm²)", value=0.5),
            gr.Number(label="Vision Sharpness", value=0.8),
            gr.Number(label="Hearing Ability (dB)", value=60),
            gr.Dropdown(label="Physical Activity Level", choices=activity_options, value="Moderate"),
            gr.Dropdown(label="Chronic Diseases", choices=disease_options, value="None"),
            gr.Dropdown(label="Medication Use", choices=medication_options, value="None"),
            gr.Number(label="Cognitive Function", value=50),
            gr.Dropdown(label="Education Level", choices=education_options, value="Undergraduate")
        ],
        outputs=gr.Textbox(label="Result"),
        title="Age Predictor",
        description="Enter your health parameters to predict your age based on a neural network model.",
        examples=[
            # Example 1: Healthy young adult
            [175, 70, 120, 80, 180, 22.9, 90, 0.9, 0.9, 70, "High", "None", "None", 85, "Undergraduate"],
            # Example 2: Middle-aged with hypertension
            [168, 82, 145, 95, 220, 29, 110, 0.7, 0.7, 60, "Low", "Hypertension", "Regular", 65, "Undergraduate"],
            # Example 3: Elderly with diabetes
            [162, 68, 135, 85, 250, 26, 150, 0.5, 0.5, 50, "Low", "Diabetes", "Regular", 45, "None"],
            # Example 4: Person with heart disease
            [172, 78, 140, 90, 240, 27, 125, 0.6, 0.6, 55, "Moderate", "Heart Disease", "Regular", 55, "Postgraduate"]
        ]
    )
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=True) 