import pandas as pd
import numpy as np
import torch
import os
from tabulate import tabulate
from model import AgePredictionNN

def data_cleaner(file_name):
    """
    Cleans and preprocesses data for inference
    """
    train_df = pd.read_csv(file_name)
    clean_df = train_df.drop([
        "Gender", "Smoking Status", "Alcohol Consumption", "Diet", 
        "Family History", "Mental Health Status", "Sleep Patterns", 
        "Stress Levels", "Pollution Exposure", "Sun Exposure", "Income Level"
    ], axis='columns')
    
    # Handle missing values
    columns_with_empty_cells = clean_df.columns[clean_df.isna().any()].tolist()
    i = 1
    for col in columns_with_empty_cells:
        clean_df[col] = clean_df[col].fillna(f"None{i}")
        i += 1
    
    # Process blood pressure
    clean_df[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = clean_df["Blood Pressure (s/d)"].str.split('/', expand=True)
    clean_df['Systolic Blood Pressure'] = clean_df['Systolic Blood Pressure'].astype(int)
    clean_df['Diastolic Blood Pressure'] = clean_df['Diastolic Blood Pressure'].astype(int)
    clean_df.pop("Blood Pressure (s/d)")
    
    # Handle categorical variables manually
    categorical_columns = ["Chronic Diseases", "Medication Use", "Education Level", "Physical Activity Level"]
    
    for cat_col in categorical_columns:
        unique_values = clean_df[cat_col].unique()
        for value in unique_values:
            if pd.notna(value):  # Skip NaN values
                col_name = f"{cat_col}_{value}"
                clean_df[col_name] = (clean_df[cat_col] == value).astype(int)
        clean_df.pop(cat_col)
    
    # Convert to numpy array and then tensor
    x_numpy = clean_df.to_numpy().astype(np.float32)
    x_numpy[x_numpy == True] = 1
    x_numpy[x_numpy == False] = 0
    
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    
    return x_tensor

def predict_ages(model_path, data_tensor, limit=None):
    """
    Use the trained model to predict ages
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = AgePredictionNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    
    # Limit samples if specified
    if limit and limit < len(data_tensor):
        data_tensor = data_tensor[:limit]
    
    # Move data to device
    data_tensor = data_tensor.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(data_tensor)
    
    # Convert predictions to integers
    predicted_ages = predictions.cpu().numpy().flatten()
    
    return predicted_ages

def compare_predictions(model_path, sample_size=20):
    """
    Compare predictions with true ages for a sample of the training data
    """
    # Load the training data to get true ages
    train_data = pd.read_csv('train.csv')
    train_subset = train_data.iloc[:sample_size]
    true_ages = train_subset['Age (years)'].tolist()
    
    # Create a copy of the data without the 'Age (years)' column for data_cleaner
    train_data_copy = train_data.copy()
    temp_file = 'temp_train_without_age.csv'
    if 'Age (years)' in train_data_copy.columns:
        train_data_copy = train_data_copy.drop('Age (years)', axis=1)
    train_data_copy.to_csv(temp_file, index=False)
    
    # Process the data
    train_tensor = data_cleaner(temp_file)
    train_tensor = train_tensor[:sample_size]  # Take only the requested samples
    
    # Get predictions
    predicted_ages = predict_ages(model_path, train_tensor)
    predicted_ages_int = np.round(predicted_ages).astype(int).tolist()
    
    # Create comparison table
    comparison_table = [[i + 1, true, pred] for i, (true, pred) in enumerate(zip(true_ages, predicted_ages_int))]
    print(tabulate(comparison_table, headers=["Row", "True Age", "Predicted Age"], 
                   tablefmt="pretty", stralign="center"))
    
    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return true_ages, predicted_ages

def main():
    model_path = 'age_predictor.pth'
    
    # Make sure the model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Compare predictions with true values
    print("Comparing predictions with true ages for training samples:")
    true_ages, predicted_ages = compare_predictions(model_path)
    
    # Make predictions on test data
    print("\nMaking predictions on test data:")
    test_tensor = data_cleaner('test.csv')
    predicted_test_ages = predict_ages(model_path, test_tensor, limit=50)
    
    # Print first 10 test predictions
    test_predictions = np.round(predicted_test_ages[:10]).astype(int).tolist()
    for i, age in enumerate(test_predictions):
        print(f"Sample {i+1}: Predicted Age = {age}")

if __name__ == "__main__":
    main() 