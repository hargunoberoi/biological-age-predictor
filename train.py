import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import os
from model import AgePredictionNN

def preprocess_data(file_path):
    """
    Load and preprocess the training data
    """
    train_df = pd.read_csv(file_path)
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
    
    # One-hot encode categorical variables
    categorical_columns = ["Chronic Diseases", "Medication Use", "Education Level", "Physical Activity Level"]
    clean_df = pd.get_dummies(clean_df, columns=categorical_columns)
    
    # Split features and target
    x_train = clean_df.drop("Age (years)", axis="columns")
    y_train = clean_df["Age (years)"]
    
    # Convert to numpy arrays
    x_train_numpy = x_train.to_numpy().astype(np.float32)
    y_train_numpy = y_train.to_numpy().astype(np.float32)
    
    # Convert boolean values to 0 and 1
    x_train_numpy[x_train_numpy == True] = 1
    x_train_numpy[x_train_numpy == False] = 0
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train_numpy, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_numpy, dtype=torch.float32)
    
    return x_train_tensor, y_train_tensor

def train_model(x_train, y_train, model_save_path='age_predictor.pth'):
    """
    Train the model using the provided data
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    # Initialize model, loss function, and optimizer
    model = AgePredictionNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training parameters
    num_epoch = 10000
    loss_history = []
    tolerance = 5
    
    # Training loop
    for epoch in range(1, num_epoch + 1):
        model.train()
        predict = model(x_train)
        loss = criterion(predict.squeeze(), y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        loss_history.append(round(current_loss, 2))
        
        # Early stopping check
        if len(loss_history) > tolerance:
            if all(loss_history[-1] == last_loss for last_loss in loss_history[-tolerance-1:-1]):
                print(f"Training completed, Epoch: {epoch}, Final Loss: {current_loss:.4f}")
                break
        
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epoch}], Loss: {current_loss:.4f}')
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss History', color='blue')
    plt.title('Loss History Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_history.png')
    plt.show()
    
    return model, loss_history

def main():
    # Load and preprocess data
    x_train, y_train = preprocess_data('train.csv')
    
    # Train the model
    model, loss_history = train_model(x_train, y_train)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main() 