import torch
from torch import nn
import torch.nn.functional as F

class AgePredictionNN(nn.Module):
    def __init__(self):
        super(AgePredictionNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(25, 64)
        self.ln1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 16)
        self.ln2 = nn.LayerNorm(16)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(16, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Kaiming initialization for layers with GELU
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        # Xavier initialization for output layer
        nn.init.xavier_normal_(self.fc3.weight)
        
        # Initialize biases to small values
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        # First layer with LayerNorm and Dropout
        identity = x  # For potential skip connection in future layers
        
        outHiddenLayer1 = self.fc1(x)
        outHiddenLayer1 = self.ln1(outHiddenLayer1)
        # Using GELU as a more modern activation function
        activate1 = F.gelu(outHiddenLayer1)
        activate1 = self.dropout1(activate1)
        
        # Second layer with LayerNorm, Dropout and skip connection
        outHiddenLayer2 = self.fc2(activate1)
        outHiddenLayer2 = self.ln2(outHiddenLayer2)
        activate2 = F.gelu(outHiddenLayer2)
        activate2 = self.dropout2(activate2)
        
        # Output layer
        outHiddenLayer3 = self.fc3(activate2)
        return outHiddenLayer3 