import torch
import torch.nn as nn

class GELU_PyTorch_Tanh(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

# Define model class
class EMixnet(nn.Module):
    def __init__(self, base_model):
        super(EMixnet, self).__init__()
        self.base_model = base_model
        self.base_features = nn.Sequential(*list(self.base_model.children())[:-1])

        self.fc1 = nn.Linear(1536, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.gelu = GELU_PyTorch_Tanh()
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.final_fc = nn.Linear(2048, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract the basic features
        x = self.base_features(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        x = self.final_fc(x)
        x = self.sigmoid(x)
        return x