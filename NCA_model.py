import torch
import torch.nn as nn
import torch.nn.functional as F

class NCAGenerator(nn.Module):
    def __init__(self, input_dim=128, output_dim=2048, hidden_dim=256):
        super(NCAGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
