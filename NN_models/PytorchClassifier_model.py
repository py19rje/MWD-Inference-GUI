import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierModel(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.25)
        # Compute flattened size
        dummy = torch.zeros(1, 1, input_length)
        x = self.pool1(self.bn1(self.conv1(dummy)))
        x = self.pool2(self.bn2(self.conv2(self.drop1(x))))
        flat_size = x.view(1, -1).shape[1]
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 3)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, 1) -> (B, 1, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return x