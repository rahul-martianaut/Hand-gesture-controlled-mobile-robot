import torch
import torch.nn as nn



class KeyPointClassifier_model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeyPointClassifier_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.softmax(self.fc3(x), dim=1)
        return x