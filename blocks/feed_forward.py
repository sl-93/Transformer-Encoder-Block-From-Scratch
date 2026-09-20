import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(self, 
                 d_model, 
                 d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        return x