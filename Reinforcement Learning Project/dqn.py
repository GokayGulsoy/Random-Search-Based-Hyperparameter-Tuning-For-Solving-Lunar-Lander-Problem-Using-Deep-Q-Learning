import torch
import torch.nn as nn
import torch.nn.functional as F

# class that implements 3-layer neural network
# for Deep-Q learning RL algorithm
class QNetwork(nn.Module):
    """
    Standard Deep Q-Network for LunarLander-v3
    Input: State (8 values)
    Output: Q-Values for Actions (4 values)
    """
    def __init__(self, num_observations, num_actions, seed=42, hidden_size=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Linear(num_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x)
