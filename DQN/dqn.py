'''
Cначала создай сеть QNetwork с входом из 16 признаков, 
скрытым слоем на 64 нейрона и 4 выходами 
— по одному Q-value на действие. 
Используй Linear → ReLU → Linear; 
пока не добавляй Replay Buffer и цикл обучения.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from .replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    