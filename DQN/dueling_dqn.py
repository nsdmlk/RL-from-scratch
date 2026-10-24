import torch
import torch.nn as nn
from .dqn import QNetwork

class DuelingQNetwork(QNetwork):
    
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=4):
        super(DuelingQNetwork, self).__init__(input_dim, hidden_dim, output_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def create_target_network(self):
        target_network = DuelingQNetwork(
            input_dim=self.fc1.in_features,
            hidden_dim=self.fc1.out_features,
            output_dim=self.advantage_stream[-1].out_features
        )

        target_network.load_state_dict(self.state_dict())
        target_network.eval()

        return target_network

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values