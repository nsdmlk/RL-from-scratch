import torch
import torch.nn as nn

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
    
    def create_target_network(self):
        target_network = QNetwork(input_dim=self.fc1.in_features, hidden_dim=self.fc1.out_features, output_dim=self.fc2.out_features)
        target_network.load_state_dict(self.state_dict())
        target_network.eval()  # Set the target network to evaluation mode
        return target_network
    
    def td_target(self, reward, next_state, done, discount_factor, target_network):
        with torch.no_grad():
            next_q_values = target_network(next_state)
            max_next_q_value = torch.max(next_q_values, dim=1).values
            td_target = reward + (1 - done) * discount_factor * max_next_q_value
        return td_target

    def update(self, states, actions, td_target, optimizer):
            q_values = self.forward(states)
    
            action_q_values = q_values.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
    
            loss = nn.HuberLoss()(action_q_values, td_target)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            return loss.item()
        
    def td_error(self, state, action, reward, next_state, done, discount_factor, target_network):
        with torch.no_grad():
            # Compute the Q-values for the current state
            q_values = self.forward(state)
            
            # Gather the Q-values corresponding to the taken actions
            action_q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            
            # Compute the TD target using the target network
            td_target = self.td_target(reward, next_state, done, discount_factor, target_network)
            
            # Compute the TD error
            td_error = td_target - action_q_values
        
        return td_error