
import torch
from .dqn import QNetwork
from .replay_buffer import ReplayBuffer
import gymnasium as gym
import torch.nn.functional as F

class DoubleDQNAgent:
    
    def __init__(self, input_dim, hidden_dim, output_dim, discount_factor=0.99):
        self.q_network = QNetwork(input_dim, hidden_dim, output_dim)
        self.target_network = self.q_network.create_target_network()
        for param in self.target_network.parameters(): param.requires_grad_(False)
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001/2)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
    def sync_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def select_action(self, state, epsilon):
        output_dim = self.q_network.fc2.out_features  # Get the output dimension from the Q-network
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, output_dim, (1,))
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                return torch.argmax(q_values, dim=1)
            
    def load_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def compute_td_target(self, rewards, next_states, dones):
        with torch.no_grad():
            # Online network SELECTS the action
            next_actions = self.q_network(next_states).argmax(dim=1)

            # Target network EVALUATES the selected action
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            return rewards + (1 - dones) * self.discount_factor * next_q_values
        
    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # The target computation uses no_grad internally.
        td_target = self.compute_td_target(rewards, next_states, dones)
        loss = self.q_network.update(
            states, actions, td_target, self.optimizer
        )

        return loss
    
    def reset_environment(self):
        self.env = gym.make("CartPole-v1")
        state, _ = self.env.reset()
        return torch.tensor(state, dtype=torch.float32)
    
    def step_environment(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action.item())
        return torch.tensor(next_state, dtype=torch.float32), reward, terminated, truncated
    
    def train(self, num_episodes, batch_size, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        epsilon = epsilon_start
        for episode in range(num_episodes):
            state = self.reset_environment()
            done = False
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, terminated, truncated = self.step_environment(action)
                self.load_to_buffer(state, action.item(), reward, next_state, terminated)
                state = next_state
                done = terminated or truncated
                loss = self.train_step(batch_size)
                
                # if loss is not None and episode % 100 == 0:  # Print loss every 10 episodes
                #     # print(f"Episode {episode}, Loss: {loss}")
            
            # Decay epsilon after each episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Periodically sync the target network with the Q-network
            if episode % 10 == 0:
                self.sync_target_network()
                
def evaluation():
    agent = DoubleDQNAgent(input_dim=4, hidden_dim=64, output_dim=2)
    agent.train(num_episodes=1000, batch_size=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995)
    
    # Evaluate the learned policy
    total_rewards = 0
    num_eval_episodes = 100
    for _ in range(num_eval_episodes):
        state = agent.reset_environment()
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # No exploration during evaluation
            next_state, reward, terminated, truncated = agent.step_environment(action)
            total_rewards += reward
            state = next_state
            done = terminated or truncated
            
    average_reward = total_rewards / num_eval_episodes
    print(f"Average Reward over {num_eval_episodes} evaluation episodes: {average_reward}")
                
if __name__ == "__main__":
    evaluation()   