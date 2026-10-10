import numpy as np
import gymnasium

class QLearningAgent:
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((n_states, n_actions))
        
    def select_action(self, state): 
        '''
        Epsilon-greedy action selection
        '''
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)  # Explore: random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: best action from Q-table
        
    def decay_exploration_rate(self):
        '''
        Decay the exploration rate after each episode
        '''
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, 0.01)  # Ensure it doesn't go below a minimum value
        
    def update(self, state, action, reward, next_state, terminated):
        '''
        Update Q-table using the Q-learning update rule
        '''
        done = terminated
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1.0 - terminated)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
        # Decay exploration rate
        if done:
            self.decay_exploration_rate()
    
    
        
    def train(self, env, n_episodes=1000, max_steps_per_episode=100):
        '''
        Train the agent in the given environment
        '''
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.update(state, action, reward, next_state, terminated)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}, Exploration Rate: {self.exploration_rate:.4f}")
    
    def evaluate(self, env, n_episodes=100, max_steps_per_episode=100):
        '''
        Evaluate the agent's performance in the given environment
        '''
        total_rewards = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                action = np.argmax(self.q_table[state])  # Always exploit during evaluation
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
        
        average_reward = np.mean(total_rewards)
        print(f"Average Reward over {n_episodes} episodes: {average_reward}")
        
if __name__ == "__main__":
    # Example usage with FrozenLake environment
    env = gymnasium.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    agent = QLearningAgent(n_states, n_actions)
    agent.train(env, n_episodes=1000)
    agent.evaluate(env, n_episodes=100)    