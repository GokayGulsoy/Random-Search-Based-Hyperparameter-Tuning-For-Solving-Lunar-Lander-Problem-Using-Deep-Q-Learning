import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
import torch.nn.functional as F
from dqn import QNetwork 


BUFFER_SIZE = int(1e5)  # Replay memory size
TAU = 1e-3              # Soft update of target parameters
UPDATE_EVERY = 4        # How often to update the network

# detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)

class Agent:
    def __init__(self, state_size, action_size, seed=42, gamma=0.99, lr=1e-4, hidden_size=64, batch_size=64):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed 
        self.GAMMA = gamma 
        self.LR = lr
        self.BATCH_SIZE = batch_size
        
        # Q-Network for Policy & Target
        self.qnetwork_policy = QNetwork(state_size, action_size, seed, hidden_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=self.LR)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.BATCH_SIZE, seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0: 
            # if enough samples are available in memory, get random 
            # subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss_val = self.learn(experiences, self.GAMMA)

                return loss_val

    def act(self, state, eps=0.0):
        """Returns actions for a given state as per current policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # setting policy network to evaluation mode
        self.qnetwork_policy.eval()
        with torch.no_grad():
            action_values = self.qnetwork_policy(state)
        
        self.qnetwork_policy.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            # exploit best action
            return np.argmax(action_values.cpu().data.numpy())
        else: 
            # explore
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        # get max predicted Q values (for next states) from target model
        Q_targets_max = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_max *(1 - dones))
        
        # get the expected Q values from local model
        Q_expected = self.qnetwork_policy(states).gather(1, actions)
        # compute loss 
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network
        self.soft_update(self.qnetwork_policy, self.qnetwork_target, TAU)
        
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)
    
        
# class that stores made transitions as a memory
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed=42):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
    