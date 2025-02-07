import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class AttentionCritic(nn.Module):
    """
    Attention-based critic network that evaluates actions taken by all agents.
    Uses scaled dot-product attention to model agent interactions.

    Args:
        state_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        n_agents (int): Number of agents
        hidden_dim (int): Size of hidden layers
    """

    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=64):
        super(AttentionCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        # Encoder network: processes concatenated state and action
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head attention
        self.n_heads = 4
        self.head_dim = hidden_dim // self.n_heads

        # Networks for attention mechanism
        # Generates keys for attention
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        # Generates queries for attention
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        # Generates values for attention
        self.value_net = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

        # Final network to produce Q-values
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def forward(self, states, actions):
        """
        Forward pass of the critic network.

        Args:
            states: Agent observations [batch_size, ep_len, n_agents, state_dim]
            actions: Agent actions [batch_size, ep_len, n_agents, action_dim]

        Returns:
            Q-values: [batch_size, ep_len, n_agents, 1]
        """
        batch_size, ep_len = states.shape[:2]

        # Reshape to process all timesteps together
        states = states.view(-1, self.n_agents, self.state_dim)
        actions = actions.view(-1, self.n_agents, self.action_dim)

        # Each agent gets the same global state
        # Process states and actions
        inputs = torch.cat([states, actions], dim=-1)
        encoded = self.encoder(inputs)

        # Generate keys, queries and values for multi-head attention
        keys = self.key_net(encoded).view(-1, self.n_agents,
                                          self.n_heads, self.head_dim)
        queries = self.query_net(
            encoded).view(-1, self.n_agents, self.n_heads, self.head_dim)
        values = self.value_net(
            encoded).view(-1, self.n_agents, self.n_heads, self.head_dim)

        # Attention computation: Compute scaled dot-product attention
        scale = torch.sqrt(torch.FloatTensor(
            [self.head_dim])).to(states.device)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention weights to values
        attended = torch.matmul(attention, values)
        attended = attended.view(-1, self.n_agents,
                                 self.n_heads * self.head_dim)

        # Generate final Q-values
        q_values = self.final_net(attended)

        # Reshape back to include episode length dimension
        return q_values.view(batch_size, ep_len, self.n_agents, 1)


class Actor(nn.Module):
    """
    Actor network that generates actions for each agent based on the full observations.

    Args:
        state_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        hidden_dim (int): Size of hidden layers        
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 확률 분포로 변환
        )

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs):
        """
        Args:
            obs: [batch_size, ep_len, state_dim] or [batch_size, state_dim]
        """
        if obs.dim() == 3:
            batch_size, ep_len, state_dim = obs.shape
            # Reshape to process all timesteps together
            obs = obs.view(-1, state_dim)
            actions = self.net(obs)
            # Reshape back
            return actions.view(batch_size, ep_len, -1)
        return self.net(obs)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling episodes.

    Args:
        capacity (int): Maximum size of buffer
        buffer_episodes (list): List to store complete episodes
        current_episode (list): Temporary storage for current episode
        n_agents (int): Number of agents
        action_dim (int): Dimension of action space
        state_dim (int): Dimension of local observation space
    """

    def __init__(self, capacity: int, n_agents: int, action_dim: int, state_dim: int):
        self.capacity = capacity
        self.buffer_episodes = []
        self.current_episode = []
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim

    def push(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
             next_state: np.ndarray, done: bool):
        """
        Add a new transition to the buffer.

        Args:
            state: [n_agents, state_dim] 
            actions: [n_agents, action_dim] 
            rewards: [n_agents, 1]   
            next_state: [n_agents, state_dim]
            done: bool
        """
        # Store transition in current episode
        self.current_episode.append(
            (state, actions, rewards, next_state, done))

        # If episode is done, store it and start new episode
        if done:
            self.buffer_episodes.append(self.current_episode)
            self.current_episode = []

            # Remove oldest episode if capacity is exceeded
            if len(self.buffer_episodes) > self.capacity:
                self.buffer_episodes.pop(0)

    def sample(self, batch_size: int):
        """Sample and process batch_size number of complete episodes for training

        Args:
            batch_size (int): Number of episodes to sample

        Returns:
            states: [batch_size, max_episode_len, n_agents, state_dim]
            actions: [batch_size, max_episode_len, n_agents, action_dim] 
            rewards: [batch_size, max_episode_len, n_agents, 1]
            next_states: [batch_size, max_episode_len, n_agents, state_dim]
            dones: [batch_size, max_episode_len, 1]
            masks: [batch_size, max_episode_len] - To handle variable episode lengths
        """

        # Randomly select batch_size episodes
        selected_episodes = random.sample(self.buffer_episodes, batch_size)

        # Find max episode length
        max_ep_len = max(len(episode) for episode in selected_episodes)

        # Initialize tensors
        states = torch.zeros(batch_size, max_ep_len,
                             self.n_agents, self.state_dim)
        actions = torch.zeros(batch_size, max_ep_len,
                              self.n_agents, self.action_dim)
        rewards = torch.zeros(batch_size, max_ep_len, self.n_agents, 1)
        next_states = torch.zeros(batch_size, max_ep_len,
                                  self.n_agents, self.state_dim)
        dones = torch.zeros(batch_size, max_ep_len, 1)
        # 1 for actual steps, 0 for padding
        masks = torch.zeros(batch_size, max_ep_len)

        # Fill tensors with data
        for i, episode in enumerate(selected_episodes):

            # Process each transition in the episode
            for t, transition in enumerate(episode):
                state, action, reward, next_state, done = transition

                # Fill tensors at appropriate timestep
                states[i, t] = torch.FloatTensor(state)
                actions[i, t] = torch.FloatTensor(
                    np.array(action)).reshape(self.n_agents, -1)

                # reward를 numpy array로 변환 후 텐서로 변환
                reward_array = np.array(reward).reshape(self.n_agents, 1)
                rewards[i, t] = torch.FloatTensor(reward_array)

                next_states[i, t] = torch.FloatTensor(next_state)
                dones[i, t] = torch.FloatTensor([done])
                masks[i, t] = 1.0  # Mark as actual step

        return states, actions, rewards, next_states, dones, masks

    def __len__(self):
        """Return the number of complete episodes in buffer"""
        return len(self.buffer_episodes)


class MAAC:
    """
    Multi-Agent Attention Critic main class.
    Coordinates multiple actors and a centralized attention critic.

    Args:
        n_agents (int): Number of agents
        state_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        lr (float): Learning rate
        gamma (float): Discount factor
        tau (float): Soft update rate for target networks
    """

    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005):
        self.num_cpu_for_training = 0  # Number of times training was done on CPU
        self.num_cuda_for_inference = 0  # Number of times inference was done on GPU
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Default to CPU for inference
        self.device = torch.device("cpu")
        self.training = False
        print(f"Initialized MAAC on {self.device}")

        # Create actor networks (one per agent)
        self.actors = [Actor(state_dim, action_dim).to(self.device)
                       for _ in range(n_agents)]
        self.actors_target = [Actor(state_dim, action_dim).to(
            self.device) for _ in range(n_agents)]

        # Create critic networks (shared among agents)
        self.critic = AttentionCritic(
            state_dim, action_dim, n_agents).to(self.device)
        self.critic_target = AttentionCritic(
            state_dim, action_dim, n_agents).to(self.device)

        # Initialize optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8
        )

        # Initialize target networks
        self.update_targets(tau=1.0)

    def get_device(self):
        """Return the current device (cpu or gpu)"""
        return self.device

    def to_training_mode(self):
        """Switch to GPU and set training mode"""
        if not self.training:
            if torch.cuda.is_available():
                new_device = torch.device("cuda")
                # print(f"Switching to {new_device} for training")

                # Move all networks to GPU
                for actor in self.actors:
                    actor.to(new_device)
                for actor_target in self.actors_target:
                    actor_target.to(new_device)
                self.critic.to(new_device)
                self.critic_target.to(new_device)

                self.device = new_device
            else:
                self.num_cpu_for_training += 1  # Increment counter if using CPU for training

            self.training = True

    def to_inference_mode(self):
        """Switch to CPU and set inference mode"""
        if self.training:
            self.device = torch.device("cpu")
            self.training = False

            # Move all networks to CPU
            for actor in self.actors:
                actor.to(self.device)
            for actor_target in self.actors_target:
                actor_target.to(self.device)
            self.critic.to(self.device)
            self.critic_target.to(self.device)

            if self.device.type == "cuda":
                self.num_cuda_for_inference += 1  # Increment counter if using GPU for inference

    def select_action(self, state, agent_id, epsilon=0.1):
        """
        Select action for a given agent using epsilon-greedy policy

        Args:
            state: [state_dim]
            agent_id: agent index
            epsilon: exploration rate
        """
        # self.to_inference_mode()  # Ensure we're on CPU for inference

        # if random.random() < epsilon:
        #     action = np.random.randint(0, self.action_dim)
        #     print(f"Random action selected: {action}")
        #     return action

        # with torch.no_grad():
        #     state_tensor = torch.FloatTensor(
        #         state).unsqueeze(0).to(self.device)
        #     action_probs = self.actors[agent_id](state_tensor)
        #     noise = torch.randn_like(action_probs) * 0.1
        #     action_probs = F.softmax(action_probs + noise, dim=-1)
        #     return torch.argmax(action_probs).item()

        if random.random() < epsilon:  # Exploration
            action = np.random.randint(0, self.action_dim)
            # print(f"Random action selected: {action} ; {self.action_dim}")
            return action
        else:  # Exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(
                    state).unsqueeze(0).to(self.device)
                action_probs = F.softmax(
                    self.actors[agent_id](state_tensor), dim=-1)
                action = torch.argmax(action_probs).item()
                return action

    def update(self, batch_size, buffer):
        """
        Update actor and critic networks using sampled batch

        Args:
            batch_size: Number of episodes to sample
            buffer: Replay buffer containing episodes

        Returns:
            tuple: (critic_loss, actor_losses) - Loss values for logging
        """
        # Switch to GPU for training
        self.to_training_mode()

        # If buffer is not full enough, return dummy values
        if len(buffer) < batch_size:
            return 0, [0] * self.n_agents

        # Sample batch of transitions
        states, actions, rewards, next_states, dones, masks = buffer.sample(
            batch_size)

        # Move to device
        """
        states  :    [batch_size, max_ep_len, n_agents, state_dim]
        actions :    [batch_size, max_ep_len, n_agents, action_dim]
        rewards :    [batch_size, max_ep_len, n_agents, 1]
        next_states: [batch_size, max_ep_len, n_agents, state_dim]
        dones   :    [batch_size, max_ep_len, 1]
        masks   :    [batch_size, max_ep_len]
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        masks = masks.to(self.device)

        batch_size, max_ep_len = states.shape[:2]

        # Update critic
        with torch.no_grad():
            next_actions = torch.stack([
                self.actors_target[i](next_states[:, :, i])
                for i in range(self.n_agents)
            ], dim=2)  # [batch_size, max_ep_len, n_agents, action_dim]

            # critic_target returns [batch_size, max_ep_len, n_agents, 1]
            next_q = self.critic_target(next_states, next_actions)

            # Make sure dimensions match for the target calculation
            # [batch_size, max_ep_len, n_agents, 1]
            dones_expanded = dones.unsqueeze(
                2).expand(-1, -1, self.n_agents, -1)
            target_q = rewards + (1 - dones_expanded) * self.gamma * next_q

        # Get current Q-values
        current_q = self.critic(states, actions)
        # Calculate critic loss using masks
        masks_expanded = masks.unsqueeze(-1).unsqueeze(-1)
        critic_loss = (masks_expanded *
                       F.mse_loss(current_q, target_q.detach(), reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update actors
        actor_losses = []
        for i in range(self.n_agents):
            current_actions = actions.clone()
            current_actions[:, :, i] = self.actors[i](states[:, :, i])

            # Calculate actor loss using masks
            actor_loss = -(masks.unsqueeze(-1).unsqueeze(-1) *
                           self.critic(states, current_actions)).mean()
            actor_losses.append(actor_loss.item())

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

        # Soft update target networks
        self.update_targets()

        return critic_loss.item(), actor_losses

    def update_targets(self, tau=None):
        """Soft update target networks"""

        tau = tau if tau is not None else self.tau

        # Update actor targets
        for actor, actor_target in zip(self.actors, self.actors_target):
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)

    def check_device_usage_warnings(self):
        """
        Check for potential device usage issues and print warning messages.
        Should be called periodically during training/evaluation.
        """
        if self.num_cpu_for_training > 0:
            print("\033[93m" +  # Yellow color for warning
                  f"Warning: Training was performed on CPU {self.num_cpu_for_training} times. "
                  + "\033[0m")  # Reset color

        if self.num_cuda_for_inference > 0:
            print("\033[93m" +  # Yellow color for warning
                  f"Warning: Inference was performed on GPU {self.num_cuda_for_inference} times. "
                  + "\033[0m")  # Reset color
