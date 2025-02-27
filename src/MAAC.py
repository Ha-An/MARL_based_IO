import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class AttentionModule(nn.Module):
    """
    Attention module for multi-agent attention critic network.
    Uses scaled dot-product attention to model agent interactions. 
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(AttentionModule, self).__init__()
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.head_dim = hidden_dim // num_heads

        # Multi-head projection layers
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, query, keys):
        batch_size = query.size(0)

        # Linear projections and split into heads
        q = self.query(query).view(
            batch_size, 1, self.num_heads, self.head_dim)
        k = self.key(keys).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(keys).view(
            batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to [batch_size(0), num_heads(1), seq_len(2), head_dim(3)] ->  텐서의 1번 차원과 2번 차원의 위치를 서로 바꿈
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention for each head
        attention_weights = torch.matmul(
            q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention to values
        # [batch_size, num_heads, 1, head_dim]
        attended_values = torch.matmul(attention_weights, v)

        # Concatenate heads and project
        attended_values = attended_values.transpose(
            1, 2).contiguous()  # [batch_size, 1, num_heads, head_dim]
        attended_values = attended_values.view(
            batch_size, 1, -1)  # [batch_size, 1, hidden_dim]
        output = self.output_projection(
            attended_values.squeeze(1))  # [batch_size, output_dim]

        return output


class AttentionCritic(nn.Module):
    def __init__(self, state_nvec, action_dims, num_agents, hidden_dim, num_heads):
        super(AttentionCritic, self).__init__()
        self.num_agents = num_agents

        # 상태 차원 (float 입력을 가정)
        self.state_dim = len(state_nvec)

        # print("self.state_dim: ", self.state_dim)

        self.fc1_state = nn.Linear(self.state_dim, hidden_dim)
        # 각 에이전트별 행동 인코딩
        self.fc1_actions = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in action_dims
        ])

        # 각 에이전트를 위한 어텐션 모듈: (state + action)를 합친 것(=hidden_dim * 2)을 AttentionModule의 input_dim으로 사용
        self.attention = AttentionModule(input_dim=hidden_dim*2,
                                         hidden_dim=hidden_dim,
                                         output_dim=hidden_dim, num_heads=num_heads)

        # 최종 Q-value를 위한 레이어
        # *3: state + action + attention
        self.fc2 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)  # 각 에이전트별로 하나의 Q-value 출력

    def forward(self, states, actions):
        """
        Args:
            states: [batch_size, num_agents, len(nvec)] 형태(각 에이전트의 MultiDiscrete 상태)
            actions: list of Tensors, 각 shape [batch_size, action_dim_i]
                     => 예) 각 에이전트 i에 대한 action(one-hot 혹은 연속 값 등)
        Return:
            Q-values: [batch_size, num_agents]
        """

        # 상태 인코딩
        # [batch_size, hidden_dim]
        encoded_state = F.relu(self.fc1_state(states))
        # [batch_size, num_agents, hidden_dim]
        encoded_states = encoded_state.unsqueeze(
            1).expand(-1, self.num_agents, -1)
        # 행동 인코딩
        encoded_actions_list = [F.relu(self.fc1_actions[i](
            actions[i])) for i in range(self.num_agents)]
        # [batch_size, num_agents, hidden_dim]
        encoded_actions = torch.stack(encoded_actions_list, dim=1)
        # [batch_size, num_agents, hidden_dim*2]
        state_action = torch.cat([encoded_states, encoded_actions], dim=-1)

        q_values_per_agent = []
        for i in range(self.num_agents):
            query = state_action[:, i]  # [batch_size, hidden_dim*2]
            keys = state_action  # [batch_size, num_agents, hidden_dim*2]
            attended = self.attention(query, keys)  # [batch_size, hidden_dim]
            # [batch_size, hidden_dim*3]
            combined = torch.cat([query, attended], dim=-1)
            x = F.relu(self.fc2(combined))
            q_value = self.fc_out(x)  # [batch_size, 1]
            q_values_per_agent.append(q_value)

        # [batch_size, num_agents]
        q_values = torch.cat(q_values_per_agent, dim=1)
        return q_values


class Actor(nn.Module):
    """
    Actor network that outputs action probabilities for a given state (MultiDiscrete).
    """

    def __init__(self, state_nvec, action_dim, hidden_dim=64):
        """
        Args:
            state_nvec: (MultiDiscrete) -> length = state_dim
            action_dim: discrete action size
        """
        super(Actor, self).__init__()
        self.state_dim = len(state_nvec)
        # print("self.state_dim: ", self.state_dim)

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        state: [batch_size, state_dim]
        return: [batch_size, action_dim] (softmax over discrete actions)
        """
        # print("state.shape: ", state.shape)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc_out(x), dim=-1)
        return action_probs


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions. 
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen=buffer_size)  # deque -> 가장 오래된 경험이 자동으로 삭제

    def push(self, state: np.ndarray, actions, reward,
             next_state: np.ndarray, done: bool):
        """
        Add a new experience to the buffer.
        """

        state_tensor = torch.as_tensor(
            state, dtype=torch.float, device=self.device)
        actions_tensor = torch.as_tensor(
            actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(
            np.array(reward), dtype=torch.float, device=self.device)
        next_state_tensor = torch.as_tensor(
            next_state, dtype=torch.float, device=self.device)
        done_tensor = torch.as_tensor(
            np.array(done), dtype=torch.float, device=self.device)

        experience = (state_tensor, actions_tensor,
                      reward_tensor, next_state_tensor, done_tensor)

        # print("experience: ", experience)
        # exit()

        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer for training

        Args:
            batch_size (int): Number of episodes to sample

        Returns:

        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.stack([exp[0] for exp in batch])
        actions = torch.stack([exp[1] for exp in batch])
        rewards = torch.stack([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])   # ...
        dones = torch.stack([exp[4] for exp in batch])

        # print("states, actions, rewards, next_states, dones:")
        # print(states.shape, actions.shape, rewards.shape,
        #       next_states.shape, dones.shape)
        # exit()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the number of transitions in buffer"""
        return len(self.buffer)


class MAAC:
    """
    Multi-Agent Attention Critic main class.
    Coordinates multiple actors and a centralized attention critic. 
    """

    def __init__(self, num_agents: int, multi_state_space_size, joint_action_space_size,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01, num_heads=4, hidden_dim=64):
        self.num_agents = num_agents
        # state의 각 차원의 discrete 개수 (MultiDiscrete) -> n-dimensional vector
        self.state_nvec = multi_state_space_size.nvec
        self.action_dims = joint_action_space_size.nvec  # 각 에이전트의 action space size
        # print("state_nvec, action_dims: ",
        #       self.state_nvec, self.action_dims)
        self.gamma = gamma
        self.tau = tau

        # Actor 네트워크 (에이전트별)
        self.actors = []
        self.actors_target = []
        self.actor_optimizers = []

        # Default to GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.training = False
        print(f"Initialized MAAC on {self.device}")

        # Create actor networks (one per agent)
        for i in range(num_agents):
            actor = Actor(state_nvec=self.state_nvec,
                          action_dim=self.action_dims[i],
                          hidden_dim=hidden_dim).to(self.device)
            actor_target = Actor(state_nvec=self.state_nvec,
                                 action_dim=self.action_dims[i],
                                 hidden_dim=hidden_dim).to(self.device)
            actor_target.load_state_dict(
                actor.state_dict())  # target network 초기화: main network 파라미터 복사
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)

            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.actor_optimizers.append(actor_optimizer)

        # Create critic networks (shared among agents)
        self.critic = AttentionCritic(state_nvec=self.state_nvec,
                                      action_dims=self.action_dims,
                                      num_agents=self.num_agents,
                                      hidden_dim=hidden_dim,
                                      num_heads=num_heads).to(self.device)
        self.critic_target = AttentionCritic(state_nvec=self.state_nvec,
                                             action_dims=self.action_dims,
                                             num_agents=self.num_agents,
                                             hidden_dim=hidden_dim,
                                             num_heads=num_heads).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic)

    def get_device(self):
        """Return the current device (cpu or gpu)"""
        return self.device

    def select_action(self, state, agent_id, epsilon):
        """
        Select action for a given agent using epsilon-greedy policy

        Args:
            state: Current state of the environment
            agent_id: ID of the agent selecting the action
            epsilon: Exploration rate

        Returns:
            action: Selected action for the agent
        """

        # epsilon-greedy exploration
        if np.random.random() < epsilon:  # Exploration
            action = np.random.randint(0, self.action_dims[agent_id])
        else:  # Exploitation
            state_tensor = torch.tensor(
                state, dtype=torch.float, device=self.device).unsqueeze(0)  # [1, state_dim]
            with torch.no_grad():
                action_probs = self.actors[agent_id](
                    state_tensor)  # [1, action_dim]
            action = torch.argmax(action_probs, dim=-1).item()

        # print("self.action_dims:", self.action_dims)
        # print("Action selected:", action)

        return action

    def update(self, batch_size, buffer):
        """
        Update actor and critic networks using sampled batch

        Args:
            batch_size: Number of transitions to sample
            buffer: Replay buffer containing transitions

        Returns:
            tuple: (critic_loss, actor_losses) - Loss values for logging
        """

        # 1. ReplayBuffer에서 샘플
        states, actions, rewards, next_states, dones = buffer.sample(
            batch_size)
        # shapes:
        # states: [batch_size, num_agents, len(nvec)]
        # actions: [batch_size, num_agents]               (각 에이전트의 실제 실행 액션(정수) )
        # rewards: [batch_size, num_agents]
        # next_states: [batch_size, num_agents, len(nvec)]
        # dones: [batch_size, num_agents]
        # print("states, actions, rewards, next_states, dones:", states.shape, actions.shape,
        #       rewards.shape, next_states.shape, dones.shape)

        # 2. Critic 업데이트
        # 2-1) Build one-hot for current actions
        current_actions_for_critic = []
        for i in range(self.num_agents):
            a_int = actions[:, i]  # [batch_size]
            a_onehot = F.one_hot(
                a_int, num_classes=self.action_dims[i]).float()
            current_actions_for_critic.append(a_onehot)

        # 2-2) Next actions from target actor
        next_actions_for_critic = []
        for i in range(self.num_agents):
            probs_i = self.actors_target[i](
                next_states)  # [batch_size, action_dim_i]
            a_int = torch.argmax(probs_i, dim=-1)         # [batch_size]
            a_onehot = F.one_hot(
                a_int, num_classes=self.action_dims[i]).float()
            next_actions_for_critic.append(a_onehot)

        # 2-3) Critic으로 Q(s, a) 계산 (현재)
        current_q_values = self.critic(states, current_actions_for_critic)

        # 2-4) Critic Target으로 Q(s', a') 계산 (다음)
        next_q_values = self.critic_target(
            next_states, next_actions_for_critic)  # [batch_size, num_agents]

        # 2-5) TD Target (dones가 [batch_size, num_agents] 형태)
        rewards = rewards.unsqueeze(-1).expand(-1, self.num_agents)
        dones = dones.unsqueeze(-1).expand(-1, self.num_agents)

        # print("states, actions, rewards, next_states, dones:", states.shape, actions.shape,
        #       rewards.shape, next_states.shape, dones.shape)
        # print("next_q_values:", next_q_values.shape)

        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # 2-6) MSE Loss
        critic_loss = F.mse_loss(current_q_values, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Actor 업데이트
        actor_losses = []
        for i in range(self.num_agents):
            # same global state for all agents
            state_i = states  # [batch_size, state_dim]

            # forward pass
            action_probs_i = self.actors[i](
                state_i)  # [batch_size, action_dim_i]

            # replace agent i's action in the list with its policy output
            current_actions_policy = current_actions_for_critic.copy()
            # policy actions (not one-hot yet, but let's feed in the raw probabilities to Critic)
            # you could also do one-hot if you prefer that approach
            current_actions_policy[i] = action_probs_i

            # [batch_size, num_agents]
            q_values = self.critic(states, current_actions_policy)
            q_i = q_values[:, i]  # [batch_size]

            # simple policy gradient: - mean( log(pi(a|s)) * Q )
            log_probs_i = torch.log(action_probs_i + 1e-10)
            actor_loss = - (log_probs_i * q_i.detach().unsqueeze(-1)).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            actor_losses.append(actor_loss.item())

        # 4. Target network soft update
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
