import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class AttentionModule(nn.Module):
    """
    Multi-head attention module for the centralized critic network.

    Attributes:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        query (nn.Linear): Linear layer for query projection
        key (nn.Linear): Linear layer for key projection
        value (nn.Linear): Linear layer for value projection
        output_projection (nn.Linear): Linear layer for output 
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        """
        Args:
            input_dim (int): Input dimension of the module
            hidden_dim (int): Hidden dimension of the module
            output_dim (int): Output dimension of the module
            num_heads (int): Number of attention heads 
        """
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
        """ 
        Args:
            query: [batch_size, input_dim]
            keys: [batch_size, num_agents, input_dim]
        Return:
            output: [batch_size, output_dim]
        """
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
    """
    Centralized critic network that evaluates the Q-values for all agents.

    Attributes:
        num_agents (int): Number of agents in the environment
        state_dim (int): Dimension of the state space
        fc1_state (nn.Linear): Linear layer for state encoding
        fc1_actions (nn.ModuleList): List of linear layers for action encoding
        attention (AttentionModule): Multi-head attention module
        fc2 (nn.Linear): Linear layer for output projection
        fc_out (nn.Linear): Linear layer for Q-value output
    """

    def __init__(self, state_nvec, action_dims, num_agents, hidden_dim, num_heads):
        """
        Args:
            state_nvec (list): List of state dimensions for each agent
            action_dims (list): List of action dimensions for each agent
            num_agents (int): Number of agents in the environment
            hidden_dim (int): Hidden dimension of the network
            num_heads (int): Number of attention heads
        """
        super(AttentionCritic, self).__init__()
        self.num_agents = num_agents

        # Input dimensions
        self.state_dim = len(state_nvec)
        # print("self.state_dim: ", self.state_dim)

        self.fc1_state = nn.Linear(self.state_dim, hidden_dim)
        # for each agent, create a linear layer for action encoding
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
    Actor network for each agent in the environment.

    Attributes:
        state_dim (int): Dimension of the state space
        fc1 (nn.Linear): Linear layer for hidden layer
        fc2 (nn.Linear): Linear layer for hidden layer
        fc_out (nn.Linear): Linear layer for output 
    """

    def __init__(self, state_nvec, action_dim, hidden_dim=64):
        """
        Args:
            state_nvec (list): List of state dimensions for each agent
            action_dim (int): Dimension of the action space
            hidden_dim (int): Hidden dimension of the network 
        """
        super(Actor, self).__init__()
        self.state_dim = len(state_nvec)
        # print("self.state_dim: ", self.state_dim)

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim]
        Return:
            [batch_size, action_dim] (softmax over discrete actions) 
        """
        # print("state.shape: ", state.shape)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc_out(x), dim=-1)
        return action_probs


class ReplayBuffer:
    """
    Proportional Prioritized Experience Replay 

    Attributes:
        buffer: 실제 transition 저장 (deque)
        priorities: 각 transition마다 우선순위 값(p_i)을 저장
        max_priority: 새 transition이 들어올 때 줄 초기 우선순위(보통 현재까지의 최대값)
        alpha: 우선순위의 지수 계수
        beta: 중요도 가중치 지수 (샘플링 편향 보정)
        beta_increment_per_sampling: 학습이 진행됨에 따라 beta를 점진적으로 1에 수렴시키기 위함
    """

    def __init__(self, buffer_size, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-4):
        """
        Args:
            buffer_size (int): Replay buffer의 최대 크기
            alpha (float): TD 오차를 p=|δ|^α 로 변환할 때 사용
            beta (float): 중요도 가중치 계산할 때 사용
            beta_increment_per_sampling (float): beta 증가량
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

        # alpha: TD오차를 p=|δ|^α 로 변환할 때 사용
        # beta: 중요도 가중치 계산할 때 사용
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # 우선순위 목록 (buffer와 동일한 크기 유지)
        self.priorities = deque(maxlen=buffer_size)

        # 새로 들어오는 transition에 초기로 줄 우선순위(대개 지금까지의 최대 우선순위값 사용)
        self.max_priority = 1.0

    def push(self, transition):
        """
        새로운 transition을 버퍼에 저장

        Args:
            transition: (state, action, reward, next_state, done) tuple
        """
        # print("experience: ", transition)
        # exit()

        # 버퍼가 찼을 때 자동으로 가장 오래된 transition 삭제
        self.buffer.append(transition)

        # 새 transition의 우선순위를 '현재까지의 max_priority'로 설정
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        """
        batch_size 개수만큼 우선순위 비례 확률로 샘플링 후, 중요도 가중치(is_weights)와 함께 반환

        Args:
            batch_size (int): 샘플링할 transition 개수
        Returns:
            (states, actions, rewards, next_states, dones, indices, is_weights)
        """
        if len(self.buffer) == 0:
            return None

        # 전체 우선순위 합
        priorities_array = np.array(self.priorities, dtype=np.float32)
        sum_of_priorities = np.sum(priorities_array**self.alpha)

        # batch_size개를 뽑을 인덱스를 확률적으로 결정
        prob = priorities_array**self.alpha / sum_of_priorities
        indices = np.random.choice(
            len(self.buffer), batch_size, p=prob, replace=False)

        # beta를 점진적으로 증가 (논문: beta는 학습 후반에 1.0 근처로)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # 중요도 가중치 계산: w_i = (1/(N * P(i)))^beta
        # N=버퍼 사이즈, P(i)=prob[i]
        # 안정성 위해 max()로 나눠서 정규화하기도 함
        is_weights = []
        for idx in indices:
            p_i = prob[idx]
            w_i = (1.0 / (len(self.buffer) * p_i))**self.beta
            is_weights.append(w_i)

        # normalize is_weights (선택 사항)
        is_weights = np.array(is_weights, dtype=np.float32)
        is_weights /= is_weights.max()  # 최대값으로 나누면 [0,1] 구간

        # 실제로 뽑힌 transition들
        sampled_transitions = [self.buffer[idx] for idx in indices]

        # 텐서 변환
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in sampled_transitions:
            states.append(t[0])
            actions.append(t[1])
            rewards.append(t[2])
            next_states.append(t[3])
            dones.append(t[4])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states = torch.as_tensor(
            np.array(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            np.array(actions), dtype=torch.long, device=device)
        rewards = torch.as_tensor(
            np.array(rewards), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.as_tensor(
            np.array(dones), dtype=torch.float32, device=device)

        # print("states, actions, rewards, next_states, dones:")
        # print(states.shape, actions.shape, rewards.shape,
        #       next_states.shape, dones.shape)
        # exit()

        return (states, actions, rewards, next_states, dones, indices, is_weights)

    def update_priorities(self, indices, td_errors):
        """
        샘플링 후 Critic 업데이트가 끝난 뒤, TD 오차(또는 |δ|)를 받아 해당 transition들의 우선순위를 다시 갱신

        Args:
            indices: 샘플링된 transition의 인덱스
            td_errors: Critic의 TD 오차 값
        """
        for i, td_error in zip(indices, td_errors):
            # p_i = |TD error| + epsilon (epsilon=1e-6 등), 여기선 생략
            p_i = abs(td_error.item()) + 1e-6  # 작은 상수를 더해서 0방지
            self.priorities[i] = p_i

            # 우선순위 최대값 갱신
            if p_i > self.max_priority:
                self.max_priority = p_i

    def __len__(self):
        return len(self.buffer)


class MAAC:
    """
    Multi-Agent Attention Critic main class.
    Coordinates multiple actors and a centralized attention critic. 

    Attributes:
        num_agents (int): Number of agents in the environment
        state_nvec (list): List of state dimensions for each agent
        action_dims (list): List of action dimensions for each agent
        gamma (float): Discount factor
        tau (float): Soft update parameter
        device (torch.device): Device to run the networks on
        actors (list): List of actor networks
        actors_target (list): List of target actor networks
        actor_optimizers (list): List of actor optimizers
        critic (AttentionCritic): Centralized critic network
        critic_target (AttentionCritic): Target critic network
        critic_optimizer (torch.optim): Critic optimizer 

    """

    def __init__(self, num_agents: int, multi_state_space_size, joint_action_space_size,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01, num_heads=4, hidden_dim=64):
        """
        Args:
            num_agents (int): Number of agents in the environment
            multi_state_space_size (spaces.MultiDiscrete): Size of the multi-agent state space
            joint_action_space_size (spaces.MultiDiscrete): Size of the joint action space
            lr_actor (float): Actor learning rate
            lr_critic (float): Critic learning rate
            gamma (float): Discount factor
            tau (float): Soft update parameter
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden dimension of the networks
        """
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

        # self.update_count = 0
        # self.hard_update_interval = 100  # 예시

        # Default to GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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

    def select_action_with_probs(self, state, agent_id, epsilon):
        """
        Version that also returns the policy probability distribution (for KL, entropy).

        Args:
            state: Current state of the environment
            agent_id: ID of the agent selecting the action
            epsilon: Exploration rate
        Returns:
            action: Selected action for the agent
            action_probs: Probability distribution over the action space
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actors[agent_id](
                state_tensor)[0]  # shape: [action_dim]

        if np.random.random() < epsilon:
            # choose random action
            action_idx = np.random.randint(0, self.action_dims[agent_id])
        else:
            action_idx = torch.argmax(action_probs, dim=-1).item()

        # Return (chosen action, entire probability distribution)
        return action_idx, action_probs.cpu().numpy()

    def calculate_kl_divergence(self, old_policy_probs, new_policy_probs):
        """
        KL Divergence between old/new policy distributions, averaged over agents.
        old_policy_probs, new_policy_probs: list of arrays (each array is shape [action_dim])
        """
        # simple discrete KL: sum p_old * log(p_old / p_new)
        kl_list = []
        for (p_old, p_new) in zip(old_policy_probs, new_policy_probs):
            p_old = np.clip(p_old, 1e-10, 1.0)
            p_new = np.clip(p_new, 1e-10, 1.0)
            kl = np.sum(p_old * np.log(p_old / p_new))
            kl_list.append(kl)
        return float(np.mean(kl_list))

    def update(self, batch_size, buffer):
        """
        Perform one update of Critic/Actor using PER samples.

        Args:
            batch_size (int): Batch size for training
            buffer (ReplayBuffer): Replay buffer for experience replay
        Return:
            critic_loss (float),
            actor_losses (list of floats),
            td_error (float),
            mean_q (float),
            std_q (float),
            policy_entropy (float),
            param_norms (tuple: (critic_norm, [actor_norms]))
        """
        sample_result = buffer.sample(batch_size)
        if sample_result is None:
            return 0., [0]*self.num_agents, 0., 0., 0., 0., (0., [0.]*self.num_agents)

        states, actions, rewards, next_states, dones, indices, is_weights = sample_result
        is_weights = torch.as_tensor(
            is_weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # === Critic Update ===
        # Build current one-hot actions
        current_actions_for_critic = []
        for i in range(self.num_agents):
            a_i = actions[:, i]  # [batch_size]
            a_onehot = F.one_hot(a_i, num_classes=self.action_dims[i]).float()
            current_actions_for_critic.append(a_onehot)

        # Next actions
        with torch.no_grad():
            next_actions_for_critic = []
            for i in range(self.num_agents):
                probs_i = self.actors_target[i](next_states)
                next_a_i = torch.argmax(probs_i, dim=-1)
                next_a_onehot = F.one_hot(
                    next_a_i, num_classes=self.action_dims[i]).float()
                next_actions_for_critic.append(next_a_onehot)

        # Q(s,a)
        current_q_values = self.critic(states, current_actions_for_critic)
        mean_q = current_q_values.mean().item()
        std_q = current_q_values.std().item()

        # Target
        next_q_values = self.critic_target(
            next_states, next_actions_for_critic)
        target_q = rewards.unsqueeze(-1) + (1 -
                                            dones.unsqueeze(-1)) * self.gamma * next_q_values

        td_errors = (current_q_values - target_q.detach())
        critic_loss = torch.mean(is_weights * td_errors**2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # PER update priorities
        td_errors_per_sample = td_errors.mean(dim=1)  # per transition
        buffer.update_priorities(indices, td_errors_per_sample)

        # === Actor Update ===
        actor_losses = []
        policy_entropy = 0.0
        for i in range(self.num_agents):
            # current policy output
            pi_i = self.actors[i](states)  # [batch_size, action_dim_i]

            # Construct a copy of the action list, but replace i-th with pi_i
            new_actions_for_critic = current_actions_for_critic.copy()
            new_actions_for_critic[i] = pi_i  # raw policy output

            # Q-values from critic
            # [batch_size, num_agents]
            q_values = self.critic(states, new_actions_for_critic)
            q_i = q_values[:, i]

            log_pi_i = torch.log(pi_i + 1e-10)
            actor_loss = - (log_pi_i * q_i.detach().unsqueeze(-1))
            # incorporate importance weight
            actor_loss = torch.mean(is_weights * actor_loss)

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            actor_losses.append(actor_loss.item())

            # policy entropy for agent i
            # H(pi) = - Sum( pi * log pi )
            ent_i = - (pi_i * log_pi_i).sum(dim=1).mean().item()
            policy_entropy += ent_i

        policy_entropy /= self.num_agents  # average over agents

        # Soft update
        self.update_targets()

        # Calculate average TD error
        avg_td_error = td_errors.mean().item()

        # Parameter Norms
        critic_norm = torch.norm(
            torch.cat([p.flatten() for p in self.critic.parameters()])).item()
        actor_norms = []
        for i in range(self.num_agents):
            anorm = torch.norm(
                torch.cat([p.flatten() for p in self.actors[i].parameters()])).item()
            actor_norms.append(anorm)
        param_norms = (critic_norm, actor_norms)

        return (critic_loss.item(),
                actor_losses,
                avg_td_error,
                mean_q,
                std_q,
                policy_entropy,
                param_norms)

    def update_targets(self, tau=None):
        """Soft update target networks"""

        tau = self.tau

        # Update actor targets
        for actor, actor_target in zip(self.actors, self.actors_target):
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)
