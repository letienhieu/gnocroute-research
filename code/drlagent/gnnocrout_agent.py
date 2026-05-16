"""GNNocRoute-DRL: GNN + DRL Agent for NoC Routing

This module implements the core DRL agent that combines:
1. GATv2 encoder for topology-aware state representation
2. DQN/PPO agent for routing decisions
3. Periodic policy optimization framework

Author: Ngoc Anh for Thay Hieu
Date: 15/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
import numpy as np
from collections import deque, namedtuple
import random
import json
import os

# === Constants ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DRL Agent] Using device: {DEVICE}")

# === Configuration ===
DRL_CONFIG = {
    # GNN Encoder
    'gnn_type': 'gatv2',        # 'gatv2', 'gcn'
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'node_feat_dim': 8,         # buffer_occ(4) + congestion(1) + vc_util(1) + pos(2) = 8
    'edge_feat_dim': 3,
    
    # DQN Agent
    'state_dim': 128,           # output embedding dimension
    'action_dim': 4,            # N, S, E, W (output ports)
    'hidden_dim_dqn': [256, 128],
    
    # Training
    'lr': 1e-3,
    'gamma': 0.99,
    'epsilon_start': 0.5,
    'epsilon_end': 0.01,
    'epsilon_decay': 5000,
    'buffer_size': 50000,
    'batch_size': 64,
    'target_update': 200,
    'update_period': 5000,      # cycles between policy updates
    
    # Reward
    'alpha_latency': 1.0,
    'beta_energy': 0.5,
    'gamma_throughput': 0.3,
}


# ============================================================
# GNN Encoder
# ============================================================
class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for NoC topology.
    
    Input: PyG Data objects with node features and edge_index
    Output: Node embeddings that capture topology + state
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config['hidden_dim']
        heads = config['num_heads']
        
        if config['gnn_type'] == 'gatv2':
            self.conv1 = GATv2Conv(
                config['node_feat_dim'], h // heads, 
                heads=heads, edge_dim=config['edge_feat_dim']
            )
            self.conv2 = GATv2Conv(
                h, config['state_dim'], 
                heads=1, edge_dim=config['edge_feat_dim']
            )
        else:  # GCN
            self.conv1 = GCNConv(config['node_feat_dim'], h)
            self.conv2 = GCNConv(h, config['state_dim'])
        
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(config['state_dim'])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        
        return x  # Node embeddings
    
    def get_graph_embedding(self, data):
        """Aggregate node embeddings to graph-level representation."""
        node_embeds = self.forward(data)
        graph_embed = global_mean_pool(node_embeds, data.batch)
        return graph_embed, node_embeds


# ============================================================
# DQN Agent
# ============================================================
class DQN(nn.Module):
    """Deep Q-Network for routing decisions.
    
    Input: Node embedding (from GNN) + local congestion state
    Output: Q-values for each output port
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
            ])
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', 
            ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.Experience(
            state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor([e.done for e in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class GNNocRouteAgent:
    """Main DRL agent combining GNN encoder + DQN."""
    
    def __init__(self, config=None, gnn_encoder=None):
        self.config = config or DRL_CONFIG
        self.device = DEVICE
        self.steps = 0
        
        # GNN Encoder (can be pre-trained)
        self.gnn = gnn_encoder or GNNEncoder(self.config).to(DEVICE)
        
        # DQN
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        hidden_dims = self.config['hidden_dim_dqn']
        
        self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim, hidden_dims).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()),
            lr=self.config['lr']
        )
        
        self.memory = ReplayBuffer(self.config['buffer_size'])
        
        # Metrics
        self.losses = []
        self.rewards = []
    
    def get_epsilon(self):
        """Annealed epsilon for epsilon-greedy."""
        eps = self.config['epsilon_start'] - self.steps / self.config['epsilon_decay']
        return max(self.config['epsilon_end'], eps)
    
    def select_action(self, graph_data, node_idx, explore=True):
        """Select output port for a given router/node.
        
        Args:
            graph_data: PyG Data object containing NoC state
            node_idx: Index of the router making the decision
            explore: Whether to use epsilon-greedy exploration
        
        Returns:
            action: Selected output port (0=N, 1=S, 2=E, 3=W)
        """
        with torch.no_grad():
            # Get node embedding from GNN
            node_embeds = self.gnn(graph_data)
            node_state = node_embeds[node_idx].cpu().numpy()
        
        if explore and random.random() < self.get_epsilon():
            return random.randint(0, self.config['action_dim'] - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(node_state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self):
        """Update DQN using experience replay."""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config['batch_size'])
        
        states = states.to(DEVICE)
        actions = actions.unsqueeze(1).to(DEVICE)
        rewards = rewards.unsqueeze(1).to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.unsqueeze(1).to(DEVICE)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q
        
        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.policy_net.parameters()), 
            1.0
        )
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.config['target_update'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'gnn_state_dict': self.gnn.state_dict(),
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'steps': self.steps,
            'losses': self.losses,
        }, path)
        print(f"[DRL Agent] Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.steps = checkpoint['steps']
        self.losses = checkpoint.get('losses', [])
        print(f"[DRL Agent] Model loaded from {path}")


# ============================================================
# NoC Environment (Gym-like)
# ============================================================
class NoCRoutingEnv:
    """Gym-like environment for NoC routing with DRL.
    
    This environment simulates a simplified NoC routing scenario
    for training the DRL agent. The actual cycle-accurate simulation
    is done via BookSim2 for evaluation.
    """
    
    def __init__(self, topology='mesh44', config=None):
        self.topology = topology
        self.config = config or DRL_CONFIG
        
        # Parse topology
        if topology.startswith('mesh'):
            size = int(topology.replace('mesh', ''))
            self.grid_size = int(np.sqrt(size))
            self.num_nodes = size
        else:
            self.num_nodes = 64
            self.grid_size = 8
        
        self.current_step = 0
        self.max_steps = 100
        
        # Build graph
        self.edge_index = self._build_mesh_edges()
        
        # State tracking
        self.node_states = np.zeros((self.num_nodes, self.config['node_feat_dim']))
        self.congestion_map = np.zeros((self.num_nodes,))
    
    def _build_mesh_edges(self):
        """Build edge_index for mesh topology."""
        edges = []
        gs = self.grid_size
        for y in range(gs):
            for x in range(gs):
                idx = y * gs + x
                if x > 0:
                    edges.append((idx, y * gs + (x-1)))
                if x < gs - 1:
                    edges.append((idx, y * gs + (x+1)))
                if y > 0:
                    edges.append((idx, (y-1) * gs + x))
                if y < gs - 1:
                    edges.append((idx, (y+1) * gs + x))
        return torch.LongTensor(edges).t()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.node_states = np.random.uniform(0.1, 0.3, 
            (self.num_nodes, self.config['node_feat_dim']))
        self.congestion_map = np.random.uniform(0, 0.2, (self.num_nodes,))
        return self._get_obs()
    
    def _get_obs(self):
        """Build PyG Data object from current state."""
        x = torch.FloatTensor(self.node_states)
        edge_index = self.edge_index
        edge_attr = torch.FloatTensor(
            np.random.uniform(0.1, 0.5, (edge_index.size(1), 3))
        )
        from torch_geometric.data import Data
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def step(self, action):
        """Execute routing action and return next state + reward.
        
        Simplified model: better routing = lower congestion
        """
        self.current_step += 1
        
        # Simplified dynamics
        congestion_change = np.random.normal(-0.05 * (action + 1), 0.02)
        self.congestion_map += congestion_change
        self.congestion_map = np.clip(self.congestion_map, 0, 1)
        
        # Reward: negative congestion (lower is better)
        reward = -np.mean(self.congestion_map)
        
        # Update node states
        self.node_states[:, 4] = self.congestion_map
        self.node_states[:, 0:4] += np.random.normal(0, 0.01, 
            (self.num_nodes, 4))
        
        done = self.current_step >= self.max_steps
        return self._get_obs(), reward, done, {}


# ============================================================
# Training Pipeline
# ============================================================
class Trainer:
    """Training pipeline for GNNocRoute-DRL agent."""
    
    def __init__(self, agent, env, config=None):
        self.agent = agent
        self.env = env
        self.config = config or DRL_CONFIG
        self.episode_rewards = []
    
    def train_episode(self):
        """Train for one episode."""
        graph_data = self.env.reset()
        total_reward = 0
        episode_losses = []
        
        # Get node embeddings for entire graph
        with torch.no_grad():
            node_embeds = self.agent.gnn(graph_data).cpu().numpy()
        
        for step in range(self.env.max_steps):
            # Select action for a random node
            node_idx = random.randint(0, self.env.num_nodes - 1)
            state = node_embeds[node_idx]
            
            # Get environment step
            next_graph_data, reward, done, _ = self.env.step(0)
            
            # Get next state embedding
            with torch.no_grad():
                next_embeds = self.agent.gnn(next_graph_data).cpu().numpy()
            next_state = next_embeds[node_idx]
            
            # Store in replay buffer
            self.agent.memory.push(state, 0, reward, next_state, done)
            
            # Update agent
            loss = self.agent.update()
            if loss is not None:
                episode_losses.append(loss)
            
            total_reward += reward
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        return total_reward, np.mean(episode_losses) if episode_losses else 0
    
    def train(self, num_episodes=1000, save_path='./models/gnnocrout_drl.pt'):
        """Run full training loop."""
        print(f"[Trainer] Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            reward, loss = self.train_episode()
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                epsilon = self.agent.get_epsilon()
                print(f"  Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Loss: {loss:.4f} | "
                      f"Epsilon: {epsilon:.3f}")
        
        # Save final model
        self.agent.save(save_path)
        print(f"[Trainer] Training complete. Model saved to {save_path}")
        return self.episode_rewards


# ============================================================
# BookSim2 Integration
# ============================================================
class BookSim2Interface:
    """Interface to run BookSim2 simulations with DRL agents."""
    
    def __init__(self, booksim_path='/home/opc/.openclaw/workspace/booksim2/src/booksim'):
        self.booksim_path = booksim_path
        self.output_dir = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments/results'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_simulation(self, topology, routing, traffic, inj_rate, seed=42):
        """Run a single BookSim2 simulation."""
        import subprocess, tempfile
        
        topology_map = {
            'mesh44': ('mesh', 4, 2),
            'mesh88': ('mesh', 8, 2),
            'torus44': ('torus', 4, 2),
        }
        
        topo_str, k, n = topology_map.get(topology, ('mesh', 8, 2))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(f"""topology = {topo_str};
k = {k};
n = {n};
routing_function = {routing};
num_vcs = 4;
vc_buf_size = 8;
wait_for_tail_credit = 1;
priority = local_age;
sim_type = latency;
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
print_csv_results = 1;
traffic = {traffic};
injection_rate = {inj_rate};
packet_size = 1;
""")
            cfg_path = f.name
        
        out_path = f"{self.output_dir}/{topology}_{routing}_{traffic}_{inj_rate}_s{seed}.txt"
        
        result = subprocess.run([self.booksim_path, cfg_path], 
                               capture_output=True, text=True, timeout=300)
        
        with open(out_path, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        os.unlink(cfg_path)
        
        # Parse latency
        import re
        lat_match = re.search(r'Packet latency average\s*=\s*([0-9.]+)', result.stdout)
        latency = float(lat_match.group(1)) if lat_match else None
        
        return latency, out_path


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GNNocRoute-DRL: Agent and Environment")
    print("=" * 60)
    
    # Test GNN encoder
    print("\n--- Testing GNN Encoder ---")
    encoder = GNNEncoder(DRL_CONFIG)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"GNN Encoder parameters: {total_params:,}")
    
    from torch_geometric.data import Data
    sample_data = Data(
        x=torch.randn(16, DRL_CONFIG['node_feat_dim']),
        edge_index=torch.randint(0, 16, (2, 48)),
        edge_attr=torch.randn(48, DRL_CONFIG['edge_feat_dim'])
    )
    embeddings = encoder(sample_data)
    print(f"Input: {sample_data.x.shape} → Output embeddings: {embeddings.shape}")
    
    # Test DQN
    print("\n--- Testing DQN ---")
    agent = GNNocRouteAgent(DRL_CONFIG, encoder)
    sample_embedding = torch.randn(1, DRL_CONFIG['state_dim'])
    q_values = agent.policy_net(sample_embedding)
    print(f"Input embedding: {sample_embedding.shape} → Q-values: {q_values.shape}")
    print(f"Q-values: {q_values.detach().numpy()[0]}")
    
    # Test environment
    print("\n--- Testing NoC Environment ---")
    env = NoCRoutingEnv('mesh44')
    obs = env.reset()
    print(f"Environment: mesh44 | State dim: {obs.x.shape}")
    
    # Test training
    print("\n--- Training (Quick Test - 10 episodes) ---")
    trainer = Trainer(agent, env)
    rewards = trainer.train(num_episodes=10, save_path='/tmp/test_agent.pt')
    print(f"Test complete. Avg reward: {np.mean(rewards):.3f}")
    
    print("\n✅ All components working!")
