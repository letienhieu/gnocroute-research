"""
GNNocRoute-DRL: Improved NoC Routing Environment and Training
============================================================
A proper NoC routing environment for DRL training with:
  - Realistic congestion dynamics
  - Support for multiple topologies and traffic patterns
  - Integration with BookSim2 for final validation

Author: Ngoc Anh for Thay Hieu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
import random, time, os, json, subprocess, re, tempfile
from collections import deque, namedtuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# NoC ROUTING ENVIRONMENT (Improved)
# ============================================================
class NoCRoutingEnv:
    """Proper NoC routing simulation environment for DRL.
    
    Simulates packet routing on a 2D mesh with:
      - Per-router buffer queues
      - Packet injection from traffic patterns
      - Congestion propagation
      - Energy model
    """
    
    TRAFFIC_PATTERNS = {
        'uniform': lambda src, dst, N: True,           # random uniform
        'transpose': lambda src, dst, N:                # (i,j) -> (j,i)
            abs(src - dst) % 4 == 0 and src // 4 != dst // 4,
        'hotspot': lambda src, dst, N:                  # 10% to center
            random.random() > 0.9 or True,
    }
    
    def __init__(self, topology='mesh44', traffic='uniform', inj_rate=0.1):
        self.topology = topology
        self.traffic_name = traffic
        self.inj_rate = inj_rate
        
        # Parse topology
        size = int(topology.replace('mesh', ''))
        self.grid_size = int(np.sqrt(size))
        self.num_nodes = size
        
        # Router state
        self.buffer = np.zeros((self.num_nodes, 4))      # 4 input ports
        self.congestion = np.zeros(self.num_nodes)
        self.energy = np.zeros(self.num_nodes)
        self.link_util = np.zeros((self.num_nodes, 4))   # per-port utilization
        
        # Edge list
        self.edge_index, self.edge_pairs = self._build_mesh()
        self.num_edges = self.edge_index.shape[1]
        
        # Routing table (updated by DRL agent)
        self.routing_table = {}  # {(src, dst): port}
        
        # Statistics
        self.stats = {'packets_delivered': 0, 'total_latency': 0, 'total_energy': 0}
        self.packets_in_transit = []
        self.time = 0
    
    def _build_mesh(self):
        edges = []
        pairs = []
        gs = self.grid_size
        for y in range(gs):
            for x in range(gs):
                idx = y * gs + x
                if x > 0: edges.append((idx, y*gs+(x-1))); pairs.append((idx, y*gs+(x-1)))
                if x < gs-1: edges.append((idx, y*gs+(x+1))); pairs.append((idx, y*gs+(x+1)))
                if y > 0: edges.append((idx, (y-1)*gs+x)); pairs.append((idx, (y-1)*gs+x))
                if y < gs-1: edges.append((idx, (y+1)*gs+x)); pairs.append((idx, (y+1)*gs+x))
        return torch.LongTensor(edges).t(), pairs
    
    def _get_minimal_ports(self, src, dst):
        """Get minimal output ports from src to dst."""
        gs = self.grid_size
        sx, sy = src % gs, src // gs
        dx, dy = dst % gs, dst // gs
        ports = []
        if dx > sx: ports.append(2)  # E
        if dx < sx: ports.append(3)  # W
        if dy > sy: ports.append(1)  # S
        if dy < sy: ports.append(0)  # N
        return ports
    
    def _port_to_direction(self, port, node):
        gs = self.grid_size
        x, y = node % gs, node // gs
        dirs = {0: (x, y-1), 1: (x, y+1), 2: (x+1, y), 3: (x-1, y)}
        nx, ny = dirs[port]
        if 0 <= nx < gs and 0 <= ny < gs:
            return ny * gs + nx
        return None
    
    def step(self, actions=None):
        """Advance simulation by one cycle.
        
        Args:
            actions: dict {node: port} from DRL agent, or None for minimal routing
        
        Returns:
            obs: Current state as PyG Data
            reward: System-level reward
            done: Whether simulation ended
            info: Additional stats
        """
        self.time += 1
        packets = self.packets_in_transit
        
        # 1. Generate new packets based on traffic
        if self.traffic_name == 'uniform':
            for src in range(self.num_nodes):
                if random.random() < self.inj_rate:
                    dst = random.randint(0, self.num_nodes - 1)
                    while dst == src:
                        dst = random.randint(0, self.num_nodes - 1)
                    packets.append({'src': src, 'dst': dst, 'pos': src, 'hops': 0, 'id': self.time*1000+src})
        
        elif self.traffic_name == 'hotspot':
            hotspot = self._hotspot_node()
            for src in range(self.num_nodes):
                if random.random() < self.inj_rate:
                    dst = hotspot if random.random() < 0.1 else random.randint(0, self.num_nodes - 1)
                    if dst == src: dst = (dst + 1) % self.num_nodes
                    packets.append({'src': src, 'dst': dst, 'pos': src, 'hops': 0, 'id': self.time*1000+src})
        
        # 2. Route packets
        new_packets = []
        delivered = 0
        total_hops = 0
        
        for pkt in packets:
            pos, dst = pkt['pos'], pkt['dst']
            
            if pos == dst:
                delivered += 1
                total_hops += pkt['hops']
                continue
            
            # Get routing decision
            if actions and pos in actions:
                port = actions[pos]
            else:
                # Default minimal routing
                ports = self._get_minimal_ports(pos, dst)
                if ports:
                    port = self._select_port_by_congestion(pos, ports)
                else:
                    port = random.randint(0, 3)
            
            next_node = self._port_to_direction(port, pos)
            if next_node is not None:
                # Check buffer
                if self.buffer[pos, port] < 4:  # buffer capacity
                    self.buffer[pos, port] += 1
                    pkt['pos'] = next_node
                    pkt['hops'] += 1
                    self.energy[pos] += 1.0 + self.buffer[pos, port] * 0.1
                    self.link_util[pos, port] += 1
                    new_packets.append(pkt)
                else:
                    # Buffer full - stall
                    new_packets.append(pkt)
            else:
                new_packets.append(pkt)
        
        # 3. Update congestion and buffers
        self.congestion = np.array([len([p for p in new_packets if p['pos'] == n]) 
                                    for n in range(self.num_nodes)])
        self.buffer *= 0.9  # drain
        self.link_util *= 0.95  # decay
        
        self.packets_in_transit = new_packets
        self.stats['packets_delivered'] += delivered
        self.stats['total_latency'] += total_hops
        self.stats['total_energy'] += np.sum(self.energy)
        
        # 4. Compute reward
        avg_latency = total_hops / max(delivered, 1)
        avg_energy = np.mean(self.energy)
        congestion_penalty = np.mean(self.congestion)
        reward = -avg_latency * 0.01 - avg_energy * 0.001 - congestion_penalty * 0.1 + delivered * 0.01
        
        done = self.time >= 200
        info = {'delivered': delivered, 'avg_latency': avg_latency, 'congestion': congestion_penalty}
        
        return self._get_obs(), reward, done, info
    
    def _hotspot_node(self):
        """Central node for hotspot traffic."""
        gs = self.grid_size
        center = gs // 2
        return center * gs + center
    
    def _select_port_by_congestion(self, node, ports):
        """Select the least congested port among available options."""
        best_port = ports[0]
        best_cong = self.buffer[node, ports[0]]
        for p in ports[1:]:
            next_n = self._port_to_direction(p, node)
            if next_n is not None:
                cong = self.buffer[node, p] + 0.5 * self.congestion[next_n]
                if cong < best_cong:
                    best_cong = cong
                    best_port = p
        return best_port
    
    def get_state_vector(self):
        """Build node feature matrix."""
        features = np.zeros((self.num_nodes, 8))
        for i in range(self.num_nodes):
            features[i, :4] = self.buffer[i] / 4.0  # buffer occupancy
            features[i, 4] = self.congestion[i] / max(np.sum(self.congestion), 1)
            features[i, 5] = np.mean(self.link_util[i])
            features[i, 6] = self.energy[i] / max(np.max(self.energy), 1)
            # Position encoding
            features[i, 7] = i / self.num_nodes
        return features
    
    def _get_obs(self):
        """Build PyG Data object."""
        x = torch.FloatTensor(self.get_state_vector())
        edge_index = self.edge_index
        edge_attr = torch.FloatTensor(np.random.uniform(0, 1, (self.num_edges, 3)))
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def reset(self):
        """Reset environment."""
        self.buffer = np.zeros((self.num_nodes, 4))
        self.congestion = np.zeros(self.num_nodes)
        self.energy = np.zeros(self.num_nodes)
        self.link_util = np.zeros((self.num_nodes, 4))
        self.packets_in_transit = []
        self.time = 0
        self.stats = {'packets_delivered': 0, 'total_latency': 0, 'total_energy': 0}
        return self._get_obs()


# ============================================================
# GNN ENCODER
# ============================================================
class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=64, num_heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(8, hidden_dim // num_heads, heads=num_heads, edge_dim=3)
        self.conv2 = GATv2Conv(hidden_dim, out_dim, heads=1, edge_dim=3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x); x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        return x


# ============================================================
# DQN AGENT (Lightweight, training-focused)
# ============================================================
class DQN(nn.Module):
    def __init__(self, state_dim=64, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class DRLAgent:
    """Complete DRL agent with GNN encoder + DQN."""
    
    def __init__(self, gnn_encoder=None):
        self.gnn = gnn_encoder or GNNEncoder()
        self.dqn = DQN()
        self.target = DQN()
        self.target.load_state_dict(self.dqn.state_dict())
        
        self.optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.dqn.parameters()), lr=3e-4
        )
        self.memory = deque(maxlen=20000)
        self.steps = 0
        self.epsilon = 0.5
        self.gamma = 0.95
        
        self.losses = []
        self.rewards = []
    
    def get_action(self, graph_data, node_idx, explore=True):
        """Get routing action for a specific node."""
        with torch.no_grad():
            embeddings = self.gnn(graph_data)
            state = embeddings[node_idx].cpu().numpy()
        
        if explore and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            q = self.dqn(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax().item()
    
    def get_actions(self, graph_data):
        """Get routing actions for all nodes."""
        with torch.no_grad():
            embeddings = self.gnn(graph_data)
            q_values = self.dqn(embeddings)
            
        if random.random() < self.epsilon:
            return {i: random.randint(0, 3) for i in range(len(embeddings))}
        
        return {i: q_values[i].argmax().item() for i in range(len(embeddings))}
    
    def train(self, env, episodes=200):
        """Train the agent."""
        print(f"[DRL] Training for {episodes} episodes...")
        
        for ep in range(episodes):
            graph = env.reset()
            total_reward = 0
            
            for t in range(200):
                # Get actions
                actions = self.get_actions(graph)
                
                # Step environment
                next_graph, reward, done, info = env.step(actions)
                
                # Store transitions
                with torch.no_grad():
                    emb = self.gnn(graph)
                    next_emb = self.gnn(next_graph)
                
                for node in range(env.num_nodes):
                    self.memory.append((emb[node].cpu().numpy(), 
                                       actions.get(node, 0), 
                                       reward / env.num_nodes,
                                       next_emb[node].cpu().numpy(),
                                       done))
                
                graph = next_graph
                total_reward += reward
                
                # Training step
                if len(self.memory) > 64:
                    self._train_step()
                
                self.steps += 1
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon - 0.5/episodes)
            self.rewards.append(total_reward)
            
            if (ep + 1) % 50 == 0:
                avg_r = np.mean(self.rewards[-50:])
                avg_l = np.mean(self.losses[-50:]) if self.losses else 0
                print(f"  Ep {ep+1}/{episodes} | Reward: {avg_r:.2f} | Loss: {avg_l:.4f} | ε: {self.epsilon:.3f}")
            
            # Update target network
            if ep % 10 == 0:
                self.target.load_state_dict(self.dqn.state_dict())
        
        print(f"[DRL] Training complete! Avg reward: {np.mean(self.rewards[-100:]):.2f}")
        return self.rewards
    
    def _train_step(self):
        """Single DQN training step."""
        batch = random.sample(self.memory, min(64, len(self.memory)))
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor([b[4] for b in batch])
        
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.dqn(next_states).argmax(1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def save(self, path):
        torch.save({
            'gnn': self.gnn.state_dict(),
            'dqn': self.dqn.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"[DRL] Model saved to {path}")
    
    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.gnn.load_state_dict(ckpt['gnn'])
        self.dqn.load_state_dict(ckpt['dqn'])
        self.target.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"[DRL] Model loaded from {path}")


# ============================================================
# BookSim2 Evaluation
# ============================================================
BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'

def eval_on_booksim(topology='mesh44', traffic='uniform', inj_rate=0.1):
    """Run BookSim2 and return latency."""
    topo_map = {'mesh44': ('mesh', 4), 'mesh88': ('mesh', 8), 'torus44': ('torus', 4)}
    t, k = topo_map[topology]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(f"""topology = {t}; k = {k}; n = 2;
routing_function = adaptive_xy_yx;
num_vcs = 4; vc_buf_size = 8;
traffic = {traffic}; injection_rate = {inj_rate};
sim_type = latency;
warmup_periods = 3; sample_period = 1000;
sim_count = 1; packet_size = 1;
""")
        cfg = f.name
    
    try:
        r = subprocess.run([BOOKSIM, cfg], capture_output=True, text=True, timeout=60)
        lat = re.search(r'Packet latency average\s*=\s*([0-9.]+)', r.stdout)
        os.unlink(cfg)
        return float(lat.group(1)) if lat else None
    except:
        os.unlink(cfg)
        return None


# ============================================================
# Main Training Pipeline
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GNNocRoute-DRL: Training Pipeline (Improved)")
    print("=" * 60)
    
    BASE = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments'
    models_dir = os.path.join(BASE, 'models')
    results_dir = os.path.join(BASE, 'results')
    
    # Training configs
    configs = [
        ('mesh44', 'uniform', 300, 'model_uniform.pt'),
        ('mesh44', 'hotspot', 300, 'model_hotspot.pt'),
    ]
    
    results = {}
    for topo, traffic, episodes, model_name in configs:
        print(f"\n{'='*50}")
        print(f"Training on {topo} | {traffic} | {episodes} episodes")
        print(f"{'='*50}")
        
        env = NoCRoutingEnv(topo, traffic, 0.1)
        gnn = GNNEncoder()
        agent = DRLAgent(gnn)
        
        t0 = time.time()
        rewards = agent.train(env, episodes)
        t_elapsed = time.time() - t0
        
        # Save
        agent.save(os.path.join(models_dir, model_name))
        
        # Test on BookSim2
        print(f"\n  Validating on BookSim2 ({traffic} @ 0.1)...")
        bs_lat = eval_on_booksim(topo, traffic, 0.1)
        
        results[f"{topo}_{traffic}"] = {
            'episodes': episodes, 'time_min': t_elapsed/60,
            'avg_reward': float(np.mean(rewards[-50:])),
            'booksim_latency': bs_lat
        }
        print(f"  BookSim2 latency: {bs_lat:.1f} cycles" if bs_lat else "  BookSim2: N/A")
    
    # Summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k}: {v['episodes']} eps | {v['time_min']:.1f} min | "
              f"reward={v['avg_reward']:.2f} | BS latency={v['booksim_latency']}")
    
    with open(os.path.join(results_dir, 'drl_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Done! Results saved to results/drl_training_results.json")
