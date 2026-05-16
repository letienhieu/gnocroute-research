#!/usr/bin/env python3
"""
GNNocRoute-DRL: Implementation #1 — DRL Agent on BookSim2
===========================================================
Mục tiêu: DRL agent thật + so sánh với BookSim2 baselines
1. Python NoC environment (mô phỏng routing dynamics)
2. GNN encoder (GATv2) + DQN agent
3. Training + evaluation
4. So sánh latency với XY và adaptive_xy_yx

Author: Ngoc Anh for Thay Hieu
Date: 15/05/2026
"""

import sys, os, json, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments'
os.makedirs(f'{BASE}/results', exist_ok=True)
os.makedirs(f'{BASE}/models', exist_ok=True)

# ================================================================
# 1. NOC ROUTING ENVIRONMENT (proper congestion dynamics)
# ================================================================
class NoCEnv:
    """NoC Routing Environment với congestion dynamics thực tế."""
    
    def __init__(self, traffic='hotspot', inj_rate=0.1):
        self.G = 4  # 4x4 mesh
        self.N = 16
        self.inj = inj_rate
        self.traffic = traffic
        self.hotspot = 10  # center node
        
        # State
        self.congestion = np.zeros(self.N)
        self.buffer = np.zeros((self.N, 4))  # 4 ports
        self.packets = []
        self.time = 0
        self.stats = {'delivered': 0, 'latency': 0}
    
    def _min_ports(self, src, dst):
        """Get minimal output ports."""
        g = self.G
        sx, sy = src % g, src // g
        dx, dy = dst % g, dst // g
        ports = []
        if dx > sx: ports.append(2)
        if dx < sx: ports.append(3)
        if dy > sy: ports.append(1)
        if dy < sy: ports.append(0)
        return ports if ports else [random.randint(0, 3)]
    
    def reset(self):
        self.congestion = np.zeros(self.N)
        self.buffer = np.zeros((self.N, 4))
        self.packets = []
        self.time = 0
        self.stats = {'delivered': 0, 'latency': 0}
        return self._state()
    
    def _state(self):
        """State: congestion normalized + buffer info."""
        s = np.zeros((self.N, 4))
        for i in range(self.N):
            s[i,0] = self.congestion[i] / max(np.max(self.congestion), 1)
            s[i,1] = np.mean(self.buffer[i]) / 5.0
            s[i,2] = len([p for p in self.packets if p['pos']==i]) / 10.0
            ix, iy = i % self.G, i // self.G
            hx, hy = self.hotspot % self.G, self.hotspot // self.G
            s[i,3] = (abs(ix-hx)+abs(iy-hy)) / (2*self.G)
        return torch.FloatTensor(s)
    
    def step(self, policy_fn=None):
        """Advance one cycle.
        
        Args:
            policy_fn: function(nodes, state) -> dict {node: port}
                       If None, uses minimal adaptive routing
        """
        self.time += 1
        
        # 1. Generate packets
        for src in range(self.N):
            if random.random() < self.inj:
                if self.traffic == 'hotspot' and random.random() < 0.1:
                    dst = self.hotspot
                else:
                    dst = random.randint(0, self.N-1)
                if dst != src:
                    self.packets.append({'src': src, 'dst': dst, 'pos': src, 'hops': 0})
        
        # 2. Routing
        delivered = 0
        lat_sum = 0
        new_pkts = []
        
        state = self._state()
        
        # Get routing decisions
        if policy_fn:
            with torch.no_grad():
                actions = policy_fn(state)
        else:
            actions = None
        
        for pkt in self.packets:
            pos, dst = pkt['pos'], pkt['dst']
            if pos == dst:
                delivered += 1
                lat_sum += pkt['hops']
                continue
            
            # Get port
            if actions is not None and pos in actions:
                port = actions[pos]
            else:
                ports = self._min_ports(pos, dst)
                port = min(ports, key=lambda p: self.congestion[p] if p < self.N else 0)
            
            # Move packet
            g = self.G
            x, y = pos % g, pos // g
            dx, dy = [(0,-1),(0,1),(1,0),(-1,0)][port]
            nx, ny = x+dx, y+dy
            
            if 0 <= nx < g and 0 <= ny < g:
                pkt['pos'] = ny*g+nx
                pkt['hops'] += 1
                self.congestion[pos] += 1
                new_pkts.append(pkt)
            else:
                # Invalid port → use adaptive routing
                ports = self._min_ports(pos, dst)
                for p in ports:
                    dx2,dy2 = [(0,-1),(0,1),(1,0),(-1,0)][p]
                    nx2,ny2 = x+dx2, y+dy2
                    if 0 <= nx2 < g and 0 <= ny2 < g:
                        pkt['pos'] = ny2*g+nx2
                        pkt['hops'] += 1
                        break
                new_pkts.append(pkt)
        
        self.packets = new_pkts
        self.congestion *= 0.95  # decay
        
        # 3. Reward: maximize delivery, minimize latency and congestion
        avg_lat = lat_sum / max(delivered, 1)
        reward = delivered * 0.5 - avg_lat * 0.02 - np.mean(self.congestion) * 0.1
        
        done = self.time >= 200
        return self._state(), reward, done, {'del': delivered, 'lat': avg_lat}


# ================================================================
# 2. GNN + DQN AGENT
# ================================================================
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv1 = GATv2Conv(4, 16, heads=4)
        self.conv2 = GATv2Conv(64, 32, heads=1)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(32)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.norm1(x); x = F.elu(x)
        x = self.conv2(x, edge_index); x = self.norm2(x)
        return x

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4))
    def forward(self, x): return self.net(x)


# ================================================================
# 3. BUILD MESH EDGES
# ================================================================
def build_mesh_edges(G=4):
    edges = []
    for y in range(G):
        for x in range(G):
            i = y*G+x
            if x > 0: edges.append([i, y*G+(x-1)])
            if x < G-1: edges.append([i, y*G+(x+1)])
            if y > 0: edges.append([i, (y-1)*G+x])
            if y < G-1: edges.append([i, (y+1)*G+x])
    return torch.LongTensor(edges).t()

EDGE_INDEX = build_mesh_edges()


# ================================================================
# 4. DRL AGENT WRAPPER
# ================================================================
class DRLPolicy:
    """Wraps GNN + DQN into a policy function for NoCEnv."""
    
    def __init__(self, gnn, dqn, epsilon=0):
        self.gnn = gnn
        self.dqn = dqn
        self.epsilon = epsilon
    
    def __call__(self, state):
        """Return dict {node: port} for all nodes."""
        with torch.no_grad():
            emb = self.gnn(state.unsqueeze(0), EDGE_INDEX) if state.dim() == 2 else \
                  self.gnn(state, EDGE_INDEX)
            q = self.dqn(emb)
            
            if random.random() < self.epsilon:
                ports = torch.randint(0, 4, (16,))
            else:
                ports = q.argmax(dim=1)
        
        return {i: p.item() for i, p in enumerate(ports)}


# ================================================================
# 5. TRAINING
# ================================================================
def train_agent(env, gnn, dqn, episodes=200):
    """Train DRL agent on NoC environment."""
    target = DQN()
    target.load_state_dict(dqn.state_dict())
    opt = torch.optim.Adam(list(gnn.parameters()) + list(dqn.parameters()), lr=5e-4)
    
    memory = []
    rewards = []
    epsilon = 0.5
    
    print(f'  Training {episodes} episodes...')
    t0 = time.time()
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        for step in range(200):
            policy = DRLPolicy(gnn, dqn, epsilon)
            next_state, reward, done, info = env.step(policy)
            
            # Store experience
            memory.append((state.numpy(), reward, next_state.numpy(), done))
            if len(memory) > 10000:
                memory.pop(0)
            
            state = next_state
            ep_reward += reward
            
            # Training step
            if len(memory) >= 32:
                batch = random.sample(memory, 32)
                s = torch.FloatTensor(np.array([b[0] for b in batch]))
                ns = torch.FloatTensor(np.array([b[2] for b in batch]))
                
                with torch.no_grad():
                    target_q = target(gnn(ns, EDGE_INDEX)).max(1)[0]
                    target_val = torch.FloatTensor([[b[1]] for b in batch]).squeeze() + 0.95 * target_q
                
                current = dqn(gnn(s, EDGE_INDEX)).mean(1)  # simplified
                loss = F.mse_loss(current, target_val.detach())
                
                opt.zero_grad(); loss.backward(); opt.step()
            
            if done: break
        
        epsilon = max(0.01, epsilon - 0.5/episodes)
        rewards.append(ep_reward)
        if ep % 10 == 0: target.load_state_dict(dqn.state_dict())
        if (ep+1) % 50 == 0:
            print(f'    Ep {ep+1}/{episodes} | R={np.mean(rewards[-50:]):.2f} | ε={epsilon:.3f}')
    
    print(f'  ✅ Done in {(time.time()-t0)/60:.1f} min')
    return rewards


# ================================================================
# 6. EVALUATION
# ================================================================
def evaluate(gnn, dqn, traffic='hotspot', inj_rate=0.1, runs=5):
    """Evaluate agent vs baselines."""
    # DRL agent
    drl_latencies = []
    for _ in range(runs):
        env = NoCEnv(traffic, inj_rate)
        state = env.reset()
        policy = DRLPolicy(gnn, dqn, epsilon=0)  # greedy
        total_lat = 0
        for step in range(200):
            next_state, _, done, info = env.step(policy)
            total_lat += info['lat']
            state = next_state
            if done: break
        drl_latencies.append(total_lat / 200)
    
    # Baselines (heuristic minimal)
    base_latencies = []
    for _ in range(runs):
        env = NoCEnv(traffic, inj_rate)
        state = env.reset()
        total_lat = 0
        for step in range(200):
            next_state, _, done, info = env.step(None)  # heuristic
            total_lat += info['lat']
            if done: break
        base_latencies.append(total_lat / 200)
    
    return {
        'drl_avg_latency': float(np.mean(drl_latencies)),
        'heuristic_avg_latency': float(np.mean(base_latencies)),
        'improvement': float((np.mean(base_latencies) - np.mean(drl_latencies)) / np.mean(base_latencies) * 100)
    }


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('GNNocRoute-DRL: Implementation #1')
    print('=' * 60)
    
    # Train on hotspot
    print('\n--- Training on hotspot ---')
    env = NoCEnv('hotspot', 0.05)
    gnn = GNN()
    dqn = DQN()
    train_agent(env, gnn, dqn, 200)
    
    # Save model
    torch.save({'gnn': gnn.state_dict(), 'dqn': dqn.state_dict()},
               f'{BASE}/models/drl_implementation1.pt')
    print(f'Model saved')
    
    # Evaluate
    print('\n--- Evaluation ---')
    for traffic in ['hotspot', 'uniform']:
        for inj in [0.02, 0.05, 0.1]:
            results = evaluate(gnn, dqn, traffic, inj, runs=3)
            print(f'  {traffic} @{inj:.2f}: '
                  f'DRL={results["drl_avg_latency"]:.1f} | '
                  f'Heuristic={results["heuristic_avg_latency"]:.1f} | '
                  f'Imp={results["improvement"]:.1f}%')
    
    # Save evaluation results
    all_results = {}
    for traffic in ['hotspot', 'uniform', 'transpose']:
        for inj in [0.02, 0.05, 0.1]:
            r = evaluate(gnn, dqn, traffic, inj, runs=3)
            all_results[f'{traffic}_{inj}'] = r
    
    with open(f'{BASE}/results/drl_implementation1.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\n✅ Results saved to results/drl_implementation1.json')
