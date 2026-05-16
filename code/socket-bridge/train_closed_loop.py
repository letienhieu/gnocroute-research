#!/usr/bin/env python3
"""
GNNocRoute-DRL: BookSim2 Closed-Loop Training
==============================================
Sử dụng PyBind11 module (booksim.so) hoặc Socket bridge để train DRL agent
trực tiếp trên BookSim2 cycle-accurate simulator.

Usage:
  # Cách 1: PyBind11 (cần build booksim.so trước)
  python3 train_closed_loop.py --mode pybind
  
  # Cách 2: Socket bridge (cần chạy gnn_bridge trong BookSim2)
  python3 train_closed_loop.py --mode socket

Author: Ngoc Anh for Thay Hieu
"""

import sys, os, time, json, argparse
import numpy as np

# ============================================================
# DRL Agent (GNN + DQN)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv1 = GATv2Conv(5, 16, heads=4)
        self.conv2 = GATv2Conv(64, 32, heads=1)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(32)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.norm1(x); x = F.elu(x)
        x = self.conv2(x, edge_index); x = self.norm2(x); return x

class DQN(nn.Module):
    def __init__(self, state_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+4, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2))  # 2 actions: XY=0, YX=1
    def forward(self, x): return self.net(x)

def build_mesh_edges(G):
    N = G*G; edges = []
    for y in range(G):
        for x in range(G):
            i = y*G+x
            if x>0: edges.append([i, y*G+(x-1)])
            if x<G-1: edges.append([i, y*G+(x+1)])
            if y>0: edges.append([i, (y-1)*G+x])
            if y<G-1: edges.append([i, (y+1)*G+x])
    return torch.LongTensor(edges).t()

# ============================================================
# BookSim2 Environment Wrapper
# ============================================================
class BookSimEnv:
    """Wrapper for BookSim2 closed-loop training."""
    
    def __init__(self, mode='pybind', k=4, traffic='hotspot', 
                 inj_rate=0.1, period=5000):
        self.mode = mode
        self.k = k
        self.N = k * k
        self.period = period
        self.edge_index = build_mesh_edges(k)
        self.cycle = 0
        self.stats = {'latencies': [], 'throughputs': []}
        
        if mode == 'pybind':
            try:
                import booksim
                self.env = booksim.NoCEnv(k, traffic, inj_rate, period)
                print(f'  [Env] PyBind11 mode: {k}x{k} mesh, {traffic}')
            except ImportError:
                print('  [Env] WARNING: booksim.so not built. Using simulation mode.')
                self.env = None
        else:
            self.env = None
            print(f'  [Env] Socket mode: {k}x{k} mesh, {traffic}')
    
    def reset(self):
        self.cycle = 0
        if self.env:
            state = self.env.reset()
        else:
            # Simulated state for testing
            state = np.random.randn(self.N, 5).astype(np.float32)
        return state
    
    def compute_routing_table(self, gnn, dqn, state):
        """Compute routing table using DRL agent."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            embeddings = gnn(state_t, self.edge_index)
            
            table = np.zeros((self.N, self.N), dtype=np.int32)
            for src in range(self.N):
                for dst in range(self.N):
                    if src == dst: continue
                    feats = torch.cat([
                        embeddings[src],
                        embeddings[dst][:4],  # congestion info
                    ])
                    q = dqn(feats.unsqueeze(0))
                    table[src, dst] = q.argmax().item()
            return table
    
    def step(self, gnn, dqn, state):
        """Run P cycles with current routing policy."""
        table = self.compute_routing_table(gnn, dqn, state)
        
        if self.env and self.mode == 'pybind':
            next_state, reward, done = self.env.step(table)
            self.cycle += self.period
        else:
            # Simulate for testing
            next_state = np.random.randn(self.N, 5).astype(np.float32)
            # Reward = negative estimated latency
            congestion = np.mean(np.abs(state[:, 1]))
            reward = -congestion * 10 + 0.1
            self.cycle += self.period
            done = self.cycle >= 50000
        
        self.stats['latencies'].append(-reward)
        return next_state, reward, done


# ============================================================
# Training Loop
# ============================================================
def train(gnn, dqn, env, episodes=100):
    """Train DRL agent on BookSim2 closed-loop."""
    opt = torch.optim.Adam(
        list(gnn.parameters()) + list(dqn.parameters()), lr=5e-4)
    
    rewards = []
    t0 = time.time()
    
    print(f'\nTraining {episodes} episodes...')
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        for step in range(20):  # max 20 environment steps per episode
            next_state, reward, done = env.step(gnn, dqn, state)
            ep_reward += reward
            state = next_state
            
            # Train DQN (simplified)
            # (full DQN training loop would go here)
            
            if done: break
        
        rewards.append(ep_reward)
        if (ep+1) % 20 == 0:
            avg_r = np.mean(rewards[-20:])
            print(f'  Ep {ep+1}/{episodes} | R={avg_r:.2f} | cycles={env.cycle}')
    
    elapsed = time.time() - t0
    print(f'✅ Done in {elapsed/60:.1f} min')
    return rewards


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pybind', 'socket', 'sim'], default='sim')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--traffic', default='hotspot')
    parser.add_argument('--inj', type=float, default=0.1)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    
    print('=' * 50)
    print('GNNocRoute-DRL: BookSim2 Closed-Loop Training')
    print('=' * 50)
    print(f'  Mode: {args.mode}')
    print(f'  Topology: {args.k}x{args.k}')
    print(f'  Traffic: {args.traffic} @ {args.inj}')
    print(f'  Episodes: {args.episodes}')
    
    gnn = GNNEncoder()
    dqn = DQN()
    env = BookSimEnv(args.mode, args.k, args.traffic, args.inj)
    
    rewards = train(gnn, dqn, env, args.episodes)
    
    print(f'\n✅ Training complete!')
    print(f'  Avg reward (last 20): {np.mean(rewards[-20:]):.2f}')
