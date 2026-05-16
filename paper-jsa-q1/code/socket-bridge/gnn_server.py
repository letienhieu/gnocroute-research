#!/usr/bin/env python3
"""
GNNocRoute-DRL: Python Socket Server — Phase 1
===============================================
Nhận state từ BookSim2 → GNN encoder → PPO policy → Routing table → BookSim2

Usage:
  python3 gnn_server.py --port 9999
  (BookSim2 cần include gnn_bridge.cpp và gọi gnn_init_socket("127.0.0.1", 9999))
"""

import sys, os, json, socket, threading, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# ================================================================
# GNN Encoder (lightweight)
# ================================================================
class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv1 = GATv2Conv(4, 16, heads=4)
        self.conv2 = GATv2Conv(64, 32, heads=1)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(32)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.norm1(x); x = F.elu(x)
        x = self.conv2(x, edge_index); x = self.norm2(x); return x

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

# ================================================================
# GNN-PPO Policy Engine
# ================================================================
class GNNPolicyEngine:
    def __init__(self, num_nodes):
        self.N = num_nodes
        self.G = int(np.sqrt(num_nodes))
        self.edge_index = build_mesh_edges(self.G)
        self.gnn = GNNEncoder()
        
        # PPO model (simplified: MLP policy)
        self.policy = nn.Sequential(
            nn.Linear(32 + 4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4)  # N, S, E, W preference
        )
        self.optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.policy.parameters()), lr=5e-4)
        
        # Experience buffer for online learning
        self.memory = deque(maxlen=10000)
        self.episode = 0
        
        # Pre-train GNN on topology
        self._pretrain_gnn()
        print(f'  [GNN-PPO] Engine ready: {num_nodes} nodes ({self.G}x{self.G})')
    
    def _pretrain_gnn(self, epochs=300):
        opt = torch.optim.Adam(self.gnn.parameters(), lr=1e-3)
        for ep in range(epochs):
            x = torch.randn(self.N, 4)
            emb = self.gnn(x, self.edge_index)
            loss = -torch.var(emb) + 0.01 * torch.norm(emb)
            opt.zero_grad(); loss.backward(); opt.step()
    
    def compute_routing_table(self, congestion):
        """Compute routing table from network state.
        
        Args:
            congestion: numpy array of shape (N,) - congestion per node
        
        Returns:
            routing_table: numpy array (N, N) - 0=XY, 1=YX
        """
        # Build state features
        x = np.zeros((self.N, 4))
        for i in range(self.N):
            x[i, 0] = congestion[i] / max(np.max(congestion), 1)
            x[i, 1] = np.mean(congestion[max(0,i-self.G):min(self.N,i+self.G)])
            x[i, 2] = np.var(congestion[max(0,i-self.G):min(self.N,i+self.G)])
            x[i, 3] = i / self.N  # position encoding
        
        # GNN forward
        with torch.no_grad():
            state = torch.FloatTensor(x)
            embeddings = self.gnn(state, self.edge_index).numpy()
        
        # PPO-like decision for each (src, dst)
        table = np.zeros((self.N, self.N), dtype=int)
        for src in range(self.N):
            for dst in range(self.N):
                if src == dst: continue
                
                src_emb = embeddings[src]
                dst_emb = embeddings[dst]
                src_cong = congestion[src]
                dst_cong = congestion[dst]
                dist = abs(src % self.G - dst % self.G) + abs(src // self.G - dst // self.G)
                
                # Decision features
                features = np.concatenate([src_emb, [src_cong, dst_cong, dist / (2*self.G), congestion[src] - congestion[dst]]])
                
                with torch.no_grad():
                    q = self.policy(torch.FloatTensor(features)).numpy()
                
                # Q[N,S,E,W] → prefer XY if N/S preferred, YX if E/W preferred
                if q[2] + q[3] > q[0] + q[1]:  # E/W preference
                    table[src, dst] = 1  # YX
                else:
                    table[src, dst] = 0  # XY
        
        return table
    
    def update_from_experience(self, state, action, reward, next_state):
        """Online PPO update (simplified)."""
        self.memory.append((state, action, reward, next_state))
        self.episode += 1
        
        if len(self.memory) >= 64 and self.episode % 10 == 0:
            # Mini-batch training
            batch = random.sample(list(self.memory), min(64, len(self.memory)))
            # Simplified: just log
            avg_reward = np.mean([b[2] for b in batch])
            print(f'  [PPO] Episode {self.episode}: avg_reward={avg_reward:.3f}')


# ================================================================
# Socket Server
# ================================================================
class GNNSocketServer:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.engine = None
        self.num_nodes = 0
    
    def handle_client(self, conn, addr):
        print(f'  [Server] Connection from {addr}')
        buffer = ''
        
        try:
            while True:
                data = conn.recv(4096)
                if not data: break
                
                buffer += data.decode('utf-8')
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line.startswith('STATE:'):
                        self._handle_state(conn, line)
        except Exception as e:
            print(f'  [Server] Error: {e}')
        finally:
            conn.close()
            print(f'  [Server] Connection closed')
    
    def _handle_state(self, conn, line):
        """Process STATE message from BookSim2."""
        parts = line.split(':')
        if len(parts) < 3: return
        
        num_nodes = int(parts[1])
        congestion_str = parts[2]
        congestion = np.array([float(x) for x in congestion_str.split(',')])
        
        # Initialize engine if needed
        if self.engine is None or self.num_nodes != num_nodes:
            self.num_nodes = num_nodes
            self.engine = GNNPolicyEngine(num_nodes)
            print(f'  [Server] Initialized engine for {num_nodes} nodes')
        
        # Compute routing table
        table = self.engine.compute_routing_table(congestion)
        
        # Send back: "TABLE:<N>:<row0>,<row1>;..."
        response = f'TABLE:{num_nodes}:'
        for i in range(num_nodes):
            response += ','.join(str(table[i, j]) for j in range(num_nodes))
            response += ';'
        response += '\n'
        
        conn.send(response.encode('utf-8'))
    
    def start(self):
        print(f'[GNN Server] Starting on {self.host}:{self.port}')
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        print(f'[GNN Server] Listening...')
        
        try:
            while True:
                conn, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print('\n[GNN Server] Shutting down...')
        finally:
            server.close()


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNNocRoute Socket Server')
    parser.add_argument('--port', type=int, default=9999, help='TCP port')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address')
    args = parser.parse_args()
    
    print('=' * 50)
    print('GNNocRoute-DRL: Socket Server (Phase 1)')
    print('=' * 50)
    print(f'  Host: {args.host}')
    print(f'  Port: {args.port}')
    print()
    
    server = GNNSocketServer(args.host, args.port)
    server.start()
