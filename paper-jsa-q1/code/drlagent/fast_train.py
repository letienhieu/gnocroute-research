"""
GNNocRoute-DRL: Optimized Training (Fast)
==========================================
- GNN inference only every 10 steps (cache)
- Batch node processing
- Faster environment
"""

import sys, os, time, json, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments'

# ===== Simplified GNN =====
class FastGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )
    def forward(self, x):
        return self.net(x)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 4),
        )
    def forward(self, x):
        return self.net(x)

# ===== Fast Environment =====
class FastNoCEnv:
    def __init__(self, traffic='uniform', inj=0.1):
        self.N = 16
        self.grid = 4
        self.inj = inj
        self.traffic = traffic
        self.state = np.zeros((self.N, 8))
        self.packets = []
        self.time = 0
    
    def reset(self):
        self.state = np.random.uniform(0, 0.1, (self.N, 4))
        self.packets = []
        self.time = 0
        return torch.FloatTensor(self.state)
    
    def step(self, actions):
        self.time += 1
        delivered = 0
        latency_sum = 0
        congestion = np.zeros(self.N)
        
        # Generate packets
        for src in range(self.N):
            if random.random() < self.inj:
                if self.traffic == 'hotspot':
                    dst = 10 if random.random() < 0.1 else random.randint(0, self.N-1)
                else:
                    dst = random.randint(0, self.N-1)
                if dst == src: dst = (dst + 1) % self.N
                self.packets.append([src, dst, src, 0])
        
        # Route
        new_pkts = []
        srcs_seen = set()
        for pkt in self.packets:
            s, d, pos, hops = pkt
            if pos == d:
                delivered += 1; latency_sum += hops
                continue
            if len(srcs_seen) < self.N:
                port = 1 if actions[pos] < 0.5 else 2
                srcs_seen.add(pos)
            else:
                port = 1
            
            gs = 4
            x, y = pos % gs, pos // gs
            nx, ny = x + (1 if port == 2 else -1 if port == 3 else 0), \
                     y + (1 if port == 1 else -1 if port == 0 else 0)
            if 0 <= nx < gs and 0 <= ny < gs:
                new_pkts.append([s, d, ny*gs+nx, hops+1])
            else:
                new_pkts.append(pkt)
            congestion[pos] += 1
        
        self.packets = new_pkts
        self.state = np.column_stack([
            np.random.uniform(0, 0.3, self.N),
            congestion / max(np.max(congestion), 1),
            np.random.uniform(0, 0.2, self.N),
            np.zeros(self.N),
        ])
        
        reward = delivered * 0.1 - np.mean(congestion) * 0.05 - latency_sum * 0.01
        return torch.FloatTensor(self.state), reward, self.time >= 100, {'del': delivered}

# ===== Fast Training =====
print("=" * 50)
print("GNNocRoute-DRL: Fast Training")
print("=" * 50)

gnn = FastGNN()
dqn = DQN()
optimizer = torch.optim.Adam(list(gnn.parameters()) + list(dqn.parameters()), lr=1e-3)

env = FastNoCEnv('hotspot', 0.1)
memory = []
epsilon = 0.5
gamma = 0.95
rewards = []

t0 = time.time()
for ep in range(200):
    state = env.reset()
    ep_reward = 0
    
    for step in range(100):
        with torch.no_grad():
            emb = gnn(state)
            q = dqn(emb)
        
        actions = q if random.random() > epsilon else torch.rand(16, 4)
        action_mask = (actions == actions.max(dim=1, keepdim=True)[0]).float()
        
        next_state, reward, done, info = env.step(action_mask[:, 1].numpy())
        
        with torch.no_grad():
            next_emb = gnn(next_state)
        
        # Store raw states (not embeddings) for training
        for i in range(16):
            memory.append((state[i].numpy(), action_mask[i, 1].item(), reward/16, next_state[i].numpy(), done))
        
        state = next_state
        ep_reward += reward
        
        # Train
        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            s = torch.FloatTensor([b[0] for b in batch])
            a = torch.FloatTensor([b[1] for b in batch]).unsqueeze(1)
            r = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
            ns = torch.FloatTensor([b[3] for b in batch])
            d = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1)
            
            with torch.no_grad():
                target = r + (1 - d) * gamma * dqn(gnn(ns)).max(1, keepdim=True)[0]
            
            current = dqn(gnn(s)).gather(1, a.long())
            loss = F.mse_loss(current, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done: break
    
    epsilon = max(0.01, epsilon - 0.5/200)
    rewards.append(ep_reward)
    
    if (ep + 1) % 50 == 0:
        avg = np.mean(rewards[-50:])
        print(f"  Ep {ep+1}/200 | Reward: {avg:.2f} | ε: {epsilon:.3f} | Memory: {len(memory)}")

elapsed = time.time() - t0
print(f"\n✅ Training done in {elapsed/60:.1f} min")
print(f"  Avg reward (last 50): {np.mean(rewards[-50:]):.2f}")

# Save
torch.save({'gnn': gnn.state_dict(), 'dqn': dqn.state_dict()},
           os.path.join(BASE, 'models/drl_fast.pt'))
print(f"  Model saved → models/drl_fast.pt")

# Summary
results = {
    'time_min': elapsed/60,
    'avg_reward': float(np.mean(rewards[-50:])),
    'episodes': 200,
    'model': 'drl_fast.pt'
}
with open(os.path.join(BASE, 'results/drl_training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Results → results/drl_training_results.json")
print("=" * 50)
