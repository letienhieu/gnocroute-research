#!/usr/bin/env python3
"""
GNNocRoute-DRL Training Script # RUNNING IN BACKGROUND
Log: /home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments/training_log.txt
"""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../drlagent'))
import numpy as np
import torch
from gnnocrout_agent import GNNEncoder, GNNocRouteAgent, NoCRoutingEnv, Trainer, DRL_CONFIG

BASE = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments'
LOG = os.path.join(BASE, 'training_log.txt')
MODEL_DIR = os.path.join(BASE, 'models')
DATA_DIR = os.path.join(BASE, 'results')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    with open(LOG, 'a') as f:
        f.write(f'[{ts}] {msg}\n')
    print(f'[{ts}] {msg}')

# ===== STEP 1: Load/Train GNN Encoder =====
log('=== PHASE 1: GNN Encoder Training ===')
encoder_path = os.path.join(MODEL_DIR, 'gnn_encoder_mesh44.pt')
if os.path.exists(encoder_path):
    encoder = GNNEncoder(DRL_CONFIG)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    log(f'Loaded GNN encoder from {encoder_path}')
else:
    encoder = GNNEncoder(DRL_CONFIG)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    env = NoCRoutingEnv('mesh44')
    for epoch in range(500):
        graph = env.reset()
        embeddings = encoder(graph)
        loss = -torch.var(embeddings) + 0.01 * torch.norm(embeddings, 'fro')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch+1) % 100 == 0:
            log(f'  Encoder epoch {epoch+1}/500 | Loss: {loss.item():.4f}')
    torch.save(encoder.state_dict(), encoder_path)
    log(f'GNN encoder saved → {encoder_path}')

# ===== STEP 2: Train DRL Agent (multiple topologies) =====
log('\n=== PHASE 2: DRL Agent Training ===')

configs = [
    ('mesh44', 500,  'gnnocrout_drl_mesh44.pt'),
    ('mesh88', 300,  'gnnocrout_drl_mesh88.pt'),
]

results = {}
for topo, episodes, model_name in configs:
    log(f'\n--- Training on {topo} ({episodes} episodes) ---')
    t0 = time.time()
    
    agent = GNNocRouteAgent(DRL_CONFIG, encoder)
    env = NoCRoutingEnv(topo)
    trainer = Trainer(agent, env, DRL_CONFIG)
    
    rewards = trainer.train(
        num_episodes=episodes,
        save_path=os.path.join(MODEL_DIR, model_name)
    )
    
    elapsed = time.time() - t0
    avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    log(f'  Completed {episodes} episodes in {elapsed/60:.1f} min')
    log(f'  Avg reward (last 100): {avg_reward:.3f}')
    
    results[topo] = {
        'episodes': episodes,
        'time_min': elapsed/60,
        'avg_reward': float(avg_reward),
        'min_reward': float(min(rewards)),
        'max_reward': float(max(rewards)),
    }

# ===== STEP 3: Summary =====
log('\n=== TRAINING SUMMARY ===')
for topo, r in results.items():
    log(f'  {topo}: {r["episodes"]} eps | {r["time_min"]:.1f} min | '
        f'avg_reward={r["avg_reward"]:.3f} [{r["min_reward"]:.3f}→{r["max_reward"]:.3f}]')

with open(os.path.join(DATA_DIR, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

log('\n✅ Training complete!')
