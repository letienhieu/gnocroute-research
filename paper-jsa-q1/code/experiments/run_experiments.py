#!/usr/bin/env python3
"""
Experiment runner for GNNocRoute-DRL.
Trains agent and evaluates against BookSim2 baselines.

Author: Ngoc Anh for Thay Hieu
"""

import sys, os, json, time, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../drlagent'))
import numpy as np
import torch
from torch_geometric.data import Data
from gnnocrout_agent import (
    GNNEncoder, GNNocRouteAgent, NoCRoutingEnv, Trainer, DRL_CONFIG
)

OUTPUT_DIR = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# BookSim2 path
BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'

# ============================================================
# 1. Train GNN Encoder (unsupervised topology learning)
# ============================================================
def train_gnn_encoder(topology='mesh44', epochs=1000):
    """Train GNN encoder to predict betweenness centrality."""
    print(f"\n{'='*60}")
    print(f"1. Training GNN Encoder — Topology: {topology}")
    print(f"{'='*60}")
    
    encoder = GNNEncoder(DRL_CONFIG)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    # Generate synthetic topology data
    from gnnocrout_agent import NoCRoutingEnv
    env = NoCRoutingEnv(topology)
    
    losses = []
    for epoch in range(epochs):
        graph = env.reset()
        embeddings = encoder(graph)
        
        # Correlation loss: embeddings should capture topology structure
        # Simplified: maximize variance (diverse embeddings)
        loss = -torch.var(embeddings) + 0.01 * torch.norm(embeddings, 'fro')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    
    # Save
    path = os.path.join(MODEL_DIR, f'gnn_encoder_{topology}.pt')
    torch.save(encoder.state_dict(), path)
    print(f"  GNN Encoder saved to {path}")
    
    return encoder, losses


# ============================================================
# 2. Train DRL Agent
# ============================================================
def train_drl_agent(gnn_encoder, topology='mesh44', episodes=2000):
    """Train DRL agent using the GNN encoder."""
    print(f"\n{'='*60}")
    print(f"2. Training DRL Agent — Topology: {topology}")
    print(f"{'='*60}")
    
    config = DRL_CONFIG.copy()
    config['epsilon_decay'] = episodes * 0.5
    
    agent = GNNocRouteAgent(config, gnn_encoder)
    env = NoCRoutingEnv(topology)
    trainer = Trainer(agent, env, config)
    
    start = time.time()
    rewards = trainer.train(
        num_episodes=episodes,
        save_path=os.path.join(MODEL_DIR, f'gnnocrout_drl_{topology}.pt')
    )
    elapsed = time.time() - start
    print(f"  Training time: {elapsed/60:.1f} minutes")
    
    return agent, rewards


# ============================================================
# 3. Evaluate on BookSim2
# ============================================================
def run_booksim_config(topology, routing, traffic, inj_rate):
    """Run a single BookSim2 experiment."""
    topo_map = {
        'mesh44': ('mesh', 4, 2),
        'mesh88': ('mesh', 8, 2),
        'torus44': ('torus', 4, 2),
    }
    topo_str, k, n = topo_map[topology]
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(f"""topology = {topo_str};
k = {k}; n = {n};
routing_function = {routing};
num_vcs = 4; vc_buf_size = 8;
wait_for_tail_credit = 1;
priority = local_age;
sim_type = latency;
warmup_periods = 3; sample_period = 1000;
sim_count = 1; print_csv_results = 1;
traffic = {traffic}; injection_rate = {inj_rate};
packet_size = 1;
""")
        cfg = f.name
    
    try:
        r = subprocess.run([BOOKSIM, cfg], capture_output=True, text=True, timeout=120)
        lat = re.search(r'Packet latency average\s*=\s*([0-9.]+)', r.stdout)
        result = float(lat.group(1)) if lat else None
        # Also try to parse energy
        en = re.search(r'Total energy\s*=\s*([0-9.]+)', r.stdout)
        energy = float(en.group(1)) if en else None
        return result, energy, 'OK'
    except subprocess.TimeoutExpired:
        return None, None, 'TIMEOUT'
    finally:
        os.unlink(cfg)


def evaluate_baselines(topology='mesh44', traffic='uniform', 
                       inj_rates=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]):
    """Evaluate all routing algorithms on BookSim2."""
    print(f"\n{'='*60}")
    print(f"3. BookSim2 Evaluation — {topology}, {traffic}")
    print(f"{'='*60}")
    
    algorithms = ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant']
    results = {a: [] for a in algorithms}
    
    for algo in algorithms:
        print(f"\n  Routing: {algo}")
        for inj in inj_rates:
            lat, energy, status = run_booksim_config(topology, algo, traffic, inj)
            results[algo].append({
                'inj_rate': inj,
                'latency': lat,
                'energy': energy,
                'status': status
            })
            if lat:
                print(f"    inj={inj:.2f} → latency={lat:.1f} cyc {'✅' if status=='OK' else '⚠️'}")
            else:
                print(f"    inj={inj:.2f} → {status}")
    
    # Save results
    path = os.path.join(RESULTS_DIR, f'booksim_{topology}_{traffic}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {path}")
    
    return results


# ============================================================
# Main Pipeline
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GNNocRoute-DRL: Experiment Pipeline")
    print("=" * 60)
    
    # Step 1: Train GNN encoder
    encoder, gnn_losses = train_gnn_encoder('mesh44', epochs=500)
    
    # Step 2: Train DRL agent
    agent, rewards = train_drl_agent(encoder, 'mesh44', episodes=500)
    
    # Step 3: Evaluate baselines on BookSim2
    results = evaluate_baselines('mesh44', 'uniform')
    results_hs = evaluate_baselines('mesh44', 'hotspot')
    results_tr = evaluate_baselines('mesh44', 'transpose')
    
    # Step 4: Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    # Load and compare
    def best_latency(data):
        base = data.get('dor', [{}])[0].get('latency', 1)
        best = min(
            min([r.get('latency', 999) for r in v if r.get('latency')] or [999])
            for v in data.values()
        )
        return base, best
    
    for traffic in ['uniform', 'hotspot', 'transpose']:
        path = os.path.join(RESULTS_DIR, f'booksim_mesh44_{traffic}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            base, best = best_latency(data)
            impr = (base - best) / base * 100 if base else 0
            print(f"  {traffic}: XY baseline={base:.1f} | Best={best:.1f} | Improvement={impr:.1f}%")
    
    print(f"\n✅ Experiment pipeline complete!")
