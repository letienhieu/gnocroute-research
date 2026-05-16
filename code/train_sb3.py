#!/usr/bin/env python3
"""
GNNocRoute-DRL: Stable-Baselines3 DRL Training
===============================================
Dùng stable-baselines3 DQN trên custom Gym environment
- Môi trường NoC routing với congestion dynamics
- State: congestion, buffer, topology
- Action: N/S/E/W port selection
- Reward: delivery - latency - congestion

Author: Ngoc Anh for Thay Hieu
Date: 15/05/2026
"""

import sys, os, time, json, random, math
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments'

# ================================================================
# 1. CUSTOM GYM ENVIRONMENT
# ================================================================
class NoCRoutingEnv(gym.Env):
    """NoC Routing Environment for Gym/Stable-Baselines3.
    
    State: per-router congestion + buffer + position
    Action: output port (0=N, 1=S, 2=E, 3=W)  
    Reward: packet delivery - latency penalty - congestion penalty
    """
    
    def __init__(self, traffic='hotspot', inj_rate=0.05):
        super().__init__()
        self.G = 4  # grid size
        self.N = 16  # number of routers
        self.inj = inj_rate
        self.traffic = traffic
        self.hotspot = self.G * (self.G // 2) + (self.G // 2)  # center
        
        # Action: 4 output ports
        self.action_space = spaces.Discrete(4)
        
        # Observation: congestion(16) + buffer(16) + position(16) = 48
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(48,), dtype=np.float32)
        
        # State
        self.congestion = np.zeros(self.N)
        self.buffer = np.zeros(self.N)
        self.packets = []
        self.time = 0
        self.delivered = 0
        self.latency_sum = 0
    
    def _min_ports(self, src, dst):
        """Get minimal output ports from src to dst."""
        g = self.G
        sx, sy = src % g, src // g
        dx, dy = dst % g, dst // g
        ports = []
        if dx > sx: ports.append(2)  # E
        if dx < sx: ports.append(3)  # W  
        if dy > sy: ports.append(1)  # S
        if dy < sy: ports.append(0)  # N
        return ports if ports else [random.randint(0, 3)]
    
    def seed(self, seed=None):
        """Set seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def reset(self):
        """Reset environment."""
        self.congestion = np.zeros(self.N)
        self.buffer = np.zeros(self.N)
        self.packets = []
        self.time = 0
        self.delivered = 0
        self.latency_sum = 0
        return self._get_obs()
    
    def _get_obs(self):
        """Build observation vector."""
        obs = np.concatenate([
            self.congestion / max(np.max(self.congestion), 1),
            self.buffer / 5.0,
            np.arange(self.N) / self.N  # position encoding
        ])
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute routing action.
        
        The action determines which output port to use for the current packet.
        """
        self.time += 1
        info = {}
        
        # 1. Generate packet (one per step to match single-agent RL)
        src = random.randint(0, self.N - 1)
        if self.traffic == 'hotspot' and random.random() < 0.2:
            dst = self.hotspot
        else:
            dst = random.randint(0, self.N - 1)
        
        if dst == src:
            dst = (dst + 1) % self.N
        
        # 2. Route the packet
        pos = src
        hops = 0
        max_hops = 30
        
        for _ in range(max_hops):
            if pos == dst:
                break
            
            # Get minimal ports
            ports = self._min_ports(pos, dst)
            
            # Use RL action if it's valid, else use heuristic
            if action in ports:
                port = action
            else:
                # Pick best among valid ports
                port = min(ports, key=lambda p: self.congestion[p] if p < self.N else 0)
            
            # Move packet
            g = self.G
            x, y = pos % g, pos // g
            dx, dy = [(0, -1), (0, 1), (1, 0), (-1, 0)][port]
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < g and 0 <= ny < g:
                pos = ny * g + nx
                hops += 1
                self.congestion[pos] += 0.1
            else:
                # Invalid move → try another port
                for p in ports:
                    dx2, dy2 = [(0, -1), (0, 1), (1, 0), (-1, 0)][p]
                    nx2, ny2 = x + dx2, y + dy2
                    if 0 <= nx2 < g and 0 <= ny2 < g:
                        pos = ny2 * g + nx2
                        hops += 1
                        break
        
        # 3. Update state
        self.congestion = self.congestion * 0.95
        self.buffer = self.congestion * 0.5
        
        # 4. Reward
        delivered = 1.0 if pos == dst else 0.0
        reward = delivered * 1.0 - hops * 0.05 - np.mean(self.congestion) * 0.3
        
        if delivered:
            self.delivered += 1
            self.latency_sum += hops
        
        done = self.time >= 500
        info['delivered'] = delivered
        info['hops'] = hops
        info['delivery_rate'] = self.delivered / max(self.time, 1)
        
        return self._get_obs(), reward, done, info


# ================================================================
# 2. DEFINE NEURAL NET ARCHITECTURE 
# ================================================================
class CustomCNN(nn.Module):
    """Custom policy network that uses GNN features."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_space.n),
        )
    
    def forward(self, x, **kwargs):
        return self.net(x)


# ================================================================
# 3. TRAINING WITH STABLE-BASELINES3
# ================================================================
def train_with_sb3(traffic='hotspot', inj_rate=0.05, timesteps=50000):
    """Train DRL agent using stable-baselines3 DQN."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import ProgressBarCallback
    
    print(f'\n{"="*60}')
    print(f'Training DQN on {traffic} @ inj={inj_rate}')
    print(f'Timesteps: {timesteps}')
    print(f'{"="*60}')
    
    # Create environment
    env = NoCRoutingEnv(traffic, inj_rate)
    
    # Create DQN agent
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=0.1,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        verbose=1,
        seed=42,
    )
    
    # Train
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - t0
    
    print(f'\n✅ Training done in {elapsed/60:.1f} min')
    
    # Save
    model_path = f'{BASE}/models/dqn_{traffic}_{str(inj_rate).replace(".","")}'
    model.save(model_path)
    print(f'Model saved: {model_path}')
    
    return model, env, elapsed


# ================================================================
# 4. EVALUATION
# ================================================================
def evaluate_model(model, env, episodes=50):
    """Evaluate trained model."""
    delivery_rates = []
    avg_hops = []
    
    for ep in range(episodes):
        obs = env.reset()
        total_delivered = 0
        total_hops = 0
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_delivered += info.get('delivered', 0)
            total_hops += info.get('hops', 0)
            steps += 1
            if done:
                break
        
        delivery_rates.append(total_delivered / max(steps, 1))
        avg_hops.append(total_hops / max(total_delivered, 1))
    
    return {
        'avg_delivery_rate': float(np.mean(delivery_rates)),
        'avg_hops': float(np.mean(avg_hops)),
        'std_delivery': float(np.std(delivery_rates)),
    }


# ================================================================
# 5. MAIN
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('GNNocRoute-DRL: Stable-Baselines3 Training')
    print('=' * 60)
    
    results = {}
    
    # Train on hotspot
    model, env, elapsed = train_with_sb3('hotspot', 0.05, 50000)
    eval_results = evaluate_model(model, env, 30)
    results['hotspot_005'] = {
        'train_time_min': elapsed/60,
        **eval_results
    }
    print(f'Evaluation: delivery_rate={eval_results["avg_delivery_rate"]:.3f}, hops={eval_results["avg_hops"]:.1f}')
    
    # Train on hotspot @ 0.1
    model2, env2, elapsed2 = train_with_sb3('hotspot', 0.1, 30000)
    eval2 = evaluate_model(model2, env2, 30)
    results['hotspot_01'] = {
        'train_time_min': elapsed2/60,
        **eval2
    }
    
    # Train on uniform
    model3, env3, elapsed3 = train_with_sb3('uniform', 0.05, 30000)
    eval3 = evaluate_model(model3, env3, 30)
    results['uniform_005'] = {
        'train_time_min': elapsed3/60,
        **eval3
    }
    
    # Summary
    print(f'\n{"="*60}')
    print('TRAINING SUMMARY')
    print(f'{"="*60}')
    for k, v in results.items():
        print(f'  {k:15s}: train={v["train_time_min"]:.1f}min | delivery={v["avg_delivery_rate"]:.2f} | hops={v["avg_hops"]:.1f}')
    
    with open(f'{BASE}/results/dqn_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ Results saved to results/dqn_training_results.json')
