# RL-from-Scratch

> Reinforcement learning algorithms implemented from scratch in PyTorch. No Stable-Baselines3, no RLlib — every algorithm written line by line.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-in--progress-orange.svg)]()

**rl-from-scratch** is a learning project: implementing classic and modern RL algorithms from first principles, tested on Gymnasium and MuJoCo environments.

Goal: understand RL deeply by building it, not by importing it.

---

## Algorithms

### Value-based

| Algorithm | Status | Environment | Notes |
|---|---|---|---|
| Q-Learning | ✅ | FrozenLake, Taxi | Tabular |
| DQN | ✅ | CartPole | Replay buffer, target network |
| Double DQN | ✅ | CartPole | Reduces overestimation |
| Dueling DQN | 🚧 | CartPole | Value + advantage streams |
| Rainbow | 📋 | Atari | Planned |

### Policy-based

| Algorithm | Status | Environment | Notes |
|---|---|---|---|
| REINFORCE | ✅ | CartPole | Monte Carlo policy gradient |
| A2C | ✅ | CartPole, LunarLander | Advantage actor-critic |
| PPO | ✅ | LunarLander, MuJoCo | Clipped surrogate objective |
| TRPO | 📋 | MuJoCo | Trust region (planned) |

### Off-policy actor-critic

| Algorithm | Status | Environment | Notes |
|---|---|---|---|
| DDPG | ✅ | Pendulum | Deterministic policy |
| TD3 | ✅ | MuJoCo | Twin critics, delayed updates |
| SAC | ✅ | MuJoCo | Maximum entropy RL |

### Model-based

| Algorithm | Status | Environment | Notes |
|---|---|---|---|
| Dreamer V3 | 🚧 | DMControl | Latent dynamics |
| TD-MPC2 | 📋 | DMControl | Planning + learning |
| World Models | 📋 | CarRacing | VAE + RNN |
---

## Installation

```bash
git clone https://github.com/nsdmlk/rl-from-scratch.git
cd rl-from-scratch
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, Gymnasium, MuJoCo.

---

## Quick start

```python
from rl_from_scratch.ppo import PPO
from rl_from_scratch.envs import make_env

env = make_env("LunarLander-v2")
agent = PPO(env, lr=3e-4, n_steps=2048, batch_size=64)
agent.train(total_steps=500_000)
agent.save("ppo_lunarlander.pt")
```

Evaluation:

```python
agent.load("ppo_lunarlander.pt")
returns = agent.evaluate(n_episodes=100)
print(f"Mean return: {returns.mean():.1f}")
```

---

## Project structure

```
rl_from_scratch/
├── algorithms/
│   ├── q_learning.py
│   ├── dqn.py
│   ├── reinforce.py
│   ├── a2c.py
│   ├── ppo.py
│   ├── ddpg.py
│   ├── td3.py
│   ├── sac.py
│   └── dreamer.py
├── networks/
│   ├── mlp.py
│   ├── cnn.py
│   └── actor_critic.py
├── buffers/
│   ├── replay_buffer.py
│   └── rollout_buffer.py
├── envs/
│   ├── wrappers.py
│   └── make_env.py
└── utils/
    ├── logging.py
    └── seeding.py
```

---

## Design principles

1. **Readable over fast.** Every algorithm is written for clarity first, performance second.
2. **No magic.** No Stable-Baselines3 imports. No hidden wrappers.
3. **Tested.** Each algorithm has a test that runs a short training and checks the return is above threshold.
4. **Logged.** Every run logs to TensorBoard: returns, losses, entropy, gradient norms.
5. **Reproducible.** Fixed seeds, deterministic where possible.

---

## What I'm learning

- **PPO clipping** — why the ratio constraint stabilizes training.
- **SAC entropy** — how temperature affects exploration.
- **TD3 twin critics** — why overestimation bias matters.
- **Dreamer latent dynamics** — how to learn world models.
- **Reward shaping** — how to design rewards for sparse tasks.

---

## What I'm NOT doing

- **Not chasing SOTA.** Stable-Baselines3 will beat this on every benchmark.
- **Not supporting every environment.** Focused on Gymnasium + MuJoCo.
- **Not production-ready.** This is a learning project.
- **Not publishing papers** from this. This is foundation.

---

## Roadmap

- [ ] Q-Learning
- [ ] DQN, Double DQN
- [ ] REINFORCE, A2C
- [ ] PPO
- [ ] DDPG, TD3, SAC
- [ ] Dueling DQN, Rainbow
- [ ] Dreamer V3
- [ ] TD-MPC2
- [ ] Vectorized environments
- [ ] Multi-GPU training
- [ ] Blog post: "RL from scratch — what I learned"

---

## References

Papers:
- [PPO](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [SAC](https://arxiv.org/abs/1801.01290) — Haarnoja et al., 2018
- [TD3](https://arxiv.org/abs/1802.09477) — Fujimoto et al., 2018
- [Dreamer V3](https://arxiv.org/abs/2301.04104) — Hafner et al., 2023
- [DQN](https://www.nature.com/articles/nature14236) — Mnih et al., 2015

Courses:
- [CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) — Berkeley, Sergey Levine
- [HuggingFace Deep RL](https://huggingface.co/learn/deep-rl-course)

Code references (for comparison):
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — minimalist implementations
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — production library

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgments

Built during undergraduate studies at Beijing Institute of Technology.
