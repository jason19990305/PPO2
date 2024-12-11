# PPO2 (Proximal Policy Optimization)

A PyTorch implementation of PPO (Proximal Policy Optimization) algorithm for continuous action spaces.

## Features

- Continuous action space support
- State normalization
- Reward scaling
- Generalized Advantage Estimation (GAE)
- Orthogonal initialization
- Learning rate decay
- Policy entropy coefficient
- Configurable neural network architecture

## Project Structure

- `continuous.py`: Main PPO implementation including Actor, Critic and Agent classes
- `normalization.py`: State normalization and reward scaling implementations
- `replaybuffer.py`: Replay buffer for storing transitions

## Key Components

- Actor-Critic Architecture
- Running Mean/Std Normalization
- Replay Buffer
- PPO with Clipped Objective
- GAE for Advantage Estimation

## Usage

1. Import the required classes:

```
from continuous import PPOAgent
from normalization import StateNormalizer, RewardScaler

2. Create environment and agent:

```python
env = gym.make('YourEnvironment-v0')
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dim=256
)
```

3. Train the agent:
```python
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
    
    if episode % update_frequency == 0:
        agent.update()
```

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- PyTorch >= 1.8.0
- Gym >= 0.18.0
- NumPy >= 1.19.0

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## Contributing

Pull requests and issues are welcome!
