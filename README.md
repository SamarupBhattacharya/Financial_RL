# Mamba-DQN Trading: Selective State Space Models for Financial RL
This repository implements a Deep Q-Learning (DQN) agent for automated stock trading, utilizing a Mamba (Selective State Space Model) backbone for feature extraction. Unlike traditional architectures that use MLPs or LSTMs, this project leverages the linear-time sequence modeling of Mamba to capture long-horizon dependencies in financial time-series data while maintaining computational efficiency.

## Overview
Financial markets are non-stationary and noisy. Traditional Reinforcement Learning (RL) agents often struggle with "causal confusion"—mistaking random noise for actionable signals. This project addresses this by:
- **Sequential Encoding**: Using a 20-day look-back window processed through a Mamba block.
- **Selective Memory**: Leveraging Mamba’s hardware-aware selection mechanism to focus on high-relevance price movements.
- **End-to-End RL**: Integrating the architecture with Stable-Baselines3 and gym-anytrading for robust policy training.

## Tech StackRL Framework: 
- Stable-Baselines3 (DQN Implementation)
- Sequence Model: Mamba-SSM (Selective State Space Models)
- Environment: Gym-Anytrading
- Data Source: Yahoo Finance (via yfinance)

## Installation:
This project requires CUDA for the official Mamba implementation.Bash# Clone the repository
```python
git clone https://github.com/your-username/mamba-dqn-trading.git
cd mamba-dqn-trading

# Install core dependencies
pip install yfinance gymnasium gym-anytrading stable-baselines3 torch

# Install Mamba (Requires CUDA)
pip install mamba-ssm
```
## Architecture:
The agent utilizes a custom Mamba Feature Extractor. Instead of flattening the input window, the observation $(Batch, Window, Features)$ is passed through a Mamba block. The final hidden state is then fed into the Q-network to predict the optimal action (Buy/Sell).

## Usage
1. **Training the Agent**: To train the agent on historical Apple (AAPL) data:
   ```python
   python model.py
   ```
2. **Evaluation**: The script will automatically run an inference loop post-training, visualizing the agent's "Long" and "Short" positions against the price chart.
   
## Results:
The agent learns to identify trend reversals and support/resistance levels.

## Conclusion:
This project was developed as part of an exploration into data-efficient imitation learning and sequential modeling.
