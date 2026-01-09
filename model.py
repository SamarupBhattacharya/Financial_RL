import torch
import torch.nn as nn
import yfinance as yf
import gymnasium as gym
import gym_anytrading
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


# --- 1. Custom Mamba Feature Extractor ---
class MambaTradingExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, d_model=128, d_state=16):
        super().__init__(observation_space, features_dim)

        # observation_space.shape is (window_size, n_features)
        self.input_dim = observation_space.shape[1]
        self.seq_len = observation_space.shape[0]

        # Project input features to Mamba dimension
        self.embedding = nn.Linear(self.input_dim, d_model)

        # Mamba Core: Selective State Space Model
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension
            d_state=d_state,  # SSM state dimension
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        # Final output head for SB3
        self.fc = nn.Linear(d_model, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        # Mamba expects (Batch, Seq, Dim)
        x = self.embedding(observations)
        x = self.mamba(x)

        # We take the final state representation to make a decision
        x = x[:, -1, :]
        return self.relu(self.fc(x))


# --- 2. Data Loading & Environment Setup ---
def get_env():
    # Fetching Apple stock data
    df = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

    # window_size=20 means the agent looks at the last 20 days of prices to decide
    env = gym.make("stocks-v0", df=df, frame_bound=(20, 500), window_size=20)
    return env


# --- 3. Model Training ---
if __name__ == "__main__":
    env = DummyVecEnv([get_env])

    # Configure the DQN to use our Mamba Extractor
    policy_kwargs = dict(
        features_extractor_class=MambaTradingExtractor,
        features_extractor_kwargs=dict(features_dim=256, d_model=128),
    )

    model = DQN(
        "MlpPolicy",  # Base class
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=500,
        target_update_interval=1000,
        exploration_fraction=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Training Mamba-DQN Agent...")
    model.learn(total_timesteps=15000)
    model.save("mamba_trading_dqn")

    # --- 4. Inference & Visualization ---
    print("\nStarting Inference...")
    test_env = get_env()
    obs, info = test_env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        if terminated or truncated:
            print(f"Inference complete. Final Results: {info}")
            break

    # Plot results
    plt.figure(figsize=(15, 6))
    test_env.render_all()
    plt.title("Trading Agent Results: Mamba-DQN Architecture")
    plt.show()
