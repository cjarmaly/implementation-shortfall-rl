import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np

class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that splits the observation into
    1. scalar features: [shares_norm, time_norm, current_vol] (fist three dimensions)
    2. vol history sequence: remaining dims, treated as a length-T sequence

    The scalar features pass through a linear layer.
    The vol history passes through a GRU.
    Outputs are concatenated into a single feature vector.
    """

    def __init__(self, observation_space: gym.Space, gru_hidden_size: int = 32):
        self.n_scalar = 3
        self.vol_history_length = observation_space.shape[0] - self.n_scalar
        features_dim = self.n_scalar + gru_hidden_size

        super().__init__(observation_space, features_dim=features_dim)

        # scalar features pass through a linear layer
        self.scalar_net = nn.Linear(self.n_scalar, self.n_scalar)

        # vol history passes through a GRU
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden_size,
            batch_first=True,
            num_layers=1
        )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """

        # Split observation
        scalars = observations[:, :self.n_scalar]
        vol_history = observations[:, self.n_scalar:]

        # Process scalar features
        scalar_features = self.scalar_net(scalars)

        # Process vol history
        vol_sequence = vol_history.unsqueeze(-1)
        _, hidden = self.gru(vol_sequence)
        gru_features = hidden.squeeze(0)

        # Concatenate outputs
        return torch.cat([scalar_features, gru_features], dim=1)

class GRUActorCriticPolicy(ActorCriticPolicy):
    """
    Custom actor-critic policy that uses the GRU feature extractor, as opposed to the default MLP.
    Pass as the policy argument to any SB3 on-policy algorithm (PPO).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            **kwargs,
            features_extractor_class=GRUFeatureExtractor,
            features_extractor_kwargs={"gru_hidden_size": 32},
            )