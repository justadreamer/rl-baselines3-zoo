import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ModifiedNatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(ModifiedNatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]

        # self.cnn = nn.Sequential( #4 x 150 x 75
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),  #32 x 37 x 18
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  #64 x 15 x 8
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), #64 x 13 x 6
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )


        self.cnn = nn.Sequential( #3 x 75 x 150
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #16 x 36 x 74
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 x 17 x 36
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3), # 64 x 7 x 17
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3), # 128 x 2 x 7
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = observation_space.sample()
            n_flatten = self.cnn(th.as_tensor(sample[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
