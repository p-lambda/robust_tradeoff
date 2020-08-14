import torch


class LinfAug(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, x):
        noise = torch.rand_like(x) * 2 * self.epsilon - self.epsilon
        return torch.clamp(x + noise, 0.0, 1.0)
