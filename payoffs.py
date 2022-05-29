import torch
from torch import Tensor


class Payoff:
    def evaluate(self, S, K):
        return Tensor([0])


class Call(Payoff):
    def __init__(self):
        self.type = 'Call'

    def evaluate(self, S, K):
        return torch.maximum(S - K, Tensor([0]))


class Digital(Payoff):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.type = 'Digital'

    def evaluate(self, S, K):
        return torch.minimum(Tensor([1]), torch.maximum(Tensor([0]), (S - K + self.epsilon) / (2 * self.epsilon)))
