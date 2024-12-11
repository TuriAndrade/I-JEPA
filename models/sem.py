from torch import nn
import torch.nn.functional as F


class SEM(nn.Module):
    def __init__(self, L, V, tau, **kwargs):
        super().__init__()
        self.L = L
        self.V = V
        self.tau = tau

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        assert dim == self.L * self.V, "Invalid dim for SEM."

        logits = x.view(batch_size * seq_len, self.L, self.V)

        return F.softmax(logits / self.tau, -1).view(batch_size, seq_len, dim)
