import torch
from torch import nn
import torch.nn.functional as F
from .utils import (
    trunc_normal_,
)


class MINE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        init_std=0.02,
        ma_et=1.0,
        ma_rate=0.01,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.init_std = init_std
        self.ma_et = ma_et
        self.ma_rate = ma_rate

        nn.init.normal_(self.fc1.weight, std=self.init_std)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=self.init_std)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=self.init_std)
        nn.init.constant_(self.fc3.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def f(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

    def mutual_information(self, joint, marginal):
        t = self.f(joint)
        et = torch.exp(self.f(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def forward(self, joint, marginal):
        mi_lb, t, et = self.mutual_information(joint, marginal)
        self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * torch.mean(et)

        # unbiasing use moving average
        loss = -(torch.mean(t) - (1 / self.ma_et.mean()).detach() * torch.mean(et))

        return loss, mi_lb
