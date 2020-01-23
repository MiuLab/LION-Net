import torch
import torch.nn as nn


class Attn(nn.Module):
    def __init__(self, method, k_dim, q_dim, v_dim):
        super(Attn, self).__init__()
        self.method = method
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        if self.method == "mul":
            self.q_proj = nn.Linear(self.q_dim, self.k_dim)
            self.k_fact = torch.sqrt(torch.Tensor([self.k_dim]))
        elif self.method == "add":
            self.a_dim = v_dim
            self.k_proj = nn.Linear(self.k_dim, self.a_dim)
            self.q_proj = nn.Linear(self.q_dim, self.a_dim)
            self.a_proj = nn.Linear(self.a_dim, 1)
        if self.method != "none":
            self._o_dim = self.v_dim
        else:
            self._o_dim = 0

    def forward(self, k, v, q, m=None):
        """
            All inputs are supposed to be float tensors.

            k:                  batch_size * seq_len * k_dim
            v:                  batch_size * seq_len * v_dim
            q:                  batch_size * q_dim
            m (optional):       batch_size * seq_len
        """
        if self.method != "none":
            energy = self.score(q, k)
            if m is not None:
                energy.masked_fill_(float('-inf'))
            score = nn.Softmax(dim=1)(energy)
            attn = (score.unsqueeze(2) * v).sum(dim=1, keepdim=True)
            return score, attn
        else:
            score = torch.zeros(
                v.size(0), v.size(1), device=v.device)
            attn = torch.FloatTensor(
                v.size(0), 1, self._o_dim, device=v.device)
            return score, attn

    def score(self, q, k):
        if self.method == 'mul':
            q = self.q_proj(q).unsqueeze(2)
            energy = torch.bmm(v, q).squeeze(2)
            # re-scaled version
            energy = energy / self.k_fact
        elif self.method == "add":
            k = self.k_proj(k)
            q = self.q_proj(q).unsqueeze(1)
            hidden = k + q
            energy = self.a_proj(hidden.tanh()).squeeze(2)
        return energy

    @property
    def o_dim(self):
        return self._o_dim
