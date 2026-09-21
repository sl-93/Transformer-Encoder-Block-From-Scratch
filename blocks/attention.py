import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, 
                 d_model, 
                 num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # --------------------------------
        # Split heads
        # --------------------------------
        Q = Q.view(batch_size,
                   seq_len,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)

        K = K.view(batch_size,
                   seq_len,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)

        V = V.view(batch_size,
                   seq_len,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)

        # --------------------------------
        # Attention scores
        # --------------------------------
        scores = Q @ K.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        # --------------------------------
        # Causal mask
        # --------------------------------
        mask = torch.tril(torch.ones(seq_len,
                                     seq_len,
                                     device=x.device))

        scores = scores.masked_fill(mask == 0,
                                    float("-inf"))

        # --------------------------------
        # Attention weights
        # --------------------------------
        weights = F.softmax(scores, dim=-1)

        # --------------------------------
        # Weighted values
        # --------------------------------
        output = weights @ V

        # --------------------------------
        # Merge heads
        # --------------------------------
        output = output.transpose(1, 2)

        output = output.contiguous().view(batch_size,
                                          seq_len,
                                          self.d_model)

        return self.W_o(output)