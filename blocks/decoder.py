import torch.nn as nn
from blocks.causal_attention import CausalSelfAttention
from blocks.feed_forward import FeedForward

class TransformerDecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = CausalSelfAttention(d_model,
                                             num_heads)

        self.ffn = FeedForward(d_model,
                               d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # 1. Causal self-attention
        attention_output = self.attention(x)

        # 2. Residual + LayerNorm
        x = self.norm1(x + attention_output)

        # 3. Feed Forward
        ffn_output = self.ffn(x)

        # 4. Residual + LayerNorm
        x = self.norm2(x + ffn_output)

        return x