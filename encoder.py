import torch.nn as nn
from blocks.attention import SelfAttention
from blocks.feed_forward import FeedForward

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048):
        super().__init__()

        self.attention = SelfAttention(d_model,
                                       num_heads)

        self.ffn = FeedForward(d_model,
                               d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, 
                x):

        # Self-attention + residual + norm
        attention_output = self.attention(x)

        x = self.norm1(x + attention_output)

        # FFN + residual + norm
        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        return x