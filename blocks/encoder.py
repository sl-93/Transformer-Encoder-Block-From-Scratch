import torch.nn as nn
from blocks.positional_encoding import PositionalEncoding
from blocks.attention import SelfAttention
from blocks.feed_forward import FeedForward

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 max_seq_len=5000):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size,
                                      d_model)

        # Positional Embedding
        self.positional_encoding = PositionalEncoding(d_model,
                                                      max_seq_len)

        # Attention
        self.attention = SelfAttention(d_model,
                                       num_heads)

        # Feed Forward
        self.ffn = FeedForward(d_model,
                               d_ff)

        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, 
                input_ids):

        # ---------------------------------------
        # Token IDs → Embeddings
        # ---------------------------------------
        x = self.embedding(input_ids)

        # ---------------------------------------
        # Add positional information
        # ---------------------------------------
        x = self.positional_encoding(x)

        # Self-attention + residual + norm
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)

        # FFN + residual + norm
        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        return x