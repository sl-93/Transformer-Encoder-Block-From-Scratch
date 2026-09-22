import torch.nn as nn
from blocks.positional_encoding import PositionalEncoding
from blocks.decoder import TransformerDecoderBlock


class GPTModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 d_model,
                 num_heads,
                 d_ff,
                 num_layers,
                 max_seq_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      d_model)

        self.position = PositionalEncoding(d_model,
                                           max_seq_len)

        self.blocks = nn.ModuleList([TransformerDecoderBlock(d_model,
                                                             num_heads,
                                                             d_ff)
                                                             for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model,
                                 vocab_size)

    def forward(self, input_ids):

        # Token embeddings
        x = self.embedding(input_ids)

        # Positional information
        x = self.position(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Vocabulary projection
        logits = self.lm_head(x)

        return logits