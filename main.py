from blocks.encoder import TransformerEncoderBlock
import torch

# Model configuration
vocab_size = 50000
d_model = 512
num_heads = 8
d_ff = 2048
max_seq_len = 512


# Create model
model = TransformerEncoderBlock(vocab_size=vocab_size,
                                d_model=d_model,
                                num_heads=num_heads,
                                d_ff=d_ff,
                                max_seq_len=max_seq_len)

# Fake tokenized input
input_ids = torch.randint(0,
                          vocab_size,
                          (32, 100))

# Forward pass
output = model(input_ids)

print("Input shape :", input_ids.shape)
print("Output shape:", output.shape)