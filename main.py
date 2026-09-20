from encoder import TransformerEncoderBlock
import torch

x = torch.randn(2,
                10,
                512)

block = TransformerEncoderBlock(d_model=512,
                                num_heads=8,
                                d_ff=2048)

output = block(x)

print(x.shape)
print(output.shape)