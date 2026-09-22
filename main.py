import torch
import torch.nn.functional as F
from blocks.simple_GPT import GPTModel


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. Tiny dataset
# ============================================================

text = """
hello world
hello machine learning
hello transformer
machine learning is powerful
transformers are powerful
machine learning is interesting
hello world
hello machine learning
"""


# ============================================================
# 3. Create a very simple character tokenizer
# ============================================================

# Get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character -> integer
stoi = {ch: i
        for i, ch in enumerate(chars)}

# integer -> character
itos = {i: ch
        for i, ch in enumerate(chars)}


def encode(text):

    return [stoi[ch]
            for ch in text]


def decode(tokens):

    return "".join(itos[token]
                   for token in tokens)


# Convert entire dataset to token IDs
data = torch.tensor(encode(text),
                    dtype=torch.long)

print("Vocabulary size:", vocab_size)
print("Dataset size:", len(data))


# ============================================================
# 4. Create training examples
# ============================================================

block_size = 32


def get_batch(batch_size):

    # Random starting positions
    starts = torch.randint(0,
                           len(data) - block_size,
                           (batch_size,))

    # Input
    x = torch.stack([data[i:i + block_size]
                     for i in starts])

    # Target = input shifted by one
    y = torch.stack([data[i + 1:i + block_size + 1]
                     for i in starts])

    return x.to(device), y.to(device)


# ============================================================
# 5. Model configuration
# ============================================================

model = GPTModel(vocab_size=vocab_size,
                 d_model=128,
                 num_heads=4,
                 d_ff=512,
                 num_layers=2,
                 max_seq_len=block_size)

model = model.to(device)


# ============================================================
# 6. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=3e-4)

# ============================================================
# 7. Training
# ============================================================

num_steps = 1000
batch_size = 16

for step in range(num_steps):

    # --------------------------------------------------------
    # Get training batch
    # --------------------------------------------------------

    input_ids, targets = get_batch(batch_size)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(input_ids)

    # logits:
    # (B, S, vocab_size)

    # targets:
    # (B, S)

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                           targets.reshape(-1))

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    optimizer.zero_grad()
    loss.backward()

    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if step % 100 == 0:

        print(f"Step {step:4d} | "
              f"Loss: {loss.item():.4f}")


# ============================================================
# 8. Generate text
# ============================================================

@torch.no_grad()
def generate(model,
             input_ids,
             max_new_tokens):

    model.eval()

    for _ in range(max_new_tokens):

        # If sequence becomes longer than block_size,
        # keep only the latest tokens.
        input_context = input_ids[:, -block_size:]

        # Forward pass
        logits = model(input_context)

        # Take prediction for the LAST position
        logits = logits[:, -1, :]

        # Convert logits to probabilities
        probabilities = F.softmax(logits,
                                  dim=-1)

        # Sample next token
        next_token = torch.multinomial(probabilities,
                                       num_samples=1)

        # Append new token
        input_ids = torch.cat([input_ids, next_token],
                              dim=1)

    return input_ids


# ============================================================
# 9. Test generation
# ============================================================

prompt = "hello"

prompt_tokens = torch.tensor([encode(prompt)],
                             dtype=torch.long,
                             device=device)

generated_tokens = generate(model,
                            prompt_tokens,
                            max_new_tokens=50)

generated_text = decode(generated_tokens[0].tolist())

print("\nGenerated text:")
print(generated_text)