# Transformer & GPT From Scratch

A hands-on PyTorch implementation of Transformer components built **from scratch for learning and understanding Large Language Models (LLMs)**.

This project started with a Transformer Encoder Block and has evolved into a small **GPT-style decoder-only language model**, covering the core concepts behind modern autoregressive LLMs.

The goal is not to build a production-scale LLM, but to understand what happens **inside a Transformer — from token embeddings and attention all the way to training and text generation.**

---

## 🚀 What This Project Covers

### Transformer Foundations

* Token embeddings
* Positional encoding
* Query, Key, and Value projections
* Scaled dot-product attention
* Multi-head self-attention
* Residual connections
* Layer Normalization
* Feed-forward networks
* Transformer Encoder Block

### GPT / Decoder-Only Architecture

* Causal self-attention
* Causal masking
* Transformer Decoder Block
* Stacked decoder blocks
* Final LayerNorm
* Language Model (LM) head
* Next-token prediction
* Autoregressive text generation

### Training

* Character-level tokenization
* Input/target sequence creation
* Shifted targets for next-token prediction
* Cross-entropy loss
* Backpropagation
* AdamW optimization
* Mini-batch training

---

## 🧠 Architecture

The project currently contains both the original Transformer encoder components and a GPT-style decoder-only model.

### Encoder Block

The encoder block follows the classical Transformer structure:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Residual + LayerNorm
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual + LayerNorm
  │
  ▼
Output
```

### GPT-Style Decoder Block

The decoder block replaces standard self-attention with **causal self-attention**, preventing each token from attending to future tokens.

```text
Input
  │
  ▼
Causal Self-Attention
  │
  ▼
Residual + LayerNorm
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual + LayerNorm
  │
  ▼
Output
```

### Complete GPT Model

```text
Token IDs
   │
   ▼
Token Embedding
   │
   ▼
Positional Encoding
   │
   ▼
Decoder Block × N
   │
   ▼
Final LayerNorm
   │
   ▼
Language Model Head
   │
   ▼
Logits
   │
   ▼
Next Token
```

The current `GPTModel` implements token embeddings, positional encoding, a stack of decoder blocks, final normalization, and a vocabulary projection through the LM head.

---

## 🔍 Causal Self-Attention

The GPT implementation uses a causal attention mask so that a token cannot see future tokens.

For a sequence of length 5:

```text
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

The attention scores are masked before applying softmax:

```python
scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)
```

Since:

```text
softmax(-∞) = 0
```

future tokens receive zero attention probability.

The implementation performs:

```text
Q = XWq
K = XWk
V = XWv

        ↓

QKᵀ / √dk

        ↓

Causal Mask

        ↓

Softmax

        ↓

Attention Weights × V

        ↓

Merge Heads

        ↓

Output Projection
```

The current implementation explicitly constructs the lower-triangular causal mask and applies it before softmax.

---

## 🎯 Next-Token Prediction

The model is trained as an autoregressive language model.

Given:

```text
hello machine learning
```

the training examples are shifted by one token:

```text
Input:

hello machine learning
```

```text
Target:

machine learning ...
```

Conceptually:

```text
"I"              → "love"
"I love"         → "machine"
"I love machine" → "learning"
```

For a model output of:

```text
(B, S, vocab_size)
```

the target tensor has shape:

```text
(B, S)
```

Each target value is simply the ID of the correct next token.

---

## 📐 Tensor Shapes

For example, with:

```text
Batch size       = 2
Sequence length  = 6
d_model          = 512
Number of heads  = 8
Vocabulary size  = 50,000
```

the main tensors have the following shapes:

```text
Input IDs
(B, S)
(2, 6)

       ↓

Token Embeddings
(B, S, d_model)
(2, 6, 512)

       ↓

Q / K / V
(B, S, d_model)

       ↓

Split Heads
(B, num_heads, S, head_dim)

(2, 8, 6, 64)

       ↓

Attention Scores
(B, num_heads, S, S)

(2, 8, 6, 6)

       ↓

Attention Output
(B, num_heads, S, head_dim)

(2, 8, 6, 64)

       ↓

Merge Heads
(B, S, d_model)

(2, 6, 512)

       ↓

LM Head

(B, S, vocab_size)

(2, 6, 50,000)
```

The final tensor contains a vocabulary score for every position in every sequence.

---

## 📉 Training

The project includes a complete training loop.

The basic training process is:

```text
Input IDs
    │
    ▼
GPT Model
    │
    ▼
Logits
    │
    ▼
Cross-Entropy Loss
    │
    ▼
Backward Pass
    │
    ▼
Gradients
    │
    ▼
AdamW
    │
    ▼
Updated Parameters
```

The core training step is:

```python
input_ids, targets = get_batch(batch_size)

logits = model(input_ids)

loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1)
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

For example:

```text
logits:
(B, S, V)
    ↓
(B × S, V)

targets:
(B, S)
    ↓
(B × S)
```

This allows `CrossEntropyLoss` to treat every token position as an individual next-token prediction problem.

---

## 🧪 Dataset

The current example uses a small character-level dataset:

```text
hello world
hello machine learning
hello transformer
machine learning is powerful
transformers are powerful
machine learning is interesting
```

A simple character tokenizer converts the text into integer token IDs.

For example:

```text
text
 ↓
characters
 ↓
integer IDs
 ↓
embedding
 ↓
Transformer
```

The character-level tokenizer is intentionally simple so that the focus remains on understanding the Transformer architecture and training process rather than on implementing a sophisticated tokenizer.

---

## ✍️ Text Generation

After training, the model can generate text autoregressively.

For example:

```text
Prompt:

hello
```

The model predicts one token:

```text
hello m
```

Then the new token is added to the context:

```text
hello m...
```

and the model predicts another token.

This continues until the requested number of new tokens has been generated.

The generation process therefore looks like:

```text
Prompt
  │
  ▼
GPT
  │
  ▼
Predict next token
  │
  ▼
Append token
  │
  ▼
GPT again
  │
  ▼
Predict next token
  │
  ▼
...
```

The implementation takes the logits corresponding to the **last sequence position**, converts them to probabilities with softmax, samples the next token, and appends it to the sequence.

---

## 📁 Project Structure

```text
Transformer-Encoder-Block-From-Scratch/
│
├── blocks/
│   ├── attention.py
│   ├── causal_attention.py
│   ├── decoder.py
│   ├── feed_forward.py
│   ├── positional_encoding.py
│   ├── encoder.py
│   └── simple_GPT.py
│
├── main.py
└── README.md
```

### `blocks/attention.py`

Implements standard self-attention and multi-head attention components.

### `blocks/causal_attention.py`

Implements causal self-attention for autoregressive language modeling, including:

* Q/K/V projections
* Multi-head splitting
* Scaled dot-product attention
* Causal masking
* Softmax
* Weighted values
* Head merging

### `blocks/decoder.py`

Implements the GPT-style Transformer Decoder Block:

```text
Causal Self-Attention
        ↓
Residual + LayerNorm
        ↓
Feed Forward
        ↓
Residual + LayerNorm
```

### `blocks/feed_forward.py`

Implements the Transformer feed-forward network:

```text
d_model
   ↓
d_ff
   ↓
ReLU
   ↓
d_model
```

### `blocks/positional_encoding.py`

Implements sinusoidal positional encoding.

### `blocks/simple_GPT.py`

Combines the components into a complete GPT-style model:

```text
Embedding
→ Positional Encoding
→ Decoder Blocks
→ LayerNorm
→ LM Head
```

### `main.py`

Contains the end-to-end example:

* Dataset
* Character tokenizer
* Batch creation
* Model initialization
* Training loop
* Cross-entropy loss
* Backpropagation
* AdamW
* Text generation

The current `main.py` uses a small character-level dataset, a 2-layer model with `d_model=128`, 4 attention heads, `d_ff=512`, and trains for 1000 steps.

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/sl-93/Transformer-Encoder-Block-From-Scratch.git
cd Transformer-Encoder-Block-From-Scratch
```

### 2. Install PyTorch

```bash
pip install torch
```

### 3. Run the training script

```bash
python main.py
```

You should see output similar to:

```text
Device: cpu
Vocabulary size: ...
Dataset size: ...

Step    0 | Loss: ...
Step  100 | Loss: ...
Step  200 | Loss: ...
...
Step  900 | Loss: ...

Generated text:
hello ...
```

The exact loss and generated text will vary because the model is randomly initialized and the batches are sampled randomly.

---

## 📚 Learning Progression

This repository is being developed incrementally to understand Transformers from the inside out.

### Day 1 — Attention

* Q, K, V
* Scaled dot-product attention
* Attention scores
* Attention weights

### Day 2 — Transformer Encoder

* Multi-head attention
* Residual connections
* LayerNorm
* Feed-forward networks
* Encoder block

### Day 3 — Transformer Input

* Token IDs
* Embeddings
* Positional encoding
* Transformer input pipeline

### Day 4 — Causal Attention

* Decoder-only architecture
* Causal masking
* Autoregressive prediction
* GPT-style attention

### Day 5 — GPT Training

* Decoder blocks
* GPT model
* Character-level tokenization
* Input/target shifting
* Logits
* Cross-entropy
* Backpropagation
* AdamW
* Text generation

### Next Steps

Planned improvements include:

* [ ] Train on a larger text corpus
* [ ] Implement a better tokenizer
* [ ] Add temperature sampling
* [ ] Add top-k sampling
* [ ] Add model checkpointing
* [ ] Add validation loss
* [ ] Add train/validation split
* [ ] Implement Pre-LN architecture
* [ ] Implement RMSNorm
* [ ] Implement RoPE
* [ ] Implement KV cache
* [ ] Add attention visualization
* [ ] Add model evaluation
* [ ] Experiment with larger GPT configurations

---

## 📖 References

* Vaswani et al., **Attention Is All You Need**
* Radford et al., **Improving Language Understanding by Generative Pre-Training**
* Brown et al., **Language Models are Few-Shot Learners**
* Andrej Karpathy, **Let's build GPT from scratch**
* Sebastian Raschka, **Build a Large Language Model (From Scratch)**

---

## 🎯 Purpose

This project is primarily an **educational implementation**.

The code intentionally favors:

* readability
* explicit tensor operations
* simple architecture
* minimal abstraction
* understanding over performance

The goal is to answer questions such as:

> What exactly happens to a token after it enters a Transformer?

> How are Q, K, and V calculated?

> How does causal masking work?

> Why does the model output `(B, S, vocab_size)` logits?

> How are the targets constructed?

> How is cross-entropy calculated?

> How does backpropagation update the Transformer parameters?

> How does the same model go from training to autoregressive generation?

Rather than treating the Transformer as a black box, this repository builds it piece by piece to make its internal mechanics visible.
