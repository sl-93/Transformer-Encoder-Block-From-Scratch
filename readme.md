# Transformer Encoder Block From Scratch

<p align="center">
  <b>A minimal, educational implementation of a Transformer Encoder Block using PyTorch</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Transformer-From%20Scratch-orange" />
  <img src="https://img.shields.io/badge/Deep%20Learning-LLMs-purple" />
</p>

---

## 📌 Overview

This repository implements a **Transformer Encoder Block from scratch using PyTorch**, with the goal of understanding what happens inside a Transformer rather than treating it as a black-box architecture.

The implementation focuses on the core building blocks introduced in **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**:

* Multi-Head Self-Attention
* Query, Key, and Value projections
* Scaled Dot-Product Attention
* Residual Connections
* Layer Normalization
* Position-wise Feed-Forward Network
* Tensor reshaping and multi-head computation

The project is intentionally kept small and modular so that each operation can be followed directly from the input tensor to the final encoder output.

---

## 🧠 Transformer Encoder Block

A Transformer Encoder Block can be summarized as:

```text
                    Input X
                       │
                       ▼
             ┌───────────────────┐
             │ Multi-Head         │
             │ Self-Attention     │
             └─────────┬─────────┘
                       │
                       ▼
                Residual Add
                       │
                       ▼
                 LayerNorm
                       │
                       ▼
             ┌───────────────────┐
             │ Feed-Forward      │
             │ Network (FFN)     │
             └─────────┬─────────┘
                       │
                       ▼
                Residual Add
                       │
                       ▼
                 LayerNorm
                       │
                       ▼
                    Output
```

Conceptually:

$$
X \rightarrow MHA(X) \rightarrow Add+Norm
\rightarrow FFN \rightarrow Add+Norm
$$

---

## 🔍 What Happens Inside?

### 1. Input

The encoder receives a sequence of token representations:

$$
X \in \mathbb{R}^{B \times S \times D}
$$

where:

* \(B\) = batch size
* \(S\) = sequence length
* \(D\) = model dimension (`d_model`)

For example:

```text
X = (32, 100, 512)
```

means:

```text
32   → batch size
100  → sequence length
512  → embedding/model dimension
```

---

### 2. Query, Key, and Value

The input is projected into three representations:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

These projections allow the model to learn different roles for the same token representation.

* **Query (Q):** What information am I looking for?
* **Key (K):** What information do I contain / how should I be matched?
* **Value (V):** What information should actually be passed forward?

---

### 3. Multi-Head Attention

Instead of performing one large attention operation, the model divides the representation into multiple attention heads.

For:

```text
d_model = 512
num_heads = 8
```

each head operates on:

$$
d_{head} = \frac{512}{8} = 64
$$

After reshaping:

```text
Before splitting heads:

Q → (B, S, 512)

After splitting heads:

Q → (B, 8, S, 64)
```

The same transformation is applied to `K` and `V`.

---

### 4. Scaled Dot-Product Attention

Each attention head calculates:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

The implementation follows these steps:

```text
Q @ Kᵀ
      ↓
Scale by √dₖ
      ↓
Softmax
      ↓
Attention Weights
      ↓
Weights @ V
      ↓
Attention Output
```

For example:

```text
Q          → (B, H, S, Dₕ)
K          → (B, H, S, Dₕ)

Q @ Kᵀ     → (B, H, S, S)

Weights @ V
           → (B, H, S, Dₕ)
```

This is the core mechanism that allows every token to interact with other tokens in the sequence.

---

## 🔄 Residual Connections

After attention, the original input is added back:

$$
X' = X + Attention(X)
$$

This is called a **residual connection**.

It provides a direct path for information and gradients through the network and makes deeper Transformer architectures easier to optimize.

---

## 📏 Layer Normalization

The residual output is normalized:

$$
Y = LayerNorm(X')
$$

LayerNorm operates across the feature dimension of each token representation.

The simplified process is:

$$
\mu = \frac{1}{D}\sum_i x_i
$$

$$
\sigma^2 =
\frac{1}{D}\sum_i(x_i-\mu)^2
$$

$$
\hat{x} =
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

followed by learnable scale and shift parameters.

---

## ⚡ Feed-Forward Network

After attention, each token independently passes through a feed-forward network:

$$
FFN(x)
=
W_2\,ReLU(W_1x+b_1)+b_2
$$

A typical configuration is:

```text
d_model = 512
d_ff    = 2048
```

So the transformation becomes:

```text
512
 ↓
2048
 ↓
512
```

An important distinction is:

> **Attention mixes information between tokens, while the FFN transforms each token representation independently.**

---

## 🔁 Complete Encoder Block

The complete computation can therefore be represented as:

$$
X_1 = LayerNorm(X + MHA(X))
$$

$$
X_2 = LayerNorm(X_1 + FFN(X_1))
$$

where:

* `MHA` = Multi-Head Attention
* `FFN` = Feed-Forward Network

The final output has the same shape as the input:

```text
Input  → (B, S, D)
Output → (B, S, D)
```

This makes Transformer blocks composable: multiple blocks can be stacked on top of each other.

---

## 🏗️ Project Structure

```text
Transformer-Encoder-Block-From-Scratch/
│
├── blocks/
│   ├── attention.py
│   ├── feed_forward.py
│   └── transformer_encoder_block.py
│
├── encoder.py
├── main.py
└── README.md
```

The implementation is deliberately separated into smaller components so that the flow of data through the architecture is easy to inspect.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sl-93/Transformer-Encoder-Block-From-Scratch.git
cd Transformer-Encoder-Block-From-Scratch
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch
```

### 4. Run the project

```bash
python main.py
```

---

## 📐 Example Tensor Flow

For:

```text
Batch Size  = 32
Sequence    = 100
d_model     = 512
Heads       = 8
d_head      = 64
```

the main tensor transformations are:

```text
Input
(B, S, D)
    │
    ▼
Q, K, V projections
(B, S, 512)
    │
    ▼
Split into heads
(B, 8, 100, 64)
    │
    ▼
Q @ Kᵀ
(B, 8, 100, 100)
    │
    ▼
Softmax
(B, 8, 100, 100)
    │
    ▼
Attention × V
(B, 8, 100, 64)
    │
    ▼
Merge heads
(B, 100, 512)
    │
    ▼
Residual + LayerNorm
(B, 100, 512)
    │
    ▼
Feed-Forward Network
(B, 100, 512)
    │
    ▼
Residual + LayerNorm
(B, 100, 512)
```

Understanding these shape transformations is one of the main goals of this project.

---

## 🎯 Learning Objectives

This project was built to develop a deeper understanding of Transformer internals by implementing the architecture directly rather than relying on high-level Transformer APIs.

By studying this repository, you can understand:

* How Q, K, and V are generated
* Why Q and K are transposed during attention
* Why attention scores have shape `(S, S)`
* Why attention is scaled by \(\sqrt{d_k}\)
* How multiple attention heads work
* How tensors are reshaped and transposed
* How residual connections work
* What LayerNorm actually does
* Why the FFN expands and contracts the representation
* How the different components form a complete Transformer block

---

## 📚 Reference

This implementation is based on the Transformer architecture introduced in:

> Vaswani et al., **"Attention Is All You Need"**, 2017.

The original paper introduced the Transformer architecture and the self-attention mechanism that forms the foundation of modern Transformer-based language models.

📄 Paper: https://arxiv.org/abs/1706.03762

---

## 🔮 Future Improvements

This repository currently focuses on understanding a single Transformer Encoder Block.

Possible extensions include:

* [ ] Add positional encoding
* [ ] Implement a complete Transformer Encoder
* [ ] Stack multiple encoder blocks
* [ ] Add dropout
* [ ] Add attention visualization
* [ ] Implement the original sinusoidal positional encoding
* [ ] Implement RoPE
* [ ] Implement a Transformer Decoder
* [ ] Implement causal self-attention
* [ ] Build a complete encoder-decoder Transformer
* [ ] Train a small Transformer on a real dataset
* [ ] Extend the implementation toward a small language model

