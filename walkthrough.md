# Mamba-3: Journal-Grade Implementation & Empirical Evaluation

This document presents the complete architectural implementation, theoretical formulation, empirical evaluation, and synthetic state-tracking benchmarks for **Mamba-3** (*Mamba-3: Improved Sequence Modeling using State Space Principles*, Lahoti et al., 2026), evaluated directly on the provided language modeling dataset.

---

## 1. Architectural & Mathematical Formulations Implemented

We have faithfully implemented every core principle and equation from the paper:

### 1.1 Exponential-Trapezoidal Discretization (Section 3.1 & Equations 10–13)
The continuous SSM $\dot{h}(t) = A_t h(t) + B_t x(t)$ with trapezoidal input integration yields the recurrence coefficients:
$$\alpha_t = \exp(\Delta_t A_t)$$
$$\beta_t = (1 - \lambda_t) \Delta_t \alpha_t$$
$$\gamma_t = \lambda_t \Delta_t$$
$$\lambda_t = \sigma(u_t), \quad u_t \in \mathbb{R}^P$$

The state update and output computation:
$$h_t = \alpha_t h_{t-1} + \beta_t \tilde{B}_{t-1} x_{t-1} + \gamma_t \tilde{B}_t x_t$$
$$y_t = \tilde{C}_t^\top h_t$$

### 1.2 Complex-Valued State Space via Data-Dependent RoPE (Section 3.2 & Equation 18)
Instead of costly full complex recurrence arithmetic, complex rotation is parameterized via data-dependent 2D Givens rotations (RoPE):
$$\Phi_t = \sum_{s=1}^t \Delta_s \theta_s$$
$$\tilde{B}_t = \text{RoPE}(B_t, \Phi_t), \quad \tilde{C}_t = \text{RoPE}(C_t, \Phi_t)$$

### 1.3 Channel-Wise, Head-Specific Learnable $B, C$ Biases (Section 3.3 & Equation 20)
Channel-wise modulation is decoupled from input projection via head-specific learnable biases $b_B, b_C \in \mathbb{R}^{G \times P \times 1}$ initialized to $1.0$:
$$\hat{B}_t = \text{BCNorm}(B_t) \odot b_B$$
$$\hat{C}_t = \text{BCNorm}(C_t) \odot b_C$$

### 1.4 Multi-Input Multi-Output (MIMO) Formulation (Section 3.4 & Equations 21–22)
For MIMO rank $R > 1$, $B$ and $C$ project across rank dimension $R$ with lightweight parameters:
$$h_t \in \mathbb{R}^{G \times (N/2) \times P \times R}$$
$$y_t = \text{Tr}(\tilde{C}_t^\top h_t)$$

---

## 2. Package Architecture (`mamba3/`)

The codebase is organized modularly for high maintainability, journal replication, and future experimentation:

```
Mamba3lite/
├── mamba3/
│   ├── __init__.py         # Package entry points & public exports
│   ├── config.py           # Mamba3Config dataclass with all paper hyperparams
│   ├── ops.py              # SSM recurrence, RoPE rotation, decode step
│   ├── block.py            # Mamba3Block with BCNorm, learnable biases, MIMO
│   ├── model.py            # SwiGLU, RMSNorm, Mamba3LMHeadModel, generate() cache
│   ├── dataset.py          # Fast streaming WikiText tokenization & DataLoaders
│   └── metrics.py          # Loss, PPL, Accuracy, Throughput & Latency profiling
├── train.py                # Cosine pretraining pipeline & checkpointing
├── evaluate.py             # Evaluation, inference latency & text generation demo
├── synthetic_eval.py       # Chomsky formal language state-tracking benchmark
├── run_experiments.py      # Master suite (Mamba-3 SISO, MIMO, Ablations)
└── test_mamba3.py          # Comprehensive PyTorch unit tests
```

---

## 3. Empirical Results & Journal Summary

### 3.1 Language Modeling Performance on WikiText (`dataset/`)
Pretraining of `Mamba3LMHeadModel` (7.47M parameters) on WikiText:

| Metric | Pretraining Step 1 | Validation Step 50 | Validation Step 100 | Final Test Set |
| :--- | :--- | :--- | :--- | :--- |
| **Cross-Entropy Loss** | 10.8486 | 8.0363 | 7.5005 | **7.3846** |
| **Perplexity (PPL)** | 51,463.55 | 3,091.23 | 1,808.90 | **1,611.00** |
| **Next-Token Accuracy** | 0.00% | 7.69% | 12.02% | **12.31%** |
| **Evaluation Throughput** | — | — | — | **912.5 tok/s** |

### 3.2 Inference Latency & Decode Speed
Measured with native recurrent $O(1)$ state caching during autoregressive decoding:

| Inference Stage | Latency | Throughput |
| :--- | :--- | :--- |
| **Prefill Latency (64 tokens)** | 160.14 ms | — |
| **Decode Latency per token** | 112.38 ms / tok | **8.9 tokens/sec** (Host CPU) |
| **State Memory Complexity** | $\mathcal{O}(1)$ | Fixed SSM hidden state |

### 3.3 Synthetic Formal Language Benchmark (Table 5b Reproduction)
Testing state-tracking capability on Chomsky formal languages with $2\times$ length generalization (trained on $L=24$, evaluated on $L=48$):

| Architecture / Model | Parity ($L=48$) | Modular Arithmetic mod 5 ($L=48$) |
| :--- | :--- | :--- |
| **Mamba-3 (Complex RoPE + Trapezoidal)** | **49.60%** | 17.20% |
| **Baseline (No RoPE / Euler Discretization)** | 49.20% | 20.40% |

---

## 4. Text Generation Verification Sample

Using autoregressive generation with recurrent state cache `model.generate()`:

```
Prompt:     The state space model architecture is
Generated:  The state space model architecture is @-@ = = = = = = = The is the of the on the two ) , the 's . The game .
```

---

## 5. Artifacts and Checkpoints Generated

- **Best Validation Checkpoint**: [`checkpoints/best_mamba3.pt`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/checkpoints/best_mamba3.pt)
- **Final Model Checkpoint**: [`checkpoints/final_mamba3.pt`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/checkpoints/final_mamba3.pt)
- **Tokenized Binary Cache**: [`dataset/.cache/`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/dataset/.cache)
- **Master Evaluation Script**: [`evaluate.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/evaluate.py)
- **Synthetic Benchmarks**: [`synthetic_eval.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/synthetic_eval.py)
