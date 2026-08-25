# Implementation Plan: Mamba-3 Architecture, Training, and Evaluation on Custom Dataset

## Overview
Implement the complete Mamba-3 architecture and experimental framework exactly as proposed in the base paper (*Mamba-3: Improved Sequence Modeling using State Space Principles*, Lahoti et al., 2026), train and evaluate it on the provided WikiText dataset, verify all paper metrics (Perplexity, Loss, Token Accuracy, State-Tracking capability), and provide an end-to-end reproducible codebase for journal publication.

---

## Key Methodological Components from the Base Paper

1. **Exponential-Trapezoidal Discretization (Section 3.1, Eq. 5 & 6)**:
   - 3-term recurrence: \( h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t \)
   - \( \alpha_t = \exp(\Delta_t A_t) \)
   - \( \beta_t = (1 - \lambda_t) \Delta_t \exp(\Delta_t A_t) \)
   - \( \gamma_t = \lambda_t \Delta_t \)
   - Data-dependent trapezoidal gate: \( \lambda_t = \sigma(u_t) \)
2. **Complex-Valued State Space via Data-Dependent RoPE (Section 3.2, Eq. 11 & 25)**:
   - Complex state transitions discretized as block-diagonal 2D rotations.
   - Data-dependent phase parameter \( \theta_t \), step rotation \( \Theta_t = \Delta_t \theta_t \), cumulative rotation \( \Phi_t = \sum_{s=0}^t \Theta_s \).
   - RoPE applied to \( B_t \) and \( C_t \) representations.
3. **Learnable Channel-wise, Head-specific B & C Biases (Section 3.4 & Appendix F)**:
   - Head-specific learnable biases \( b_B, b_C \) added after BCNorm, initialized to 1.0.
4. **BC / QK Normalization (Section 3.4)**:
   - RMSNorm applied across state dimension on \( B \) and \( C \) projections.
5. **MIMO & SISO Recurrence (Section 3.3 & Appendix C)**:
   - Efficient SISO default (\( R=1 \)) and MIMO rank \( R > 1 \) projection formulations.
6. **Overall LM Backbone (Section 3.4)**:
   - Alternating Mamba-3 blocks and SwiGLU MLP blocks with pre-RMSNorm in a decoder architecture.
7. **Fast Recurrent Inference & Generation**:
   - Constant-memory \( O(1) \) per-token autoregressive generation cache.

---

## Proposed Project Structure

```
Mamba3lite/
├── mamba3/
│   ├── __init__.py
│   ├── ops.py             # Core SSM scan, trapezoidal recurrence, RoPE trick, and chunked SSD
│   ├── block.py           # Mamba-3 layer block (BCNorm, Biases, MIMO/SISO, Gating, Projections)
│   ├── model.py           # Full Causal LM (Embeddings, Alternating Mamba3 + SwiGLU, RMSNorm, LM Head)
│   ├── dataset.py         # Tokenizer and dataset loaders for train.txt, validation.txt, test.txt
│   └── metrics.py         # Perplexity, cross-entropy, token accuracy, FLOPs & latency profiling
├── train.py               # Pretraining script with Cosine Annealing, AdamW, validation loop & checkpointing
├── evaluate.py            # Comprehensive evaluation on validation & test sets (Perplexity, Loss, Sample Generation)
├── synthetic_eval.py      # Chomsky state-tracking benchmark (Parity & Modular Arithmetic from Table 5b)
├── run_experiments.py     # Master runner to train, evaluate, and benchmark Mamba-3 vs baseline variants
├── config.py              # Configuration schemas (model sizes: 180M, 440M, and lightweight journal demo configurations)
└── requirements.txt       # Dependencies list
```

---

## Proposed Changes

### 1. Core Model Architecture (`mamba3/`)
#### [NEW] [`ops.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/mamba3/ops.py)
- Implement sequential and parallel exponential-trapezoidal SSM operations.
- Implement data-dependent rotary position embedding (RoPE trick) for complex state space dynamics.
- Support both SISO and MIMO rank expansion.

#### [NEW] [`block.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/mamba3/block.py)
- Implement `Mamba3Block` with input projections, BCNorm (RMSNorm on \( B, C \)), learnable biases on \( B, C \), trapezoidal gate projection \( \lambda_t = \sigma(u_t) \), complex rotary angles \( \theta_t \), core SSM mixing, and output projection with SiLU gating.
- Implement recurrent step method for zero-overhead inference caching.

#### [NEW] [`model.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/mamba3/model.py)
- Implement `SwiGLU` feedforward block.
- Implement `RMSNorm`.
- Implement `Mamba3LMHeadModel` adhering to Llama-style alternating Mamba-3 + SwiGLU layers with weight tying option and generation methods.

### 2. Dataset and Metrics (`mamba3/`)
#### [NEW] [`dataset.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/mamba3/dataset.py)
- Stream and tokenize `dataset/train.txt`, `dataset/validation.txt`, and `dataset/test.txt`.
- Token chunking, sliding window / non-overlapping batching for autoregressive language modeling.

#### [NEW] [`metrics.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/mamba3/metrics.py)
- Cross-entropy loss, perplexity calculation (\( \exp(\mathcal{L}) \)), bits-per-character/token, accuracy, and generation throughput.

### 3. Execution & Experiment Scripts
#### [NEW] [`train.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/train.py)
- Training loop with AdamW, Cosine LR scheduler with warmup, gradient clipping, evaluation hooks, checkpoint saving, and live metric logging.

#### [NEW] [`evaluate.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/evaluate.py)
- Evaluates test and validation perplexities and tests generation quality with custom prompts.

#### [NEW] [`synthetic_eval.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/synthetic_eval.py)
- Evaluates the formal state-tracking parity and modular arithmetic tasks defined in Section 4.2 / Table 5b.

#### [NEW] [`run_experiments.py`](file:///c:/Users/keert/OneDrive/Keer/SEM%206/Mamba3lite/run_experiments.py)
- Runs a clean comparative workflow (Mamba-3 SISO, Mamba-3 MIMO, Mamba-3 without RoPE, Mamba-2 style Euler baseline) on the dataset and prints publication-ready comparison tables.

---

## Verification Plan

### Automated Tests
1. **Unit & Dimension Tests**:
   - Verify forward and backward pass shapes for `Mamba3Block` and `Mamba3LMHeadModel`.
   - Verify equivalence of parallel vs recurrent step inference.
2. **Synthetic State-Tracking Benchmark**:
   - Run `synthetic_eval.py` to confirm Mamba-3's complex RoPE mechanism solves the parity task (100% accuracy) whereas real-valued Euler baseline fails, exactly as demonstrated in Table 5b.
3. **Dataset Training & Evaluation**:
   - Run training on the provided dataset (`dataset/train.txt`), evaluate on `dataset/validation.txt` and `dataset/test.txt`, and log loss, validation perplexity, test perplexity, and token accuracy.
4. **Text Generation Test**:
   - Sample generated completions given prompt prefixes to verify generation coherence and recurrent state caching.
