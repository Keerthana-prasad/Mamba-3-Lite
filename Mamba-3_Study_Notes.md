# Mamba-3 — Complete Study Notes
*(Digitized from handwritten notes + completed for remaining sections of the base paper: "Mamba-3: Improved Sequence Modeling using State Space Principles")*

---

## 0. Big Picture

Mamba-3 introduces **three** core improvements over Mamba-2, all motivated from the **SSM (state-space model) viewpoint**:

1. **Exponential-Trapezoidal Discretization** — a more expressive way to turn the continuous SSM into a discrete recurrence (Section 3.1).
2. **Complex-valued SSM** — lets the model do "rotational" state tracking that real-valued SSMs (like Mamba-2) provably cannot do (Section 3.2).
3. **MIMO (Multi-Input, Multi-Output)** — restructures the state update as a matmul instead of an outer product, raising arithmetic intensity (FLOPs/byte) during decode without hurting latency (Section 3.3).

---

## 1. Preliminaries — State Space Model (SSM)

### 1.1 Mathematical model
- SSM = **mathematical model** describing how a hidden state **h(t)** evolves in **continuous time**.
- Continuous system:
  - **h′(t) = A(t) h(t) + B(t) x(t)**  (state ODE)
  - **y(t) = C(t)ᵀ h(t)**  (output equation)
  - h(t) = hidden state, A = state transition matrix, B = input matrix, x(t) = input, C = output/readout matrix.
- But real data (tokens) is **discrete** (token1, token2, …), not continuous. So we need **discretization**: convert the continuous ODE into a discrete recurrence that can run over a sequence.

### 1.2 Discretization (general)
- Discrete system: **h_t = Ā h_{t-1} + B̄ x_t**, **y_t = C h_t**
- Neural nets/hardware can't literally "solve" a continuous ODE step by step efficiently — this discrete recurrence *is* what "neural network conversion" of the continuous ODE looks like.
- Diagram (as drawn): input token embedding → xₜ₋₁, xₜ feed into successive **SSM** blocks together with the *previous* hidden state h_{t-1}, producing hₜ, which is read out via C to get yₜ = Chₜ (next-token logits).
- Standard closed-form discretization (Zero-Order Hold, ZOH):
  - **Ā = exp(ΔA) = e^{ΔA}**
  - **B̄ = A⁻¹ (e^{ΔA} − I) ΔB** = (ΔA)⁻¹(exp(ΔA) − I)ΔB
  - Δ = **trainable step-size parameter** (learned).
- **Assumption used to derive ZOH:** the input signal x(t) is treated as **piecewise constant** over each step (Zero-Order **Hold**) — because in general we can't know/compute the continuous x(t) exactly, we only observe it at discrete points, so we *assume* it is held constant between samples.
- Consequence: "Input integral cannot be computed exactly in general → we only know discrete points → we approximate."

---

## 2. Exponential-Euler Discretization (= Mamba-1 / Mamba-2's method)

### 2.1 Motivation
- Exponential-Euler = a **simplification** for solving ODEs/SDEs (the exponential part is solved *exactly*, only the input part is approximated with Euler's rule).
- The integrand to evaluate: **f(T) = e^{(T_t − T)A} B x(T)**.

### 2.2 Euler's rule
- Euler's rule just says: **evaluate f at the right endpoint T = T_t**, and multiply by the width Δ.
  - Assumption: the value is **held constant** across the whole interval → this is exactly the ZOH assumption, i.e. an error of O(Δ) is introduced.
- Applying this:
  - ∫ f(T) dT ≈ e^{(T_t − T_t)A} B x(T_t) · Δ = **Δ B_t x_t**
  - This avoids the matrix inverse **A⁻¹(e^{ΔA} − I)** that appears in exact ZOH — that whole term is approximated simply as **ΔB** (no matrix inverse, no extra matrix exponential evaluation needed for B̄).
- **Result — Exponential-Euler recurrence:**
$$h_t = e^{Δ_t A_t}\, h_{t-1} + Δ_t B_t x_t$$
- This is the (previously heuristic, now theoretically justified by the base paper) recurrence used by **Mamba-1 and Mamba-2**.

### 2.3 Why "no matrix inverse" matters
- **B̄ = ΔB**, so:
  $$\bar B = ΔB$$
- No matrix inverse, no extra matrix exponential evaluation is required — this is why Mamba-1/2's discretization is cheap and hardware-friendly.

### 2.4 Where it sits in the general family
- Base paper's framing: "**Exponential-Trapezoidal**" instantiation is a strict **generalization** of exponential-Euler, and is **equivalent** to applying an implicit convolution (data-dependent).
- Recovering special cases via the blend parameter λ_t:
  - **λ_t = 1** → recovers Mamba-2's Euler rule.
  - **λ_t = 1/2** → recovers the classical trapezoidal rule.

---

## 3. Exponential-Trapezoidal Discretization (Mamba-3's core idea)

### 3.1 Motivation: second-order discretization
- Euler's rule only uses **one** endpoint of the interval (first-order accurate).
- The **Trapezoidal rule blends both endpoints** (data-dependent weight λ) → second-order accurate → smaller error.
- Integrand (same as before): **f(T) = e^{(T_t − T)A} B x(T)**, evaluated over interval [T_{t-1}, T_t].
  - At the **right endpoint** T = T_t: **f(T_t) = B_t x_t**
  - At the **left endpoint** T = T_{t-1}: **f(T_{t-1}) = e^{(T_t − T_{t-1})A} B_{t-1} x_{t-1} = e^{ΔA} B_{t-1} x_{t-1}**
- The hidden state accumulates the effect of **both endpoints at every instant**, not just the current one.

### 3.2 Trapezoidal approximation of the integral
$$\int \approx Δ_t\left[(1-λ_t)\,e^{Δ_t A_t} B_{t-1}x_{t-1} + λ_t\, B_t x_t\right]$$

### 3.3 Final recurrence (Proposition 1 in the paper)
$$\boxed{h_t = e^{Δ_tA_t} h_{t-1} + (1-λ_t)Δ_t e^{Δ_tA_t} B_{t-1}x_{t-1} + λ_tΔ_t B_t x_t}$$
Written with named coefficients (matches Table 1 of paper):
$$h_t = α_t h_{t-1} + β_t B_{t-1}x_{t-1} + γ_t B_t x_t, \quad α_t{=}e^{Δ_tA_t},\ β_t{=}(1{-}λ_t)Δ_te^{Δ_tA_t},\ γ_t{=}λ_tΔ_t$$

- **λ_t ∈ [0, 1]** controls **where** the approximation is taken from — it is **data-dependent** ("data-dependent blend"), i.e. **learned per token**, unlike the classical trapezoidal rule which is fixed at λ=1/2.
  - Example from notes: λ_t = 0.7 → 70% weight on one endpoint, 30% on the other.
- In Mamba-2, only **one** input (current token) feeds the recurrence. In Mamba-3, the recurrence now receives **two** inputs — the **previous** input (weighted by β_t) and the **current** input (weighted by γ_t).

### 3.4 Sequential vs. parallel computation
- The recurrence is **sequential**: h_1 must be computed before h_2, before h_3, … — this is slow on GPUs, which thrive on **parallel** matrix multiplication, not sequential dependency chains.
- **Key insight:** you can **unroll** the recurrence and write the *entire* output sequence y_0, y_1, …, y_T as a **single matrix multiplication** (this is what enables efficient parallel training — see §3.6/§6 below, the SSD / mask-matrix trick).

---

## 4. Expressivity

- **Expressivity** = how many different kinds of functions/behaviors a model can represent.
- λ = 1 → Euler; λ = 1/2 → (classical) trapezoidal; λ = 0.73 (say) → a genuinely *new*, in-between discretization.
- Every "exponential-time" model can behave differently — the model can **adapt** to the numerical integration of its input.
- **This flexibility (a learnable λ) means the model can represent a much wider variety of input-sequence dynamics — the paper claims this directly improves expressivity.**

### 4.1 Error-rate comparison
- **Euler's rule**: local truncation error ∝ Δ², leading to a **global** error of **O(Δ)** — "first order."
- **Trapezoidal rule**: local error ∝ Δ³, leading to a **global** error of **O(Δ²)** — "second order."
- Numerically: for Δ = 0.1 → Δ³ = 0.001 vs Δ² = 0.01, i.e., the cubic (trapezoidal) local error shrinks **much faster** as the step size Δ decreases → the approximation improves rapidly.
- **Why we need to approximate at all:** the integral ∫f(τ)dτ has **no closed form** — B(τ) and x(τ) both change with τ, *and* the exponential term changes too. So the paper proposes to **approximate the integral using the general trapezoidal rule** (Proposition 1).

### 4.2 From the ordinary trapezoidal rule to the *general* (weighted) one
- **Ordinary trapezoidal rule:** ∫ₐᵇ f(x)dx ≈ (b−a)/2 · (f(a) + f(b))  — i.e., "average height × width," giving the trapezoid's area.
- **Rectangle rule** (Euler): ∫ₐᵇ f(x)dx ≈ f(a)·(b−a).
- **General (weighted) trapezoidal rule:** introduce λ ∈ [0,1]:
  $$\int_a^b f(x)\,dx \approx (b-a)\big[(1-λ)f(a) + λf(b)\big]$$
  - λ = 1/2 → recovers the **ordinary trapezoidal rule**.
  - λ = 0 → recovers the **Euler / rectangle rule**.
  - λ = 1 → the other rectangle rule (right endpoint only).

### 4.3 Remark 1 (Data-dependence of A_t)
- In Mamba-2, A_t is **data-independent**; only the overall discrete transition α_t = e^{ΔₜAₜ} is data-dependent, via Δ_t.
- In Mamba-3, the authors found data-dependent A_t performs similarly to data-independent A_t, but chose data-dependent **A_t** anyway, for consistency (so *all* SSM parameters are data-dependent).

### 4.4 Remark 3 (Error rate, from the base paper)
- Mamba-3's exponential-trapezoidal is a **second-order** discretization of the state-input integral: error scales as **O(Δ_t³)**, *provided* the trapezoidal parameter satisfies λ_t = 1/2 + O(Δ_t) (a stability condition).
- This is a **10× smaller error** compared to Mamba-2's Euler-based discretization, under the standard stability assumption.
- **However**, ablations show that **not enforcing** this λ_t = 1/2 + O(Δ_t) constraint gives **better empirical performance** — so Mamba-3 leaves λ_t free (learned, unconstrained) for better empirical performance, at the cost of the clean theoretical error-rate guarantee. (Confirmed by ablation in Appendix A.3: default free λ_t gets ppl 15.72 vs. fixed-1/2 15.76 vs. Euler-only 15.81.)

### 4.5 Intuition recap
- Instead of using **only the current input**, like Mamba-2, Mamba-3 uses **previous input + current input**.
- Options considered for combining them (to find a higher-order accurate approximation of the input):
  1. Ignore the previous input (→ Euler / Mamba-2).
  2. Simply average current & previous input (→ classical trapezoidal, λ=1/2).
  3. **Learn a data-dependent weighting λ_t** between current & previous input (→ Mamba-3's choice).

---

## 5. The Recurrence Contains an *Implicit* Convolution

### 5.1 Setting up the state-input variable
- From Proposition 1 we derived:
  $$h_t = α_t h_{t-1} + β_t B_{t-1}x_{t-1} + γ_t B_t x_t \qquad \text{— eqn (a)}$$
- Base paper's claim: *"our generalized exponential-trapezoidal discretization is equivalent to applying a data-dependent convolution of size 2 on the state-input to the SSM."*
- Instead of the recurrent equation just depending on the raw input, it **naturally contains an implicit convolution.**
- Recall the underlying continuous ODE: dh(t)/dt = A h(t) + B(t)x(t).
- Define the **state-input**: **v_t = B_t x_t**  — eqn (b)  (multiply the raw input by B first).

### 5.2 Substituting back
- Replace B_{t-1}x_{t-1} and B_t x_t in (a) using (b):
  $$h_t = α_t h_{t-1} + β_t v_{t-1} + γ_t v_t$$
- Compare with a "normal" SSM: now the recurrence receives **two inputs**:
  - **previous input** → x_{t-1} → multiply by B_{t-1} → **v_{t-1}**
  - **current input** → x_t → multiply by B_t → **v_t**
- I.e., **the input, after being transformed (converted into state space) via B**, is what enters the recurrence — not the raw input directly.

### 5.3 The "hidden" convolution
- Define **z_t = β_t v_{t-1} + γ_t v_t**.
- Then the recurrence becomes: **h_t = α_t h_{t-1} + z_t** — this looks **exactly** like a normal (Mamba-2-style) SSM recurrence, just with z_t (a mixed/blended input) substituted in place of a plain input term.
- **z_t is precisely a 1-D convolution of width 2** applied to the state-input sequence v_t = B_t x_t, with **kernel [k₁, k₂] = [β_t, γ_t]**:
  - Suppose we're looking at the sequence v_{t-1}, v_t (kernel window of size 2).
  - Output = k₁·v_{t-1} + k₂·v_t, i.e. **z_t = β_t v_{t-1} + γ_t v_t**.
  - At time t, the kernel "covers" v_{t-1} and v_t.
- **⇒ Convolution is hidden inside the recurrence.** Formally: *we must apply a width-2 convolution to v_t before passing it into the (linear) recurrence h_t = α_t h_{t-1} + z_t* — **there is no separate/explicit conv layer** needed; it's baked into β_t, γ_t.

### 5.4 Remark 4 — Important distinction: convolution on *state-input*, not raw input
- x₀, x₁, x₂, … → (❌ **NOT**: raw convolution) → new input → SSM.
- Correctly: x_t → multiply by B_t → v_t = B_t x_t → **mix** (β·v_{t-1} + γ·v_t) → update hidden state.
- **The convolution acts on the state-input B_t x_t, not on the raw input x_t.**
- This is different from the standard "short causal convolution" used by Mamba-1/2, GDN, etc., which is applied directly to x_t (and often B_t, C_t) **outside** the core recurrence, as an independent pre-processing step.
- Practical implication (Section 4.2 of paper / architecture): because Mamba-3's discretization *already* induces this convolution-like mixing (together with learnable B, C biases), the model can **empirically drop the external short causal convolution** that was previously believed to be essential in Mamba-1/2-style architectures.

---

## 6. Parallel (Matrix) Form — Unrolling the Recurrence & the Mask Matrix

### 6.1 Unroll the *simpler* Mamba-2 case first (Exponential-Euler)
Growing from **h₀ = γ₀ B₀ x₀**:
- **h₁ = α₁ h₀ + γ₁ B₁ x₁ = α₁γ₀ B₀x₀ + γ₁B₁x₁**
- **h₂ = α₂ h₁ + γ₂ B₂ x₂ = α₂α₁γ₀ B₀x₀ + α₂γ₁ B₁x₁ + γ₂B₂x₂**
- In general, unrolling **h_t = α_t h_{t-1} + γ_t B_t x_t** shows every output is a weighted sum of **all past inputs** x₀,…,x_t, where the weight on an old input is the **product of all the decay terms (α's) between then and now.**
- Output: **y_t = C_tᵀ h_t**.
- So every output y_t is a weighted sum of all past inputs x₀, x₁, …, x_t, where weights involve products of the α's.

### 6.2 The mask ("L") matrix
Stacking these as a matrix equation (with h expanded in terms of x₀…x_T), we get a **lower-triangular mask matrix** L such that Y = (L ⊙ CBᵀ) X:

$$
L =
\begin{bmatrix}
1 & & & \\
α_1 & 1 & & \\
α_2α_1 & α_2 & 1 & \\
\vdots & & & \ddots \\
α_{T\cdots1} & \cdots & & 1
\end{bmatrix}
\cdot \text{Diag}(γ)
$$

(This is exactly Eq. 3 in the base paper — Mamba-2's SSD mask.)

### 6.3 Extending to Mamba-3's Exponential-Trapezoidal recurrence
- Because Mamba-3's recurrence has **two** input terms per step (β_t B_{t-1}x_{t-1} and γ_t B_t x_t), unrolling it produces a mask that is **richer** than Mamba-2's diagonal-γ mask:
$$
L =
\begin{bmatrix}
γ_0 & & & \\
(γ_0α_1+β_1) & γ_1 & & \\
α_2(γ_0α_1+β_1) & (γ_1α_2+β_2) & γ_2 & \\
\vdots & & & \ddots
\end{bmatrix}
$$
- The paper shows this factors as a product of **two** structured matrices (Eq. 7):
$$
L = \underbrace{\begin{bmatrix}1\\α_1&1\\α_2α_1&α_2&1\\\vdots&&&\ddots\end{bmatrix}}_{\text{1-semiseparable (decay) mask}} \cdot \underbrace{\begin{bmatrix}γ_0\\β_1&γ_1\\0&β_2&γ_2\\\vdots&&&\ddots\end{bmatrix}}_{\text{2-band (convolution) mask}}
$$
- i.e., **the structured mask induced by exponential-trapezoidal discretization = (decay mask) × (2-band convolutional mask)** — this matches Figure 1 (left) of the base paper, and is exactly the algebraic form of "an implicit width-2 convolution sitting inside the decay recurrence" from Section 5 above.
- This proves Mamba-3 is still an instance of the **SSD (Structured State-Space Duality)** framework — just with a more expressive **L**, generalizing Mamba-2's L (which only has a diagonal γ term, no 2-band convolution part).

---

## 7. Complex-Valued SSMs (Section 3.2 — new content)

### 7.1 Motivation: state tracking
- Real, non-negative-eigenvalue transition matrices (as in Mamba-1/2) provably **cannot** represent "rotational" hidden-state dynamics — this makes them fail at simple **state-tracking** tasks, e.g. computing **parity** of a bit sequence (Σxₜ mod 2), which a single-layer LSTM *can* solve.
- Complex eigenvalues **can** represent rotation, and rotation is exactly what's needed for parity-like tasks (e.g. h_t = R(πx_t) h_{t-1}, a 2-D rotation matrix conditioned on the input bit).

### 7.2 Complex SSM formulation (Proposition 2)
- Consider a complex-valued SSM:
$$h'(t) = \text{Diag}(A(t) + iθ(t))\,h(t) + (B(t)+i\hat B(t))\,x(t), \qquad y(t) = \text{Re}\big[(C(t)+i\hat C(t))^\top h(t)\big]$$
- Under exponential-Euler discretization, this complex system is **exactly equivalent** to a **real-valued** SSM with **double the state dimension (N)**, whose transition matrix is a **block-diagonal matrix of 2×2 rotation matrices**:
$$h_t = e^{Δ_tA_t}\,R_t\,h_{t-1} + Δ_t B_t x_t,\qquad y_t = C_t^\top h_t$$
$$R_t = \text{Block}\{R(Δ_tθ_t[i])\}_{i=1}^{N/2},\qquad R(θ)=\begin{bmatrix}\cosθ&-\sinθ\\\sinθ&\cosθ\end{bmatrix}$$
- Intuition: pair up real dimensions into 2-D subspaces; each pair rotates by an input-dependent angle θ_t while also decaying by the usual scalar e^{Δ_tA_t}.

### 7.3 Connection to RoPE ("the RoPE trick") — Proposition 3
- Because the rotation matrices R_t commute and are orthogonal (R_t⁻¹ = R_tᵀ), the recurrence can be **rewritten**: instead of rotating the hidden state every step, you can equivalently **pre-rotate B and C** by the *cumulative product* of rotations up to time t:
$$h_t = e^{Δ_tA_t}h_{t-1} + \Big(\textstyle\prod_{i=0}^t R_i^\top\Big) Δ_tB_tx_t,\qquad y_t=\Big[\Big(\textstyle\prod_{i=0}^tR_i^\top\Big)C_t\Big]^\top h_t$$
- This is **structurally identical to RoPE** (rotary position embeddings, Su et al. 2023) — except the rotation angles θ are **data-dependent** rather than a fixed frequency schedule. This is why the authors call it a **data-dependent rotary embedding**.
- Practical benefit: this "RoPE trick" lets the complex-valued update be computed with **the same efficient machinery as vanilla RoPE**, at minimal extra compute cost vs. a real-valued SSM.

### 7.4 Extending to exponential-trapezoidal (Proposition 4)
- Combining complex SSM with the exponential-trapezoidal discretization (from §3) gives the **final Mamba-3 recurrence**:
$$h_t = α_th_{t-1} + β_t\Big(\prod_{i=0}^{t-1}R_i^\top\Big)B_{t-1}x_{t-1} + γ_t\Big(\prod_{i=0}^{t}R_i^\top\Big)B_tx_t,\qquad y_t = \Big[\Big(\prod_{i=0}^tR_i^\top\Big)C_t\Big]^\top h_t$$
- I.e., both the "previous-token" and "current-token" state-input terms get their own (cumulative) rotation applied, on top of the width-2 convolutional blend from Section 5.

### 7.5 Architecture split (Figure 2)
- Mamba-3's SSM transition **A** is complex: **A = (real part) + i(imaginary part Θ)**.
- The real part is passed into the same SSD "black box" machinery as Mamba-2 (the decay).
- The imaginary part Θ is handled via the **RoPE trick** (applied to B, C).

### 7.6 Empirical validation (Table 5b)
| Model | Parity | Modular Arith. (no brackets) | Modular Arith. (brackets) |
|---|---|---|---|
| Mamba-3 (full, w/ data-dep. RoPE) | **100.0** | **98.5** | **87.8** |
| Mamba-3 w/ standard RoPE | 1.6 | 20.7 | 2.6 |
| Mamba-3 w/o RoPE | 2.3 | 1.5 | 0.7 |
| Mamba-2 | 0.9 | 47.8 | 0.9 |
| GDN [-1,1] (state-tracking variant) | 100.0 | 99.3 | 93.5 |

→ Only the **data-dependent** rotary version solves these tasks; neither plain Mamba-2 nor Mamba-3-with-fixed-RoPE can.

---

## 8. Multi-Input, Multi-Output — MIMO (Section 3.3 — new content)

### 8.1 The problem: decoding is memory-bound
- **Arithmetic intensity** = FLOPs ÷ bytes moved. SSM decode has arithmetic intensity ≈ **2.5 ops/byte** (Table 2a), while H100 bf16 matmul peak is ≈ **295 ops/byte** — i.e. decode is deeply **memory-bound**: the GPU is mostly idle, waiting on memory, not compute.
- This applies broadly to sub-quadratic models (SSMs, causal linear attention), not just Mamba.

### 8.2 From SISO to MIMO
- A standard SSM head is really **P** copies of a scalar (SISO) recurrence sharing the same α_t, Δ_t, B_t, broadcast across the head dimension P:
  $$h_t \leftarrow α_t h_{t-1} + Δ_t B_t x_t^\top \quad (\text{state } h_t\in\mathbb R^{N\times P})$$
- The compute here is dominated by an **outer product** B_t x_tᵀ (FLOPs ∝ NP), while **memory traffic is dominated by the state h_t** — so increasing FLOPs without increasing state size directly raises arithmetic intensity.
- **MIMO idea:** expand **B_t: ℝᴺ → ℝᴺˣᴿ** and **x_t: ℝᴾ → ℝᴾˣᴿ** (rank R). Now:
  - Memory traffic barely increases (state size N×P is unchanged).
  - FLOPs increase by a factor of **R**, and the outer product becomes a **matmul** — which runs efficiently on tensor cores.
  - Similarly expand **C_t: ℝᴺ → ℝᴺˣᴿ** for the output projection y_t ← C_tᵀh_t.
- Net effect: **R× more FLOPs, same state size, ~same decode wall-clock time** (since compute overlaps with memory movement) → strictly more expressive model at ~no latency cost.

### 8.3 Table 2 comparison (arithmetic intensity)
| | SISO | MIMO |
|---|---|---|
| Arithmetic intensity | Θ(1) ≈ 2.5 | **Θ(R)** (R ≪ N, P) |

### 8.4 Training MIMO models
- A MIMO SSM can be expressed as **R²** SISO SSMs (R "input" heads × R "output" heads), each computed with a shared recurrence and then summed:
$$h_t^{(j)}\leftarrow α_th_{t-1}^{(j)}+Δ_tB_t^{(j)}x_t^{(j)},\quad h_t=\textstyle\sum_j h_t^{(j)},\quad y_t^{(i)}\leftarrow(C_t^{(i)})^\top h_t$$
- This lets existing SISO training algorithms (chunked SSD-style) be reused as a black box — MIMO training costs only **~R×** SISO training cost (not R²), via a chunked algorithm with **chunk size C_MIMO = C_SISO / R**.

### 8.5 Parameter-efficient instantiation
- A naive R× expansion of all of B, C, x projections would blow up parameter count.
- Mamba-3's trick: since B, C are **already shared across heads** (multi-value attention structure), only B, C projections need to grow (small cost: D·N → D·N·R for the whole layer).
- For the **per-head** input/output/gate projections (x, y, z) — which *do* dominate parameter count — Mamba-3 keeps the **original SISO projection** and just **element-wise scales** it up to rank R with a **learnable, data-independent** vector: cost D·P + P·R per head (**additive**, not multiplicative) increase.
- To keep total parameter count matched to the SISO baseline, the MLP inner width is slightly reduced (e.g. at 1.5B: 4096 → 3824, only −6.6%).

### 8.6 Results
- MIMO (R=4) gives the best language-modeling numbers at every scale (Table 3) — e.g. at 1.5B, **+1.2 pts** average accuracy over SISO, **+1.8 pts** over the best non-Mamba-3 baseline (GDN).
- Kernel latency (Table 6): MIMO decode is only marginally slower than SISO, and both SISO/MIMO Mamba-3 kernels are competitive with or **faster than** Mamba-2/GDN kernels.
- Prefill (Table 7): MIMO does incur a **moderate prefill overhead** (more FLOPs, and prefill is compute- not memory-bound), but decode latency remains excellent.

---

## 9. Mamba-3 Architecture (Section 3.4 — new content)

- Overall backbone follows **Llama**-style: alternating **Mamba-3 blocks** and **SwiGLU (MLP) blocks**, pre-norm.
- Changes to the Mamba-2 block (see Figure 2 comparison):
  1. **Updated SSM recurrence:** SSD layer replaced by the complex-valued exponential-trapezoidal SSM (Proposition 4). SISO used by default (fair comparison); MIMO available as a stronger variant.
  2. **BC / QK Normalization (BCNorm):** RMSNorm applied after the B, C projections (mirrors QK-Norm in Transformers). This alone is enough to stabilize large-scale training, **letting Mamba-3 drop the post-gate RMSNorm** that Mamba-2 needed for stability (pure models only — hybrid models still need it for long-context extrapolation, Table 4/9).
  3. **B, C biases:** learnable, head-specific, channel-wise biases added to B and C after BCNorm. Hypothesized to add a **convolution-like, data-independent** component to the SSM (echoing Section 5's implicit-convolution story). Combined with exponential-trapezoidal discretization, this **removes the need for the external short causal convolution** used by nearly every prior linear-attention/SSM model (Mamba-1/2, GDN, etc.) — confirmed in ablation (Table 5a: "+conv" doesn't help once bias+trapezoidal are present).
  4. **RoPE box:** the imaginary/rotational part of A (Θ) is computed via the RoPE trick and applied to normalized B, C, as in Section 7.

---

## 10. Empirical Validation (Section 4 — new content, summary)

### 10.1 Language modeling (Table 3, 100B FineWeb-Edu tokens)
- Across **all 4 scales (180M, 440M, 880M, 1.5B)**, Mamba-3 SISO ≥ Mamba-2 and GDN on average downstream accuracy; **Mamba-3 MIMO** is best at every scale.
- At 1.5B: Mamba-3 MIMO improves average accuracy by **+2.2 pts over Transformer, +1.9 over Mamba-2, +1.8 over GDN**; Mamba-3 SISO alone beats the next-best baseline (GDN) by **+0.6 pts**.
- State-size efficiency (Fig. 3): Mamba-3 (MIMO) with state size **64** matches Mamba-2's perplexity at state size **128** → same quality at **half the state/latency**.

### 10.2 Retrieval (Table 4, NIAH + real-world cloze tasks)
- Mamba-3 is competitive with Transformer/GDN on associative recall & QA (TriviaQA, SQuAD), but — like other fixed-state models — struggles on **semi-structured/unstructured extraction** (SWDE, FDA), where Transformers (unbounded KV cache) dominate.
- On **synthetic NIAH**, Mamba-3 clearly beats Mamba-2, especially **out-of-distribution length generalization**.
- **Hybrid models** (5:1 linear:attention layers, NoPE attention) close much of the retrieval gap; re-adding a pre-gate grouped RMSNorm to the Mamba-3 layer further improves long-context NIAH generalization (at a small cost to in-context real-world retrieval) — Table 9 ablates norm placement/type.

### 10.3 SSM methodology ablations (Table 5a, 440M scale)
| Variant | Test ppl ↓ |
|---|---|
| Mamba-3 − bias − trapezoidal (≈ Mamba-2-like) | 16.68 |
| Mamba-3 − bias | 16.49 |
| **Mamba-3 (full)** | **15.72** |
| Mamba-3 + external conv | 15.85 (**no improvement — conv is redundant**) |

### 10.4 State-tracking (Table 5b) — see §7.6 above.

### 10.5 Inference efficiency (Section 4.3–4.4)
- **Pareto frontier** (perplexity vs. state size, Fig. 3): Mamba-3 strictly dominates Mamba-2; Mamba-3 MIMO dominates further.
- **Kernel latency** (Table 6, H100, batch 128): Mamba-3 SISO is the **fastest** decode kernel among all baselines tested (bf16, d_state=128); MIMO (R=4) adds only a little overhead.
- **End-to-end latency** (Table 7): Mamba-3 SISO fastest at all sequence lengths (512–16384 tokens); all linear models greatly outperform vLLM/Transformer attention as sequence length grows (attention scales much worse due to KV cache).
- Overall message: Mamba-3's extra methodological machinery (exp-trapezoidal update, complex state, MIMO) is **lightweight** — it doesn't cost much speed for a real quality gain ("inference-first" design philosophy).

---

## 11. Related Work (Section 5 — brief summary)

- **Three main lineages** of linear-time sequence mixers, which conceptually overlap but have different native vocabularies/motivations:
  1. **Linear attention** (kernel feature maps, delta-rule / fast-weight programmers like DeltaNet/GDN) — "associative memory" view (Q,K,V).
  2. **Test-time training/regression (TTT/TTR)** — state = compressed summary updated via online optimization steps.
  3. **State space models (SSMs)** — S4 → Mamba-1 (data-dependent selectivity) → Mamba-2 (SSD, matmul form) → **Mamba-3** (this paper). Native vocabulary is A, B, C, X (control theory), not Q, K, V.
- Mamba-3's 3 contributions are argued to be **most natural from the SSM viewpoint specifically** (discretization, complex transitions, MIMO are all classical SSM/control-theory concepts) and **less obviously motivated** from the linear-attention/associative-memory viewpoint.
- **State tracking**: prior theory (Merrill et al. 2025; Sarrof et al. 2024; Grazzi et al. 2025) shows real-valued, non-negative-eigenvalue SSMs are stuck in **TC⁰** and cannot solve tasks like S₅ permutation composition or parity — motivating Mamba-3's complex-valued update.
- **Complex SSMs historically**: S4/H3/RetNet/Megalodon used complex values but underperformed Transformers on language and mostly didn't use the RoPE trick (RetNet is a partial exception, with constant — not data-dependent — phase). Mamba-3 is (to the authors' knowledge) the **first modern recurrent LM to combine complex-valued transitions with *data-dependent* RoPE**, grounded in a theoretical equivalence (not just heuristically).
- **MIMO historically**: S4 was SISO (independent SSM per channel); S5/LRU moved to MIMO to *simplify* computation (trading off some capacity); Mamba-1/2 went back to SISO because state size mattered more once selectivity (data-dependence) was introduced. Mamba-3's MIMO is different in **motivation**: it's added purely to raise decode arithmetic intensity/FLOPs **without** shrinking the state or hurting latency — not to simplify training.

---

## 12. Conclusion (Section 6 — brief)

Mamba-3 = Mamba-2 + {exponential-trapezoidal discretization, complex-valued state (via data-dependent RoPE), MIMO}. Together these:
- Give a stronger, theoretically-grounded recurrence (justifying Mamba-1/2's discretization as a special case, "exponential-Euler").
- Unlock state-tracking capability real-valued SSMs provably lack.
- Improve hardware utilization during decode without hurting latency (MIMO).
- Let the model **drop the external short causal convolution** long thought necessary for good performance in this model family.
- Advance the quality-vs-efficiency Pareto frontier over Mamba-2, GDN, and Transformers at every scale tested (180M–1.5B).

---

## Appendix — Quick-Reference Formula Sheet

| Quantity | Meaning |
|---|---|
| h_t | hidden state at time t |
| x_t | input token (embedding) at time t |
| A_t, Δ_t | (data-dependent) continuous state-transition rate & step size |
| α_t = e^{Δ_tA_t} | discrete decay factor |
| B_t, C_t | (data-dependent) input / output projections |
| v_t = B_t x_t | "state-input" (input after B-projection) |
| λ_t ∈ [0,1] | data-dependent trapezoidal blend weight |
| β_t = (1−λ_t)Δ_t e^{Δ_tA_t} | weight on **previous** state-input v_{t-1} |
| γ_t = λ_t Δ_t | weight on **current** state-input v_t |
| θ_t | imaginary (rotational) part of complex transition A |
| R_t | 2×2-block rotation matrix built from θ_t (the "RoPE trick") |
| R (MIMO) | multi-input/output rank |
| L | SSD structured mask matrix (parallel/matmul form of the recurrence) |

**Master recurrence (Mamba-3, full form — Proposition 4):**
$$h_t = α_t h_{t-1} + β_t\Big(\textstyle\prod_{i=0}^{t-1}R_i^\top\Big)B_{t-1}x_{t-1} + γ_t\Big(\textstyle\prod_{i=0}^{t}R_i^\top\Big)B_tx_t$$
$$y_t = \Big[\Big(\textstyle\prod_{i=0}^{t}R_i^\top\Big)C_t\Big]^\top h_t$$

- Set **λ_t=1** (β_t=0) and drop rotation (R_t=I) → recovers **Mamba-2**.
- Set **λ_t=1/2**, drop rotation → recovers the **classical trapezoidal-discretized SSM**.
