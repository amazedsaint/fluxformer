# FluxFormer v3: Exact Reversible Coupling, Per‑Head Gauge Attention, and Separable Q‑Walks

## Abstract
FluxFormer v3 unifies (i) **local linear‑time transport** via split‑step orthogonal updates (Q‑walk), (ii) **sparse global hops** via **exact per‑head gauge‑biased attention**, and (iii) **exactly reversible coupling segments** that combine an invertible 1×1 channel mixer with an additive‑coupling expert. We extend transport to images through **separable 2‑D Q‑walks** and add **Top‑K reversible MoE** with capacity control. We document complexity, invariants, and scheduling (window growth, attention‑frequency k‑scheduling) to approach near‑linear runtime at long context while preserving long‑range modeling.

---

## 1. Design principles
- **Locality and causality.** Transport moves features by at most one position per sub‑step, producing a **discrete light‑cone**.
- **Orthogonality and reversibility.** All linear pieces are orthogonal/permutation ⇒ spectral norm ≈ 1; coupling segments admit **closed‑form inverse**.
- **Relabeling invariance (gauge).** Only **relative** phases matter. Position enters as **edge phases**; attention uses **per‑head linear relative bias**.
- **Sparse global hops.** Occasional windowed attention creates long‑range jumps without quadratic cost.

## 2. Q‑walk transport (1‑D and 2‑D)
Let features be paired per channel: \((a,b)\). One tick consists of:
1. **Coin:** pairwise Givens rotation \(R(\Theta)\).
2. **Split shift:** \(a\to t+1\), \(b\to t-1\).
3. **Odd coin:** apply roll to change pairing, then rotate.
4. **Edge phases:** another rotation \(R(\phi_t)\).
5. **Shift again**, then **GaugeNorm** (amplitude nonlinearity on pair radii).

**Linear core invariants.** Coins/edge‑phases are rotations; shifts are permutations. Hence exact conservation of \(\|h\|^2\).

**2‑D separable Q‑walk.** Apply the same split‑step along rows (H) then columns (W). All sub‑steps remain rotations/permutations ⇒ linear invariants persist.

## 3. Exact per‑head gauge attention
For head \(h\), apply \(\text{bias}_h(i,j)=-g_h\max(0,i-j)\) under a causal mask. This yields **multi‑timescale specialization** and minimal overhead. Use **windowed** attention with window \(W\) to maintain amortized near‑linear complexity.

## 4. Exact reversible coupling segment
We define an **invertible block**:
- **Mix:** \(z = U\,x\), where \(U\) is an orthogonal 1×1 mixer (Givens product).
- **Split:** \(z=(z_1,z_2)\).
- **Coupling:**
  \[
  y_2 = z_2 + f(z_1),\qquad
  y_1 = z_1 + g(y_2).
  \]
- **Unmix:** \(y = U^{-1}\,[y_1,y_2]\).

**Inverse is closed‑form:**
\([z_1,z_2]=U\,y\), then \(z_1 = y_1 - g(y_2)\), \(z_2 = y_2 - f(z_1)\), and \(x=U^{-1}[z_1,z_2]\).
No fixed‑point iteration required.

## 5. Top‑K reversible MoE with capacity
The router picks top‑K experts per token (weights by softmax). Each expert is an invertible additive‑coupling MLP. Per‑expert capacity \(C_e=\lceil\text{cf}\cdot\frac{B\,T}{E}\rceil\) bounds compute/memory; overflow is dropped and outputs renormalize by kept mass to preserve expectation.

## 6. Schedules for efficiency
- **Window sweep:** grow \(W\) over training (\(W_0\to W_1\) by fraction \(\tau\)).
- **k‑scheduling:** change attention frequency from 1 per 8 blocks → 1 per 4 → 1 per 2 (piecewise).

## 7. Complexity
- Q‑walk + mixer: \(O(T\,C)\).
- Windowed attention: \(O(T\,W\,H)\) amortized only on scheduled layers.
- Coupling + 1×1 is linear in \(C\) and reversible ⇒ low activation memory.

## 8. Tests (in repo)
- **NumPy invariants (1‑D & 2‑D):** exact norm conservation of linear core; bounded drift with GaugeNorm.
- **Per‑head vs mean‑slope:** KL divergence > 0 indicates head specialization.
- **Reversible coupling (NumPy):** forward followed by inverse reconstructs inputs to round‑off.
- **Schedules:** parser unit tests.

## 9. Practical recommendations
- Start with small \(W\)/rare attention (k large), then grow \(W\) and shrink k late in training.
- Use reversible coupling for default FFN; switch to Top‑K MoE where capacity/token matters.
- Keep Q‑walk depth \(q\in[3,5]\) between attention hops to enlarge the light‑cone cheaply.

## 10. Conclusion
FluxFormer v3 offers a principled route to **near‑linear scaling** with **exact reversibility** and strong long‑range modeling via **per‑head gauge attention**, applicable to text and images.
