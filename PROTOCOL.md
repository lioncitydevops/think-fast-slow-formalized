# Pre-Specified Evaluation Protocol: Formalized Dual-Process System

**When a Calibrated Decision Model Should Wake a Reasoning Model in Service and Software Systems**  
*Document Version:* `protocol-v1.0`  
*Date Frozen:* 21 October 2026  
*Authors:* Yan Zhang & TzeHoung Lee (National University of Singapore)  
*Conference Target:* ASSE 2026, Osaka, Japan  
*Reference Paper:* `Lee_Zhang_ASSE2026_20261021.pdf`

---

## 1. Status of the Protocol (Section 8.1)

This evaluation protocol is pre-specified and frozen together with the implementation code at repository tag `protocol-v1.0` on 21 October 2026.
It is pre-specified rather than pre-registered; it becomes a formal registration when deposited with a public registry prior to the first live model call, and any subsequent amendment is logged with its timestamp and rationale.

The protocol execution environment guarantees:
1. Complete separation of empirical hypothesis testing from code correctness.
2. Verified unit testing (42 passing unit tests covering schema conformance, mathematical identities, and pattern mechanics).
3. Exact reproduction of theoretical and numerical results in Sections 5 and 7 (Table 6 return curves and Table 9 worked example).
4. Four staged deployment validation steps (Table 10: S0 Replay, S1 Shadow, S2 Canary, S3 Rollout).

---

## 2. Experimental Design and Benchmark Tasks (Section 8.2)

### 2.1 Benchmark Tasks
- **Task 1 (T1) — BoolQ Validation Set** ($N = 3,270$ yes/no items):
  Targets natural language factual comprehension where System 1 (Jev) demonstrates calibrated strength.
- **Task 2 (T2) — GSM8K Test Set** ($N = 1,319$ items):
  Recast as 4-option multiple choice with pre-fixed distractors, targeting Jev's documented vulnerability with arithmetic and multi-step symbolic calculation.
- **Task 3 (T3) — Cognitive Reflection & Adversarial Lure Set** (Target $N = 200$ items):
  Newly authored CRT isomorphs with fresh numbers and wording to eliminate contamination, in which intuitive heuristic answers are provably incorrect, paired with distractor-laden items producing inverse test-time scaling. The exact items and their SHA-256 hash are committed prior to any model calls.

### 2.2 Data Partitioning
Each dataset is partitioned into:
- **30% Calibration Split**: Used for act threshold certification via Clopper-Pearson exact bounds (Remark 1), nonparametric budget curve estimation $\hat{B}(c, \text{task})$ on the discrete grid $B \in \{0, 128, \dots, 16k\}$, and conformal nonconformity quantile calibration $\hat{q}$.
- **70% Test Split**: Held out for evaluating all pre-specified hypotheses and measuring overall service accuracy, token spend, and latency frontiers.

---

## 3. Machine Instantiation (Table 2 & Section 3)

| Component | System 1: Jev | System 2: Qwen3-8B |
| :--- | :--- | :--- |
| **Vendor Documented Type** | Typed answer; choice: per-option probabilities & confidence $c$; yes/no: $P(\text{yes})$ only | Free text; answer parsed from final `Answer: <choice>` line |
| **Text Generation** | None (pure representation projection) | Up to 32k tokens deliberative `<think>` trace |
| **Latency & Cost** | $\sim 0.5$s; \$0.042 per million input tokens | GPU-bound (RTX 3070 local engine); scales with budget $B$ |
| **Confidence** | Trained for calibration: $P(a_1 = y \mid c) = c$ | Verbalized confidence often overconfident |
| **Known Weak Spots** | Numbers, dates, adversarial lures (WYSIATI) | Overthinking easy problems, inverse scaling on lures |

---

## 4. Mathematical Foundations (Section 5)

### 4.1 Proposition 5.1: Escalation Identity
Under 0–1 loss, the accuracy gain from releasing System 2 answer $a_2$ instead of System 1 answer $a_1$ on escalated queries $Z$ is:
$$G(Z) = M_Z (W_2 - W_1)$$
where:
- $M_Z = P(x \in Z, a_1 \neq a_2)$ is the **label-free disagreement mass** (computable on live unlabelled traffic).
- $W_k = P(a_k = y \mid x \in Z, a_1 \neq a_2)$ is the win rate of model $k$ given disagreement (requires labels).
- For binary choice ($K=2$), $G(Z) = M_Z (2W_2 - 1)$.

### 4.2 Proposition 5.3 & Remark 1: Act-Band Error Guarantee
If System 1 is calibrated, for any threshold $\tau$:
$$P(a_1 \neq y \mid c \ge \tau) = \mathbb{E}[1 - c \mid c \ge \tau] \le 1 - \tau$$
**Remark 1 Finite-Sample Certificate:** Given $n$ labelled calibration items, scan $\tau$ from high to low starting where $n_\tau \ge \lceil \ln\delta / \ln(1 - \varepsilon) \rceil$ (59 items for $\varepsilon = \delta = 0.05$), stopping at the first threshold where the exact Clopper-Pearson upper confidence bound exceeds $\varepsilon$.

### 4.3 Proposition 5.4 & Corollary 5.5: Paired Transition Decomposition
On escalated queries $Z$, System 2 answers blind ($b$) and informed ($i$). Queries partition into **Stable** ($i=b$), **Anchor** ($b \neq a_1, i=a_1$), **Reactance** ($b=a_1, i \neq a_1$), and **Drift** ($b \neq a_1, i \notin \{a_1, b\}$):
$$G^{\text{inf}} - G^{\text{blind}} = \sum_T P(T) [P(i=y \mid T) - P(b=y \mid T)]$$
with the Stable term vanishing identically ($0.0$). Under pure anchoring at rate $\alpha$:
$$G^{\text{inf}} = (1 - \alpha) G^{\text{blind}}$$

### 4.4 Propositions 5.6–5.8: Effort as a Limited Resource
Deliberation benefit $s(c)h(B) - \lambda B$, where:
- $s(c) = [(1 - c) - rc](1 - d^\gamma)$ with $d = 2(1 - c)$.
- Closed form: $B^*(c) = \beta \ln\left(\frac{s(c)}{\lambda\beta}\right)$ if $s(c) > \lambda\beta$, else $0$.
- Law of least effort: $B^* = 0$ for $c \ge (1 - \lambda\beta)/(1 + r)$.
- Nonparametric plug-in budget $\hat{B}(c, \text{task})$ loses at most $2\eta$ in expected return (Prop 5.8).

### 4.5 Table 6: Invariant Operational Bands Across Five Return Curves

| Return Curve $h(B)$ | Act Band $c \ge$ | Give Up Band $c \le$ | Peak $B^*$ | Min $B^* > 0$ | Kept Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Saturating** | 0.90 | 0.57 | 2.76k | 57 | 100% |
| **Log-linear** | 0.90 | 0.59 | 2.62k | 10 | 97% |
| **Early-flat** | 0.91 | 0.58 | 0.55k | 8 | 71% |
| **Threshold (S-shaped)** | 0.85 | 0.57 | 5.52k | 3.74k ($B_{\text{tan}}$) | 45% |
| **Inverted-U (Inverse)** | 0.90 | 0.56 | 2.07k | 10 | 97% |

### 4.6 Fallback Mechanisms (Section 5.5)
When queries fall in the Give-Up Band ($E[\text{error}] > \varepsilon_g$):
1. **Calibrated Bayes Action** (Prop 5.9): $a^*(p) = \arg\min_a \sum_k p_k L(a, k)$, selecting safe hold under asymmetric losses.
2. **Conformal Answer Sets** (Prop 5.10): $C(x) = \{k : 1 - p_k(x) \le \hat{q}\}$ with guaranteed coverage $P(y \in C(x)) \ge 1 - \alpha$.

### 4.7 Noise vs. Bias in Majority Voting (Proposition 5.11)
Majority voting over $k$ runs corrects occasion noise when $p_i > 1/2$ ($M_k(p) > p$), but entrenches systematic bias when $p_i < 1/2$ ($M_k(p) < p$).
**Non-Circular Noise Audit Protocol:** 10 runs per query are collected — 5 runs form the majority vote, and 5 independent runs estimate item competence $\hat{p}_i$.

---

## 5. Interaction Patterns & Who Checks Whom (Section 6 & Table 7)

```
        query: text state + typed question
                       │
             ┌─────────▼─────────┐
             │   System 1: Jev   │ (every query, ~0.5s)
             │   a1, p, conf c   │
             └─────────┬─────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   [Act Band]    [Probe Check]   [Give-Up Band]
   (c >= tau)      (Qwen B=0)    (post-err > eps_g)
       │          conflict?            │
       │       ┌───────┴───────┐       ▼
       │      No              Yes  Autonomous Fallbacks:
       │       │               │   - Calibrated Bayes Action
       │       ▼               ▼   - Conformal Prediction Set
  Release a1  Release a1    Wake System 2:
  (Audit q%)                Qwen3-8B Thinking (B*)
                               │
                       ┌───────┴───────┐
                       ▼               ▼
                 [P5 Reconcile]  [P4 Verifier]
                 (2nd turn audit) (Jev checks trace)
                               │
                         Release Final
```

| Step | Who checks whom | When | Sees other's answer? | Pattern |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Jev answers | Every query | — | Base |
| **2** | Qwen checks Jev (Probe) | Every query, if enabled; $B=0$ | No | P3 |
| **3** | Nobody: $a_1$ released | $c \ge \tau$ and no probe conflict | — | Act |
| **3a** | Qwen re-checks Jev (Audit) | Random share $q$ of step 3; output unchanged | No | Monitoring |
| **4** | Qwen checks Jev | $c < \tau$ or probe conflict; budget $B^*$ | No | P1, P2 |
| **5** | Qwen reconsiders | After step 4, if enabled | Yes: $(a_1, c)$ | P5 |
| **6** | Jev checks Qwen (Verifier)| After steps 4–5, only if Qwen $b \neq a_1$ | Yes: trace & ans | P4 |
| **—** | Fallback | $B=0$ & $1-c > \varepsilon_g$, or post-thinking error $> \varepsilon_g$ | — | Fallbacks |

---

## 6. Pre-Specified Hypotheses and Falsification Criteria (Table 11)

| ID | Hypothesis | Falsification Criterion |
| :--- | :--- | :--- |
| **H1** | T1 (BoolQ): Act-band error $\le \varepsilon$ at certified $\tau$ | Act-band error confidence interval upper bound $> \varepsilon$ |
| **H1'**| T2 (GSM8K): Jev overconfident: on $c \ge 0.9$, error exceeds mean of $1-c$ | Error confidence interval lower bound $\le$ mean of $1-c$ |
| **H2** | $W_2 > W_1$ in middle band only; $W_2 \le W_1$ in top band | Top-band $W_2 - W_1$ confidence interval strictly above $0$ |
| **H3** | Anchor share $>$ Reactance share; Informed worse than Blind ($G^{\text{inf}} < G^{\text{blind}}$); Reconcile anchor $<$ Informed anchor | Any inequality reversed with CI excluding $0$ |
| **H4** | Nonparametric $B^*$ is $0$ in top band, positive in middle; curves differ by task | $B^* > 0$ in top band, or no task difference |
| **H5** | Probe disagreement predicts thinking-mode disagreement ($\text{AUROC} > 0.70$) | $\text{AUROC} \le 0.70$ |
| **H6** | Verifier cuts harmful overrides $\ge 30\%$, keeps $\ge 80\%$ of beneficial ones | Either condition missed |
| **H7** | Conformal coverage $\ge 1 - \alpha - 0.02$; Bayes action beats always-act under stated loss | Either condition missed |
| **H8** | T3 (CRT lures): Jev confidently wrong on lures; probe catches more lure errors than any confidence threshold | Any confidence threshold catches as many lure errors |
| **H9** | Voting gains confined to items with $\hat{p}_i > 1/2$; losses where $\hat{p}_i < 1/2$ ($\hat{p}_i$ from independent runs) | Net accuracy gain observed where $\hat{p}_i < 1/2$ |

---

## 7. Application Domains (Table 8)

1. **Customer Service / CRM**: Category, urgency, churn, refund; S2 plans complex tickets; fallback: priority queue hold; risk: anchoring on S1 category.
2. **Software Engineering**: PR risk tier, flaky vs real CI test failure, on-call page; S2 root causes diffs; fallback: block merge & page human; risk: rationalized overrides (P4).
3. **Requirements Engineering**: Functional vs non-functional, ambiguity, testability; S2 rewrites ambiguity; fallback: conformal set & stakeholder query; risk: overthinking clear items.
4. **Logistics / Port Operations**: Delay cause, customs hold, dangerous goods; S2 re-plans berth; fallback: hold for physical inspection; risk: WYSIATI on coherent false notices (P3).
5. **Consumer Behaviour / E-Commerce**: Review genuine, intent to cancel, return consistency; S2 crafts retention offer; fallback: human agent hold; risk: bias vs noise across customer demographics.
6. **E-Government Services**: Eligibility pre-screen, document completeness; S2 checks regulation text; fallback: mandatory human review; risk: demographic bias vs occasion noise.
7. **Agentic Software**: Tool result relevance, prompt injection detection; S2 multi-step plan; fallback: drop result & stop agent; risk: confident errors on adversarial text (WYSIATI).

---

## 8. Four Stages to Deployment (Table 10)

- **S0: Offline Replay**: Sweep benchmark/historical queries; Jev on all; Qwen blind at every budget on grid. Certify $\tau$ on calibration split.
- **S1: Shadow Mode**: Run full controller in parallel with incumbent production process; log decisions without acting. Certify act-band error $\le \varepsilon$ (59–361 labels for $\varepsilon=5\%$).
- **S2: Canary**: Controller decides small traffic fraction; fail-closed; audit share $q$; run Wald's SPRT on audited decisions ($\alpha = \beta = 0.05$, $H_0: \text{error} \le 5\%$ vs $H_1: \text{error} \ge 8\%$).
- **S3: Rollout**: All live traffic handled by controller; drift alarm triggered on significant shift in label-free disagreement rate $M$ or SPRT rejection.
