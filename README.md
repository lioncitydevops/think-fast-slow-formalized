# Thinking, Fast and Slow, Formalized

**When a Calibrated Decision Model Should Wake a Reasoning Model in Service and Software Systems**  
*Implementation based on Yan Zhang & TzeHoung Lee (National University of Singapore, 2026).*

---

## 🌟 Overview

This system mathematically formalizes Daniel Kahneman's dual-process cognitive architecture for modern artificial intelligence systems by pairing:
1. **System 1 (Jev by TypeSafe AI)**: A fast, parallel, typed decision model returning calibrated confidences $c = \max_k p_k$ and probability vectors in ~0.5s without generating text.
2. **System 2 (Qwen3-8B)**: A slow, effortful hybrid reasoning model running locally with hardware acceleration on an **NVIDIA GeForce RTX 3070 GPU**, supporting controllable thinking token budgets $B$ ($B=0$ non-thinking mode for probe, $B>0$ thinking mode with `<think>` deliberation traces).
3. **Metacognitive Controller**: Implements Kahneman's "lazy controller" via an **Expected Value of Control (EVC)** optimization:
   $$\max_B s(c) h(B) - \lambda B$$
   producing three invariant operational regions:
   - **Act Band** ($c \ge \tau$): Law of least effort, release intuitive answer $a_1$ immediately.
   - **Deliberate Band** ($c \in (\tau_{\text{giveup}}, \tau_{\text{act}})$): Engage effortful reasoning with budget $B^*(c, \text{task})$.
   - **Give-Up Band** ($c \le \tau_{\text{giveup}}$): Autonomous fallback (Calibrated Bayes Action or Conformal Prediction Sets) when queries are fundamentally ambiguous and deliberation cannot help.

---

## 📐 Mathematical Foundations

| Paper Result | Construct | Mathematical Formula / Identity |
| :--- | :--- | :--- |
| **Proposition 5.1** | **Escalation Identity** | $G(Z) = M_Z (W_2 - W_1)$, where $M_Z = P(x \in Z, a_1 \neq a_2)$ is label-free disagreement mass and $W_k = P(a_k = y \mid x \in Z, a_1 \neq a_2)$. |
| **Proposition 5.3 & Remark 1** | **Act-Band Error Guarantee** | Under calibration, $P(a_1 \neq y \mid c \ge \tau) \le 1 - \tau$. Certified via Clopper-Pearson exact confidence bounds and audited via **Wald's SPRT**. |
| **Proposition 5.4 & Cor. 5.5** | **Paired Transitions** | Partitions escalated traffic into **Stable**, **Anchor**, **Reactance**, and **Drift**. Verifies $G^{\text{inf}} - G^{\text{blind}} = \sum_T P(T)[P(i=y \mid T) - P(b=y \mid T)]$. |
| **Proposition 5.6 & 5.7** | **Effort Allocation** | Closed-form optimum $B^*(c) = \beta \ln\left(\frac{s(c)}{\lambda \beta}\right)$ under saturating returns, with shape-free structure across 5 empirical return curves. |
| **Proposition 5.9** | **Calibrated Bayes Action** | $a^*(p) = \arg\min_{a \in \mathcal{A} \cup \{\text{hold}\}} \sum_k p_k L(a, k)$ minimizing expected loss under asymmetric misclassification costs. |
| **Proposition 5.10** | **Conformal Answer Sets** | $C(x) = \{k : 1 - p_k(x) \le \hat{q}\}$ with guaranteed finite-sample coverage $P(y \in C(x)) \ge 1 - \alpha$. |
| **Proposition 5.11** | **Noise vs. Bias** | Majority voting over $k$ runs corrects occasion noise when $p_i > 1/2$, but entrenches bias when $p_i < 1/2$. |

---

## 🔄 Five Interaction Patterns (Section 6)

1. **P1: Cascade (Default-Interventionist)**: Deliberate at a fixed token budget when $c < \tau$.
2. **P2: Budgeted Deliberation**: Allocate task-conditioned budget $\hat{B}(c, \text{task})$ estimated nonparametrically on a budget grid.
3. **P3: Disagreement Probe (Parallel-Competitive)**: Run Qwen3-8B in non-thinking mode ($B=0$) alongside Jev, thinking with full budget only on disagreement or low confidence.
4. **P4: Jev as Verifier**: When Qwen overturns Jev, Jev audits whether the reasoning trace factually supports the new answer, reverting hallucinations.
5. **P5: Blind-then-Reconcile**: System 2 reasons blind first, then audits its chain of thought in a short second turn upon seeing $(a_1, c)$ ("consider the opposite").

---

## 🏢 Seven Application Domains (Table 8)

- **Customer Service / CRM**: Category & churn classification; fallback to priority queue hold on ambiguity; anchors monitored.
- **Software Engineering**: Pull request security risk & flaky CI triage; fallback to block merge & page on-call; P4 verifier reverts rationalized overrides.
- **Requirements Engineering**: Functional vs non-functional & ambiguity checks; conformal answer sets for uncertain requirements.
- **Logistics / Port Operations**: Delay cause & customs holds; hold for physical inspection; P3 probe catches coherent false notices.
- **Consumer Behaviour / E-Commerce**: Review authenticity, intent to cancel, return policy conformance; fallback to human agent hold; audits bias vs noise across customer demographics.
- **E-Government Services**: Entitlement & eligibility pre-screening; mandatory human review fallback; audits demographic bias vs occasion noise.
- **Agentic Software**: Tool output relevance & prompt injection screening; context poisoning prevention.

---

## 🚀 Getting Started

### 1. Environment Configuration (`.env`)
The system connects to TypeSafe Jev via the official `typesafe-sdk` and to local Qwen3-8B via the local inference engine:
```bash
# TypeSafe Jev API Key
TYPESAFE_API_KEY=apikey_2173455e56f095bf4396a0bbdd99113d0c20_9151499d78ff24154fa872402b91745bbc44aa79be8e458192d24373a5e9128c
TYPESAFE_DEFAULT_MODEL=jev-latest

# Local Qwen3-8B Inference Server (OpenAI-compatible)
QWEN_API_BASE=http://127.0.0.1:8088/v1
QWEN_MODEL=qwen3-8b
```

### 2. Launch Local Qwen3-8B Engine (RTX 3070 GPU)
Run the GPU-accelerated local inference server:
```powershell
& "C:\Users\LENOVO\.docker\bin\inference\llama-server.exe" -m "C:\Users\LENOVO\LLM_ThumbDrive_exfat_backup\models\Qwen3-8B-Q4_K_M.gguf" -ngl 99 --port 8088 --host 127.0.0.1
```

### 3. Launch the Web Dashboard
```bash
python -m think_fast_slow server --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any web browser to view the interactive UI.

---

## 💻 CLI Commands

### Run a Single Query
```bash
python -m think_fast_slow run \
  --text "Diff: modified auth token validation to bypass signature check in dev mode if DEBUG=1." \
  --question "Does this pull request introduce a critical security regression?" \
  --options "security_risk, safe_to_merge" \
  --pattern P3 \
  --tau 0.88
```

### Replicate Paper Tables (Tables 6 & 9)
```bash
python -m think_fast_slow replicate --table all
```

### Run Section 8 Evaluation Harness (Table 11 Hypotheses H1-H9)
```bash
python -m think_fast_slow evaluate --mock --samples 25
```

### Run the 42 Pre-Specified Unit Tests
```bash
pytest tests/ -v
```

---

## 📊 Evaluation Harness & Proof Verification

The verified evaluation harness (`think_fast_slow/evaluation/harness.py`) and protocol (`PROTOCOL.md` frozen at `protocol-v1.0`) test:
- **Proposition 5.1 Escalation Identity**: Verified to difference $< 10^{-16}$ (machine precision).
- **Proposition 5.4 Paired Decomposition**: Verified with stable term identically zero ($< 10^{-16}$).
- **Table 11 Hypotheses (H1 to H9)**: Automated test suite asserting exact falsification bounds across BoolQ (T1), GSM8K (T2), and CRT lures (T3).
- **42 Comprehensive Unit Tests**: Passes all 42 tests covering API schemas, mathematical proofs, pattern mechanics, fallbacks, and noise audits.
- **Table 10 Four-Stage Deployment**: S0 Offline replay, S1 Shadow mode, S2 Canary with Wald's SPRT, and S3 Rollout.
