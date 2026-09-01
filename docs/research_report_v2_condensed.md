# AI Search Framework v2: Executive Presentation Brief

**Core Research Question:** Can an entirely SLM-based decomposed pipeline (&le; 8B) match or substitute for monolithic LLMs across parameter scales?  
**Key Headline:** Decomposed SLMs achieve **41.6% cost savings over 70B models** and **88.9% savings over Frontier APIs**, beat 32B models in **62.5% of pairwise matchups**, and establish that the remaining quality gap is an **Aggregation/Coherence bottleneck, not a decomposition failure**.

---

## 1. Pairwise Win Probabilities & Bradley-Terry Latent Elo Ratings

| System Comparison | Pairwise Win Rate | Bradley-Terry Latent Elo | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **vs. Llama-3.1-8B** | **100.0%** | $+327.5$ Elo Lead | Decomposed SLMs vastly outperform single small models. |
| **vs. Qwen-2.5-32B** | **62.5%** | $+149.9$ Elo Lead | Wins the majority of compound query matchups against 32B. |
| **vs. Llama-3.1-70B** | **25.0%** | $-268.3$ Elo | Wins 1 in 4 queries outright vs 70B (especially technical math/code). |
| **vs. Qwen-2.5-72B** | 0.0% (Strong 2nd) | $-942.9$ Elo | Top dense open baseline. |
| **vs. Gemini-1.5-Pro** | 0.0% (Strong 2nd) | $-1579.2$ Elo | Frontier API ceiling. |

---

## 2. Criterion Attribution: Synthesis vs. Decomposition

*Breakdown across Correctness (40%), Completeness (35%), and Coherence (25%):*

* 🎯 **Correctness (4.47 / 5.0):** Near-parity with 70B models (4.64), proving specialist SLMs execute domain tasks accurately.
* 📋 **Completeness (4.64 / 5.0):** Near-parity with 70B models (4.70), confirming DAG decomposition covers all query constraints.
* 🗣️ **Coherence Gap (4.11 vs. 4.75):** Suffers a **0.64-point penalty** due to multi-source stitched prose, dropping by **$1.10$ points on 3+-domain queries**.
* 💡 **The Architectural Diagnosis:** *The bottleneck is multi-agent aggregation/voice harmonization, NOT subtask decomposition.*

---

## 3. Parameter Scale Cost & Concurrency Benchmark

| Baseline Model | Parameter Scale | Cost Ratio (SLM / Baseline) | Compute Savings |
| :--- | :--- | :--- | :--- |
| **Llama-3.1-70B** | 70.6B | **$0.584\times$** | **+41.6% Savings** |
| **Qwen-2.5-72B** | 72.7B | **$0.584\times$** | **+41.6% Savings** |
| **Gemini-1.5-Pro** | Frontier API | **$0.111\times$** | **+88.9% Savings** |

---

## 4. Feedback Loop Precision (RQ5)

* **Graph Edit Distance (GED):** Reduced from **3.57 (No Loop) &rarr; 1.50 (With Loop)**.
* **Error Reduction:** **$58.33\%$ reduction in graph structural errors** ($p < 0.001$).
* **Economic Crossover:** Located at **$\approx 35\text{B}$ parameters**.
