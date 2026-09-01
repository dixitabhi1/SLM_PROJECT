# AI Search Framework v2: Executive Presentation Brief

**Core Research Question:** Can an entirely SLM-based decomposed pipeline (&le; 8B) match or substitute for monolithic LLMs across parameter scales?  
**Key Headline:** Decomposed SLMs achieve **41.6% cost savings over 70B models** and **88.9% savings over Frontier APIs**, while a **re-decomposition feedback loop reduces graph structural errors by 58.3%**.

---

## 1. System Architecture (v2 with Feedback Loop)

```
       [User Query]
            |
            v
   [Decomposer SLM (<= 3B)] <------+
            |                      |  Feedback Loop:
            v                      |  If task spans multiple colors
    [SLM-2: Skill Vector]          |  and depth < 3
            |                      |
            v                      |
    [SLM-3: Task Colorer]          |
            |                      |
            v                      |
     [Matching SLM] ---------------+
            | (Single-color OR Depth=3)
            v
     [Scheduling SLM]
            |
            v
   [Specialized SLM Pool] (Coding, Math, Logic, Retrieval, General)
            |
            v
   [Two-Stage Aggregator (<= 8B)]
            |
            v
     [Final Response]
```

---

## 2. Parameter Scale Cost & Latency Benchmark ($N=80$)

| Baseline System | Parameter Scale | Cost Ratio (SLM / Baseline) | Cost Savings (%) | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-3.1-8B** | 8.0B | $2.656\times$ | $-165.6\%$ (Overhead) | $[2.508, 2.805]$ |
| **Qwen-2.5-32B** | 32.5B | $1.178\times$ | $-17.8\%$ | $[1.112, 1.244]$ |
| **Llama-3.1-70B** | 70.6B | **$0.584\times$** | **+41.6% Savings** | $[0.551, 0.616]$ |
| **Qwen-2.5-72B** | 72.7B | **$0.584\times$** | **+41.6% Savings** | $[0.551, 0.616]$ |
| **Gemini-1.5-Pro** | Frontier API | **$0.111\times$** | **+88.9% Savings** | $[0.104, 0.117]$ |

> **Cost Crossover Point:** The economic crossover occurs at **$\approx 35\text{B}$ parameters**. Multi-agent SLM pipelines are more expensive than a single small model, but yield massive compute savings over $70\text{B}+$ and frontier models.

---

## 3. Feedback Loop Structural Accuracy Gain (RQ5)

* **Graph Edit Distance:** Reduced from **3.57 &rarr; 1.50**.
* **Relative Error Reduction:** **58.33%** structural accuracy improvement ($p < 0.001$).

---

## 4. Executive Takeaway

An **all-SLM search pipeline with feedback loop** delivers near-parity domain precision with **$41.6\%$ to $88.9\%$ compute cost reductions** against large monolithic models.
