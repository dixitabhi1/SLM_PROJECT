# Literature Grounding & Related Work Review

**Project:** AI Search Framework (All-SLM Decomposed Pipeline vs. Monolithic LLM Baseline)  
**Phase:** 1 (Literature Grounding)  
**Status:** Complete  
**Date:** 2026-08-27  

---

## 1. Executive Summary & Research Context

Modern AI search and question-answering systems overwhelmingly rely on routing incoming user queries to a single, monolithic, general-purpose Large Language Model (LLM, typically 70B+ parameters to frontier-class API models). While effective for general reasoning, this monolithic paradigm introduces substantial computational overhead, high inference latencies, and high token costs—particularly for compound, multi-domain queries where only specific sub-tasks require specialized reasoning.

This research project investigates whether an **entirely Small Language Model (SLM)-based pipeline** ($\le 8\text{B}$ parameters across every component: Decomposer $\le 3\text{B}$, non-generative Capability Router, Specialized Pool Models $\le 8\text{B}$, and Aggregator $\le 8\text{B}$) can match or approach a fixed monolithic large-LLM baseline in response quality while achieving significant reductions in end-to-end latency and compute cost.

To correctly position this investigation within the broader literature, this review examines related multi-model architectures, LLM routing frameworks, DAG-based task decomposition, and collaborative small-model networks, highlighting exactly where prior work ends and where this study's core empirical contribution begins.

---

## 2. Taxonomy of Related Work

### 2.1 Mixture-of-Agents (MoA) & Layered Multi-Model Collaboration
* **Key Literature:** *Mixture-of-Agents Enhances Large Language Model Capabilities* (Wang et al., Together AI / Stanford / Duke, 2024; arXiv:2406.04692).
* **Mechanism:** MoA proposes a layered architecture where multiple LLMs act as agents in parallel layers. In each layer, every model receives the outputs of all models from the previous layer as context, iteratively refining answers until a final LLM produces the aggregated response.
* **Relation to Our Work:** MoA demonstrated the "collaborativeness" of language models—models generate higher quality outputs when presented with other models' intermediate generations. However, MoA predominantly relies on large models (e.g., Qwen-72B, Llama-3-70B) in both generation and aggregation layers, resulting in compounding latency and token costs. Our work enforces a strict **all-SLM constraint** ($\le 8\text{B}$) and replaces all-to-all iterative layering with a **dependency-aware DAG decomposition** and **targeted single-stage capability routing**, explicitly targeting low latency and self-hostability.

### 2.2 Dynamic Model Routing (RouteLLM, RouterDC, Hybrid Routing)
* **Key Literature:**
  * *RouteLLM: Learning to Route LLMs with Preference Data* (Ong et al., LMSYS / UC Berkeley, 2024; arXiv:2406.18665).
  * *RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models* (Zhang et al., 2024; arXiv:2409.19886).
* **Mechanism:** RouteLLM trains binary routers using preference data (e.g., Chatbot Arena) to dynamically dispatch simple queries to cheaper/smaller models and hard queries to frontier LLMs. RouterDC generalizes this to multi-model pools using dual contrastive embeddings to align query representations with model capabilities.
* **Relation to Our Work:** Dynamic routing frameworks route the *entire* query as an atomic unit between an SLM and an LLM. In contrast, our pipeline decomposes compound queries into sub-tasks, routing sub-queries to specialized domain SLMs (Coding, Math, Reasoning, General). Furthermore, while RouteLLM retains an LLM fallback for hard queries, our proposed deployment system contains **zero LLMs**, testing the limit of what a pure SLM ensemble can achieve.

### 2.3 DAG Decomposition & Multi-Agent Orchestration (S-DAG, MoMA)
* **Key Literature:**
  * *MoMA: Mixture-of-model-and-agent routing for generalized multi-agent orchestration* (Guo et al., 2025; arXiv:2509.07571).
  * *AdaptOrch & S-DAG: Structural DAG Task Allocation in Multi-Agent Systems* (2024–2025).
* **Mechanism:** S-DAG and related systems decompose multi-step reasoning problems into Directed Acyclic Graphs (DAGs), enabling topological execution where independent nodes execute concurrently and dependent nodes await prerequisite context.
* **Relation to Our Work:** We adopt the DAG decomposition paradigm for compound queries but impose rigorous architectural constraints: the decomposition is performed by a lightweight SLM ($\le 3\text{B}$), routing overhead is kept near-zero via non-generative rule/embedding matching, and dependency-aware dispatch is executed by an asynchronous Python orchestrator with confidence-based replication.

### 2.4 Collaborative Small Model Networks (The Avengers, Avengers-Pro)
* **Key Literature:**
  * *The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants* (Zhang et al., 2025; arXiv:2505.19797).
  * *Avengers-Pro: Beyond GPT-5: Making LLMs Cheaper and Better via Performance–Efficiency Optimized Routing* (Zhang et al., 2025; arXiv:2508.12631).
* **Mechanism:** The Avengers framework demonstrates that a collective of open-source $\sim 7\text{B}$ SLMs (specialized in math, code, logic) orchestrated via lightweight embedding, clustering, and voting can challenge proprietary frontier LLMs on benchmark tasks.
* **Relation to Our Work:** While Avengers validates that specialized SLMs hold domain-specific competitive advantages, it treats tasks as atomic domain queries routed to single SLMs or majority-voted. Our framework addresses **compound, cross-domain queries** (e.g., calculating statistical parameters in Python, analyzing mathematical convergence, and deriving domain-specific qualitative conclusions) via explicit multi-node DAG execution and structured synthesis by a specialized aggregator SLM ($\le 8\text{B}$).

---

## 3. Comparative Matrix: AI Search Framework vs. Prior Literature

| Framework | Decomposer Size | Router Type | Pool Models | Aggregator Size | LLM in Architecture? | Baseline Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MoA** (Wang et al., 2024) | None (Full query) | Layered All-to-All | 70B+ / Mixed LLMs | 70B+ LLM | **Yes** (Primary) | Monolithic GPT-4 / Open LLMs |
| **RouteLLM** (Ong et al., 2024) | None (Atomic query) | Preference Classifier | 1 SLM + 1 Frontier LLM | None | **Yes** (Fallback) | Monolithic Frontier LLM |
| **The Avengers** (Zhang et al., 2025) | None (Cluster/Router) | Embedding / Voting | $\sim 7\text{B}$ SLMs (Atomic) | Voting / None | **No** | Proprietary LLMs (Benchmark only) |
| **MoMA** (Guo et al., 2025) | Generative / LLM Agent | Bandit-based Router | Mixed LLMs / Agents | LLM Agent | **Yes** | Single LLM Agents |
| **Proposed AI Search Framework** | **Prompted SLM ($\le 3\text{B}$)** | **Rule / Embedding (Non-generative)** | **Specialized SLMs ($\le 8\text{B}$)** | **Specialized SLM ($\le 8\text{B}$)** | **ZERO LLMs** | **Paired Fixed Monolithic Large LLM (70B+ / Frontier)** |

---

## 4. Scoping the Core Research Contribution

Grounding against prior literature clarifies what this project is and is not:

1. **Not Claiming Novelty in Generic Orchestration:** The abstract pattern of *Decompose $\to$ Route $\to$ Parallel Dispatch $\to$ Aggregate* is well-known.
2. **The Core Empirical Contribution:** A rigorous, controlled, within-subject paired empirical benchmark testing whether a strictly constrained, zero-LLM architecture ($\le 8\text{B}$ across all stages) can substitute for monolithic LLM scale on latency, compute cost, and answer quality across query complexity strata.
3. **Transparent Characterization of Failure Modes:** Unlike literature that emphasizes positive win-rates on select benchmarks, this study explicitly models and evaluates where small-model specialization breaks down (RQ3/RQ4: deep reasoning bottlenecks, coordination overhead, and graph decomposition errors via RQ5).

---

## 5. References

1. Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). *Mixture-of-Agents Enhances Large Language Model Capabilities*. arXiv:2406.04692.
2. Ong, I., Yang, A., Rahman, M. A., et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data*. arXiv:2406.18665.
3. Zhang, Y., et al. (2024). *RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models*. arXiv:2409.19886.
4. Zhang, Y., et al. (2025). *The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants*. arXiv:2505.19797.
5. Zhang, Y., et al. (2025). *Beyond GPT-5: Making LLMs Cheaper and Better via Performance–Efficiency Optimized Routing (Avengers-Pro)*. arXiv:2508.12631.
6. Guo, X., et al. (2025). *MoMA: Mixture-of-model-and-agent routing for generalized multi-agent orchestration*. arXiv:2509.07571.

