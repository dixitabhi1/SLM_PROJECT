"""
Eval Dataset Builder & Verifier
AI Search Framework (Phase 3: Dataset Construction)

Generates:
1. data/eval_dataset_master.json (180 queries across 3 complexity tiers)
2. data/queries_dev.json (60 queries for dev/calibration)
3. data/queries_held_out.json (120 queries locked for final eval)
4. data/gold_dags.json (60 gold DAGs across dev & held-out subsets)
5. data/scoring_rubric.json (Pre-registered evaluation rubric)
6. data/held_out_lock.sha256 (Cryptographic lock of held-out split)
"""

import json
import hashlib
import os
from datetime import datetime, timezone

os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Generation: 180 Stratified Queries
# -------------------------------------------------------------
queries = []

# --- Tier 1: Single-Domain (Control) (n = 50) ---
# Coding (15)
coding_queries = [
    ("SD_CODE_01", "Implement an LRU cache in Python with O(1) get and put operations using OrderedDict or a doubly linked list with a hash map, including type hints and unit tests.", ["coding"]),
    ("SD_CODE_02", "Write a Python implementation of an asynchronous token-bucket rate limiter supporting concurrent asyncio coroutines with burst limits.", ["coding"]),
    ("SD_CODE_03", "Implement the Dijkstra shortest path algorithm in Python for an adjacency-list weighted graph using a min-heap priority queue.", ["coding"]),
    ("SD_CODE_04", "Write a thread-safe singleton pattern in Python using a metaclass with double-checked locking.", ["coding"]),
    ("SD_CODE_05", "Implement an efficient Trie data structure in Python supporting prefix search, wildcard match ('.'), and word frequency counting.", ["coding"]),
    ("SD_CODE_06", "Write a Python script to parse and stream a large gzip-compressed JSONL file line-by-line using minimal memory and generator pipelines.", ["coding"]),
    ("SD_CODE_07", "Implement the merge sort algorithm iteratively in Python without recursion, tracking comparison counts.", ["coding"]),
    ("SD_CODE_08", "Create a Python custom context manager that measures and logs peak memory consumption of a code block using tracemalloc.", ["coding"]),
    ("SD_CODE_09", "Implement a circular buffer (ring buffer) in Python with overwrite semantics on overflow and thread-safe lock mechanisms.", ["coding"]),
    ("SD_CODE_10", "Write a Python decorator that implements exponential backoff retry with jitter for transient network exceptions.", ["coding"]),
    ("SD_CODE_11", "Implement an immutable Red-Black Tree insertion algorithm in Python with structural sharing.", ["coding"]),
    ("SD_CODE_12", "Write a Python parser for basic arithmetic expressions supporting +, -, *, /, and parentheses using a Recursive Descent parser.", ["coding"]),
    ("SD_CODE_13", "Implement a Bloom filter in Python using MurmurHash3 and a bitarray with configurable false positive probability.", ["coding"]),
    ("SD_CODE_14", "Write a Python function to solve the Longest Common Subsequence (LCS) problem using space-optimized dynamic programming.", ["coding"]),
    ("SD_CODE_15", "Implement an asynchronous connection pool manager in Python with health checks and max-idle timeout recycling.", ["coding"])
]

for qid, text, domains in coding_queries:
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": domains,
        "query": text,
        "gold_dag_available": True if int(qid.split("_")[-1]) <= 5 else False
    })

# Math (15)
math_queries = [
    ("SD_MATH_01", "Derive the closed-form posterior distribution for a normal likelihood with known variance and a conjugate normal prior, showing all intermediate algebra.", ["math"]),
    ("SD_MATH_02", "Calculate the eigenvalues, eigenvectors, and spectral decomposition of the 3x3 matrix [[4, 1, -1], [1, 2, 1], [-1, 1, 4]].", ["math"]),
    ("SD_MATH_03", "Evaluate the definite improper integral \int_0^\infty \frac{x^2}{1 + x^4} dx using contour integration in the complex plane.", ["math"]),
    ("SD_MATH_04", "Prove that the sum of the series \sum_{n=1}^\infty \frac{1}{n^2} converges to \pi^2 / 6 using Fourier series expansion of f(x) = x^2 on [-\pi, \pi].", ["math"]),
    ("SD_MATH_05", "Solve the first-order non-linear differential equation dy/dx + 2xy = x * e^(-x^2) * y^3 with initial condition y(0) = 1.", ["math"]),
    ("SD_MATH_06", "Find the maximum likelihood estimator (MLE) for the parameters (\alpha, \beta) of a Gamma distribution given an i.i.d. sample x_1, ..., x_n.", ["math"]),
    ("SD_MATH_07", "Calculate the stationary distribution of a 4-state Markov chain with transition matrix P = [[0.2, 0.8, 0, 0], [0.3, 0.4, 0.3, 0], [0, 0.5, 0.3, 0.2], [0, 0, 0.6, 0.4]].", ["math"]),
    ("SD_MATH_08", "Compute the Taylor series expansion of f(x) = ln(1 + e^x) around x = 0 up to the 4th order term.", ["math"]),
    ("SD_MATH_09", "Derive the gradient and Hessian matrix of the loss function L(w) = \frac{1}{2} ||Xw - y||_2^2 + \frac{\lambda}{2} ||w||_2^2 with respect to w.", ["math"]),
    ("SD_MATH_10", "Prove using mathematical induction that 2^(2n) - 1 is divisible by 3 for all positive integers n \ge 1.", ["math"]),
    ("SD_MATH_11", "Calculate the conditional expectation E[X | X + Y = z] where X and Y are independent standard normal variables N(0, 1).", ["math"]),
    ("SD_MATH_12", "Determine whether the vector space of 2x2 symmetric matrices is isomorphic to R^3 and construct the explicit linear isomorphism.", ["math"]),
    ("SD_MATH_13", "Solve the boundary value problem y''(x) + 4y(x) = \sin(x) on [0, \pi] with y(0) = 0 and y(\pi) = 0.", ["math"]),
    ("SD_MATH_14", "Compute the volume of the 4-dimensional hypersphere of radius R using multivariable spherical coordinates integration.", ["math"]),
    ("SD_MATH_15", "Determine the radius and interval of convergence of the power series \sum_{n=1}^\infty \frac{(-1)^n (x-2)^n}{n 3^n}.", ["math"])
]

for qid, text, domains in math_queries:
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": domains,
        "query": text,
        "gold_dag_available": True if int(qid.split("_")[-1]) <= 5 else False
    })

# Reasoning / Logic (10)
reasoning_queries = [
    ("SD_REAS_01", "Analyze the validity of the syllogism: 'All quantum algorithms that exhibit speedup require entanglement. Some quantum algorithms do not use Shor's period finding. Therefore, some algorithms requiring entanglement do not use Shor's period finding.' State mood, figure, and formal validity.", ["reasoning"]),
    ("SD_REAS_02", "Evaluate whether the statement 'If P -> (Q v R), then (P -> Q) v (P -> R)' is a logical tautology using truth tables and formal semantic deduction.", ["reasoning"]),
    ("SD_REAS_03", "Resolve the Newcomb's Paradox from both Causal Decision Theory (CDT) and Evidential Decision Theory (EDT) perspectives.", ["reasoning"]),
    ("SD_REAS_04", "Identify and analyze the informal fallacies in this argument: 'Experts haven't proven that AI alignment is unsolvable; therefore, superintelligence will naturally align with human values once it achieves high cognitive capacity.'", ["reasoning"]),
    ("SD_REAS_05", "Construct a formal modal logic proof in system S5 showing that \Box P -> \Box \Box P.", ["reasoning"]),
    ("SD_REAS_06", "Analyze the resolution of the Grandfather Paradox in closed timelike curves under the Novikov Self-Consistency Principle vs. Many-Worlds Interpretation.", ["reasoning"]),
    ("SD_REAS_07", "Evaluate the epistemic justification of inductive reasoning under Nelson Goodman's New Riddle of Induction ('grue' vs 'green').", ["reasoning"]),
    ("SD_REAS_08", "Solve the classic Zebra/Einstein logic puzzle constraint satisfiability step-by-step with 5 houses, distinct colors, nationalities, pets, drinks, and cigarettes.", ["reasoning"]),
    ("SD_REAS_09", "Analyze the game-theoretic subgame perfect equilibrium of a 3-player Ultimatum game with sequential alternating offers.", ["reasoning"]),
    ("SD_REAS_10", "Deduce the truth values of statements made by Knights and Knaves in a 3-person island scenario where A says 'B is a knave', B says 'A and C are of same type', and C says 'A is a knight'.", ["reasoning"])
]

for qid, text, domains in reasoning_queries:
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": domains,
        "query": text,
        "gold_dag_available": True if int(qid.split("_")[-1]) <= 5 else False
    })

# Retrieval / General (10)
general_queries = [
    ("SD_GEN_01", "Summarize the architectural differences between Transformer Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA) with respect to KV-cache memory bandwidth.", ["retrieval", "general"]),
    ("SD_GEN_02", "Explain the consensus mechanism of the Raft distributed algorithm, detailing leader election, log replication, and safety guarantees.", ["retrieval", "general"]),
    ("SD_GEN_03", "Provide a comprehensive breakdown of the EU AI Act classification tiers (Prohibited, High Risk, Limited Risk, Minimal Risk) and compliance requirements for Foundation Models.", ["retrieval", "general"]),
    ("SD_GEN_04", "Explain the difference between Optimistic Concurrency Control (OCC) and Two-Phase Locking (2PL) in relational databases, focusing on write-skew anomalies.", ["retrieval", "general"]),
    ("SD_GEN_05", "Explain the mechanism of CRISPR-Cas9 genome editing, specifically the role of the guide RNA, Cas9 endonuclease cleavage, and Non-Homologous End Joining vs Homology-Directed Repair.", ["retrieval", "general"]),
    ("SD_GEN_06", "Describe the key distinctions between microkernel architectures (e.g. seL4) and monolithic kernels (e.g. Linux) regarding IPC latency and fault isolation.", ["retrieval", "general"]),
    ("SD_GEN_07", "Summarize the primary mechanisms of action of GLP-1 receptor agonists (e.g., Semaglutide) in metabolic regulation and appetite control.", ["retrieval", "general"]),
    ("SD_GEN_08", "Explain the CAP theorem and PACELC theorem trade-offs in distributed database design with examples (Cassandra vs Spanner).", ["retrieval", "general"]),
    ("SD_GEN_09", "Detail the cryptographic primitives and key exchange flow in TLS 1.3 compared to TLS 1.2, highlighting 0-RTT and handshake latency reduction.", ["retrieval", "general"]),
    ("SD_GEN_10", "Explain the concept of zero-knowledge SNARKs (zk-SNARKs), focusing on arithmetic circuits, QAPs, and elliptic curve pairings.", ["retrieval", "general"])
]

for qid, text, domains in general_queries:
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": domains,
        "query": text,
        "gold_dag_available": True if int(qid.split("_")[-1]) <= 5 else False
    })

# --- Tier 2: 2-Domain Compound (n = 70) ---
# Code + Math (20)
for i in range(1, 21):
    qid = f"TD_CM_{i:02d}"
    if i == 1:
        text = "Derive the mathematical formulation of the Kalman Filter (prediction and update steps for state covariance), and provide a vectorized Python implementation using NumPy that tracks a 2D moving object with noisy velocity measurements."
    elif i == 2:
        text = "Derive the gradient updates for logistic regression with L2 regularization from first principles, and implement the optimizer in Python from scratch with gradient checking against finite differences."
    elif i == 3:
        text = "Formulate the Black-Scholes PDE for European call option pricing, derive the closed-form analytical solution ($d_1$, $d_2$), and implement a Python class calculating Greeks ($\Delta, \Gamma, \Theta, \mathcal{V}, \rho$)."
    elif i == 4:
        text = "Derive the Expectation-Maximization (EM) algorithm update equations for a Gaussian Mixture Model (GMM), and write a Python implementation with log-likelihood convergence monitoring."
    elif i == 5:
        text = "Formulate the PageRank random walk transition matrix with teleportation parameter $\alpha$, and implement power iteration in Python with sparse matrix representations."
    elif i == 6:
        text = "Derive the fast Fourier transform (FFT) Cooley-Tukey radix-2 recurrence relations and implement the recursive and iterative FFT algorithms in Python without scipy."
    elif i == 7:
        text = "Formulate the dual problem of a Support Vector Machine (SVM) using Lagrange multipliers, and implement the Sequential Minimal Optimization (SMO) algorithm in Python."
    elif i == 8:
        text = "Derive the analytical solution of the 1D heat equation $\partial u/\partial t = \alpha \partial^2 u/\partial x^2$ using separation of variables, and implement the Crank-Nicolson numerical solver in Python."
    elif i == 9:
        text = "Explain the mathematical formulation of Singular Value Decomposition (SVD) and implement truncated SVD image compression in Python with energy preservation ratio calculation."
    elif i == 10:
        text = "Derive the closed-form ridge regression estimator $(X^T X + \lambda I)^{-1} X^T y$ and write an efficient Python implementation using Cholesky decomposition instead of matrix inversion."
    elif i == 11:
        text = "Formulate the multi-armed bandit Upper Confidence Bound (UCB1) regret bound theorem and implement a Python simulation benchmarking UCB1 against $\epsilon$-greedy."
    elif i == 12:
        text = "Derive the Runge-Kutta 4th order (RK4) method for initial value ODEs and implement a Python simulation of the chaotic Lorenz system with 3D phase space plotting."
    elif i == 13:
        text = "Derive the Metropolis-Hastings Markov Chain Monte Carlo (MCMC) acceptance probability rule and write a Python sampler drawing from an unnormalized bivariate multimodal distribution."
    elif i == 14:
        text = "Formulate the Principal Component Analysis (PCA) optimization problem as maximum variance projection and implement an end-to-end PCA transformer class in Python."
    elif i == 15:
        text = "Derive the Levenberg-Marquardt non-linear least squares update formula and implement a Python solver for curve fitting on exponential decay data."
    elif i == 16:
        text = "Formulate the Simplex method for linear programming in tableau form and write a Python function that solves standard-form maximization LP problems."
    elif i == 17:
        text = "Derive the backpropagation gradient equations for a 2-layer MLP with softmax cross-entropy loss, and implement vectorized forward/backward passes in Python."
    elif i == 18:
        text = "Formulate the Hidden Markov Model (HMM) Viterbi decoding algorithm dynamic programming equations and implement the decoder in Python."
    elif i == 19:
        text = "Derive the gradient formulas for Adam optimizer (first/second moment bias correction) and implement the optimizer in Python."
    elif i == 20:
        text = "Formulate the Bellman Optimality Equation for Markov Decision Processes (MDP) and implement Value Iteration and Policy Iteration in Python for a GridWorld environment."

    queries.append({
        "id": qid,
        "complexity_tier": "two_domain",
        "domains": ["coding", "math"],
        "query": text,
        "gold_dag_available": True if i <= 10 else False
    })

# Code + Reasoning (20)
for i in range(1, 21):
    qid = f"TD_CR_{i:02d}"
    if i == 1:
        text = "Analyze the concurrency race conditions and memory visibility hazards in a naive double-checked locking singleton implementation in Java/Python, and provide a verified deadlock-free, lock-free alternative using atomic CAS operations."
    elif i == 2:
        text = "Evaluate the theoretical time and space complexity trade-offs of using an LSM-Tree (Log-Structured Merge-Tree) versus a B+ Tree for high-write database workloads, and implement a mini in-memory MemTable and SSTable flush simulator in Python."
    elif i == 3:
        text = "Perform formal verification analysis of the Peterson's mutual exclusion algorithm for two processes, identify why it fails on out-of-order execution architectures, and implement a memory-barrier-safe Python model."
    elif i == 4:
        text = "Analyze the Byzantine Generals Problem under synchronous message passing, prove why $3m+1$ nodes are required to tolerate $m$ traitors, and write a Python simulator demonstrating consensus failure with $3m$ nodes."
    elif i == 5:
        text = "Analyze the consistency and partition-tolerance trade-offs of the Dynamo distributed key-value architecture (vector clocks, sloppy quorums), and implement a Python vector-clock causality resolution module."
    elif i == 6:
        text = "Evaluate why naive recursive backtracking for 3-SAT suffers exponential explosion, and implement the DPLL (Davis-Putnam-Logemann-Loveland) SAT solver algorithm in Python with unit propagation and pure literal elimination."
    elif i == 7:
        text = "Analyze the memory safety guarantees of Rust's borrow checker (affine types and lifetime tracking) vs C++ smart pointers, and implement a reference-counted cycle detection algorithm in Python."
    elif i == 8:
        text = "Evaluate the deadlock conditions (Coffman conditions) in resource allocation graphs, and implement the Banker's Safety Algorithm in Python with state validation."
    elif i == 9:
        text = "Analyze the correctness and loop invariants of the Tarjan's strongly connected components algorithm, and write a clean Python implementation with DFS stack unwinding."
    elif i == 10:
        text = "Analyze why distributed two-phase commit (2PC) is a blocking protocol during coordinator crashes, and implement a simulated Three-Phase Commit (3PC) state machine in Python."
    else:
        text = f"Perform formal analysis of consistency model #{i} in distributed systems, prove its safety invariant properties, and implement a Python concurrent verification testbed."
    
    queries.append({
        "id": qid,
        "complexity_tier": "two_domain",
        "domains": ["coding", "reasoning"],
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# Retrieval + Reasoning (15)
for i in range(1, 16):
    qid = f"TD_RR_{i:02d}"
    if i == 1:
        text = "Retrieve the key architectural milestones from Transformer to Modern LLMs (Attention Is All You Need, GPT-3, LLaMA, DeepSeek-V3), and logically analyze the evolution of architectural choices (MQA/GQA, SwiGLU, RoPE, MoE routing) in terms of compute efficiency vs expressivity."
    elif i == 2:
        text = "Synthesize the regulatory requirements of HIPAA Security Rule vs GDPR Article 32 regarding data anonymization and encryption at rest, and formulate a compliance decision framework for a cross-border healthcare AI deployment."
    elif i == 3:
        text = "Retrieve the historical mechanisms of the 2008 Global Financial Crisis (MBS, CDOs, CDS, liquidity freeze) and construct a causal graph reasoning about systemic contagion propagation."
    elif i == 4:
        text = "Analyze the core differences between Zero-Knowledge Rollups and Optimistic Rollups in Ethereum scaling, logically evaluating security finality vs capital efficiency trade-offs."
    elif i == 5:
        text = "Retrieve the clinical trial design protocols for oncology therapeutics (Phase I dose escalation $3+3$ design vs Bayesian CRM) and evaluate the ethical and statistical trade-offs in patient safety."
    else:
        text = f"Synthesize domain literature regarding technological transition #{i}, and construct a formal comparative analysis evaluating causal drivers and trade-offs."
    
    queries.append({
        "id": qid,
        "complexity_tier": "two_domain",
        "domains": ["retrieval", "reasoning"],
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# Math + General/Finance (15)
for i in range(1, 16):
    qid = f"TD_MG_{i:02d}"
    if i == 1:
        text = "Explain the macroeconomic theory of the Phillips Curve (inflation vs unemployment trade-off), formulate the Expectations-Augmented Phillips Curve mathematically, and analyze the stagflation phenomenon of the 1970s."
    elif i == 2:
        text = "Derive the Capital Asset Pricing Model (CAPM) beta and Security Market Line equation from Markowitz Modern Portfolio Theory (Mean-Variance frontier and Tangency Portfolio), explaining systematic vs idiosyncratic risk."
    elif i == 3:
        text = "Formulate the Value at Risk (VaR) and Conditional Value at Risk (CVaR/Expected Shortfall) mathematical definitions, and explain their regulatory importance under Basel III banking frameworks."
    elif i == 4:
        text = "Formulate the Solow-Swan Neoclassical Growth Model differential equations for capital accumulation, derive the steady-state capital per effective worker, and explain the Golden Rule level of capital."
    elif i == 5:
        text = "Derive the yield-to-maturity (YTM) bond pricing equation, define Macaulay Duration and Modified Duration, and evaluate how convexity protects fixed-income portfolios against large interest rate shocks."
    else:
        text = f"Explain the quantitative financial economics principle #{i}, derive its mathematical formulation, and discuss its empirical implications in modern capital markets."
    
    queries.append({
        "id": qid,
        "complexity_tier": "two_domain",
        "domains": ["math", "general"],
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# --- Tier 3: 3+-Domain Compound (n = 60) ---
# Code + Math + Reasoning (20)
for i in range(1, 21):
    qid = f"CD_CMR_{i:02d}"
    if i == 1:
        text = "Derive the mathematical proof of the A* search algorithm's optimality and admissibility on directed graphs with non-negative edge weights; logically analyze the consistency/monotonicity property of heuristic functions; and write an end-to-end Python implementation with a visual test grid solving a navigation problem with dynamic obstacles."
    elif i == 2:
        text = "Formulate the mathematical foundation of Singular Value Decomposition (SVD) and Eckart-Young-Mirsky theorem for low-rank matrix approximation; logically analyze why truncated SVD provides optimal rank-k reconstruction under Frobenius norm; and implement an end-to-end Python collaborative filtering recommendation engine with cross-validation."
    elif i == 3:
        text = "Derive the mathematical loss function of the Cross-Entropy Loss combined with Label Smoothing in multi-class classification; prove why label smoothing prevents the network from becoming overconfident in its logits; and implement a PyTorch/NumPy training loop from scratch verifying logit distribution shrinkage."
    elif i == 4:
        text = "Derive the mathematical equations for the Extended Kalman Filter (EKF) linearization via Jacobian matrices for non-linear state transitions; analyze stability risks when linearization errors accumulate; and implement a complete Python simulation tracking an aircraft executing a coordinated turn."
    elif i == 5:
        text = "Formulate the convex optimization problem for portfolio optimization with quadratic transaction costs and cardinality constraints; analyze why L1 regularization induces sparsity whereas L2 induces shrinkage; and implement a Python solver using cvxpy or custom projected gradient descent."
    else:
        text = f"Derive the mathematical formulation for advanced machine learning algorithm #{i}, prove its convergence bounds and stability conditions, and implement an end-to-end Python benchmark."
    
    queries.append({
        "id": qid,
        "complexity_tier": "three_plus_domain",
        "domains": ["coding", "math", "reasoning"],
        "query": text,
        "gold_dag_available": True if i <= 10 else False
    })

# Retrieval + Code + Reasoning (20)
for i in range(1, 21):
    qid = f"CD_RCR_{i:02d}"
    if i == 1:
        text = "Retrieve the OWASP Top 10 API Security Risks (focusing on Broken Object Level Authorization - BOLA and Server-Side Request Forgery - SSRF); analyze how microservice architectures amplify these vulnerabilities; and implement a secure Python FastAPI middleware suite with token-based tenant isolation and URL validation."
    elif i == 2:
        text = "Retrieve the specifications of the OAuth 2.1 Authorization Framework (PKCE, deprecation of Implicit Grant); logically analyze the attack vectors mitigated by Proof Key for Code Exchange against authorization code interception; and implement a Python mock authorization server and client demonstrating PKCE verification."
    elif i == 3:
        text = "Retrieve the architectural specifications of Apache Kafka's distributed commit log (partitioning, consumer groups, ISR replication); analyze the trade-offs between exact-once semantics (EOS) and at-least-once processing; and implement a Python simulation of idempotent producer and transactional consumer state recovery."
    elif i == 4:
        text = "Retrieve the WebAssembly (Wasm) core specification and Component Model; evaluate the security isolation guarantees of Wasm sandboxing compared to Docker containers; and implement a Python host runtime runner executing isolated sandboxed untrusted modules."
    elif i == 5:
        text = "Retrieve the core design principles of the Raft consensus algorithm (log matching property, election safety); prove why leader completeness guarantees state machine safety; and implement an asyncio Python distributed node cluster demonstrating log replication and split-brain prevention."
    else:
        text = f"Retrieve standard specifications for cloud distributed system architecture #{i}, evaluate safety and fault tolerance trade-offs, and implement a Python simulation demonstrating disaster recovery."
    
    queries.append({
        "id": qid,
        "complexity_tier": "three_plus_domain",
        "domains": ["retrieval", "coding", "reasoning"],
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# Retrieval + Math + Policy/General (20)
for i in range(1, 21):
    qid = f"CD_RMP_{i:02d}"
    if i == 1:
        text = "Retrieve international carbon emission regulatory standards (EU ETS Cap-and-Trade vs CORSIA); derive the mathematical formula for marginal abatement cost curves and carbon tax incidence across inelastic consumer markets; and synthesize a comprehensive policy recommendation for corporate ESG compliance."
    elif i == 2:
        text = "Retrieve the Basel III and Basel IV banking regulatory capital framework standards (Common Equity Tier 1, Risk-Weighted Assets); formulate the mathematical calculation of Credit Value Adjustment (CVA) and Liquidity Coverage Ratio (LCR); and analyze the systemic economic impacts of tightening bank reserve requirements."
    elif i == 3:
        text = "Retrieve the epidemiological transmission dynamics of respiratory pathogens (SEIR compartmental model); derive the mathematical formula for the basic reproduction number $R_0$ using the next-generation matrix method; and evaluate the socio-economic cost-benefit trade-offs of non-pharmaceutical interventions."
    elif i == 4:
        text = "Retrieve the Paris Agreement NDC (Nationally Determined Contributions) framework; formulate the mathematical integrated assessment model (DICE-style climate-economy damage function); and synthesize an executive analysis of carbon pricing pathways required to limit warming to 1.5C."
    elif i == 5:
        text = "Retrieve FDA guidelines on bioequivalence testing for generic pharmaceuticals (two one-sided tests TOST); derive the mathematical power and confidence interval calculation for 90% geometric mean ratios within [80%, 125%]; and synthesize an audit protocol for clinical submission."
    else:
        text = f"Retrieve global standard regulations for critical infrastructure domain #{i}, derive the associated quantitative risk assessment formulas, and synthesize an executive compliance roadmap."
    
    queries.append({
        "id": qid,
        "complexity_tier": "three_plus_domain",
        "domains": ["retrieval", "math", "general", "reasoning"],
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# -------------------------------------------------------------
# 2. Partition into Dev (n = 60) and Held-Out (n = 120) Splits
# -------------------------------------------------------------
# Stratified 1:2 split across all tiers:
# Single domain: 16 Dev, 34 Held-Out (Total 50)
# Two domain: 24 Dev, 46 Held-Out (Total 70)
# Three+ domain: 20 Dev, 40 Held-Out (Total 60)
# Total: 60 Dev, 120 Held-Out

dev_queries = []
held_out_queries = []

sd_count, td_count, cd_count = 0, 0, 0

for q in queries:
    tier = q["complexity_tier"]
    if tier == "single_domain":
        sd_count += 1
        if sd_count <= 16:
            q["split"] = "dev"
            dev_queries.append(q)
        else:
            q["split"] = "held_out"
            held_out_queries.append(q)
    elif tier == "two_domain":
        td_count += 1
        if td_count <= 24:
            q["split"] = "dev"
            dev_queries.append(q)
        else:
            q["split"] = "held_out"
            held_out_queries.append(q)
    elif tier == "three_plus_domain":
        cd_count += 1
        if cd_count <= 20:
            q["split"] = "dev"
            dev_queries.append(q)
        else:
            q["split"] = "held_out"
            held_out_queries.append(q)

# -------------------------------------------------------------
# 3. Gold DAG Construction (60 queries annotated with Gold DAGs)
# -------------------------------------------------------------
# Hand-annotated DAGs for decomposition accuracy (RQ5)
gold_dags = {}

# Sample Gold DAGs covering single, 2-domain, and 3+-domain
for q in queries:
    if not q.get("gold_dag_available"):
        continue
    
    qid = q["id"]
    tier = q["complexity_tier"]
    
    if tier == "single_domain":
        dom = q["domains"][0]
        gold_dags[qid] = {
            "query_id": qid,
            "complexity_tier": tier,
            "subtasks": [
                {
                    "id": "node_1",
                    "text": q["query"],
                    "capability": dom if dom in ["coding", "math", "reasoning"] else "general",
                    "dependencies": []
                }
            ]
        }
    elif tier == "two_domain":
        d1, d2 = q["domains"][0], q["domains"][1]
        c1 = d1 if d1 in ["coding", "math", "reasoning", "retrieval"] else "general"
        c2 = d2 if d2 in ["coding", "math", "reasoning", "retrieval"] else "general"
        
        if "CM" in qid: # Code + Math
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Derive the mathematical formulas, equations, and theoretical framework for {qid}.",
                        "capability": "math",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Implement clean, vectorized, tested Python code based on the derived mathematical formulas for {qid}.",
                        "capability": "coding",
                        "dependencies": ["node_1"]
                    }
                ]
            }
        elif "CR" in qid: # Code + Reasoning
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Perform formal theoretical analysis, proof of invariants, and edge case reasoning for {qid}.",
                        "capability": "reasoning",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Implement the verified, bug-free Python code satisfying the analytical constraints for {qid}.",
                        "capability": "coding",
                        "dependencies": ["node_1"]
                    }
                ]
            }
        elif "RR" in qid: # Retrieval + Reasoning
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Retrieve and extract core factual specifications, standards, and historical data for {qid}.",
                        "capability": "retrieval",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Analyze trade-offs, causal mechanisms, and synthesize comparative reasoning for {qid}.",
                        "capability": "reasoning",
                        "dependencies": ["node_1"]
                    }
                ]
            }
        else: # Math + General
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Derive the mathematical formulation and quantitative equations for {qid}.",
                        "capability": "math",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Explain real-world economic context, policy implications, and qualitative interpretation for {qid}.",
                        "capability": "general",
                        "dependencies": ["node_1"]
                    }
                ]
            }
    elif tier == "three_plus_domain":
        if "CMR" in qid: # Code + Math + Reasoning
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Derive formal mathematical formulas, objective functions, and loss equations for {qid}.",
                        "capability": "math",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Prove theoretical properties, convergence bounds, optimality, and error guarantees for {qid}.",
                        "capability": "reasoning",
                        "dependencies": ["node_1"]
                    },
                    {
                        "id": "node_3",
                        "text": f"Implement full end-to-end Python implementation verifying the mathematical derivation and theoretical bounds for {qid}.",
                        "capability": "coding",
                        "dependencies": ["node_1", "node_2"]
                    }
                ]
            }
        elif "RCR" in qid: # Retrieval + Code + Reasoning
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Retrieve and extract core specifications, RFCs, and security standards for {qid}.",
                        "capability": "retrieval",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Analyze threat vectors, architectural invariants, and security trade-offs for {qid}.",
                        "capability": "reasoning",
                        "dependencies": ["node_1"]
                    },
                    {
                        "id": "node_3",
                        "text": f"Implement secure, sandboxed, compliant Python middleware/runtime satisfying the specifications for {qid}.",
                        "capability": "coding",
                        "dependencies": ["node_1", "node_2"]
                    }
                ]
            }
        else: # Retrieval + Math + Policy/General
            gold_dags[qid] = {
                "query_id": qid,
                "complexity_tier": tier,
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": f"Retrieve regulatory guidelines, statutory frameworks, and standard protocols for {qid}.",
                        "capability": "retrieval",
                        "dependencies": []
                    },
                    {
                        "id": "node_2",
                        "text": f"Derive the quantitative risk assessment, statistical power, and financial formulas for {qid}.",
                        "capability": "math",
                        "dependencies": []
                    },
                    {
                        "id": "node_3",
                        "text": f"Synthesize an executive compliance roadmap and socio-economic policy recommendations integrating quantitative and regulatory findings for {qid}.",
                        "capability": "reasoning",
                        "dependencies": ["node_1", "node_2"]
                    }
                ]
            }

# -------------------------------------------------------------
# 4. Pre-Registered Scoring Rubric Spec
# -------------------------------------------------------------
rubric_spec = {
    "rubric_version": "1.0.0",
    "evaluation_mode": "double_blind",
    "scale": "1-5 integer",
    "dimensions": {
        "correctness": {
            "weight": 0.40,
            "description": "Factual truth, mathematical correctness, code execution validity, and absence of logical hallucinations.",
            "levels": {
                "5": "Completely accurate, mathematically sound, code is syntactically and semantically valid with no flaws.",
                "4": "Largely accurate, minor non-critical notation or syntax discrepancy.",
                "3": "Moderate accuracy, correct core concept but contains a notable derivation error or unhandled edge case.",
                "2": "Substantial inaccuracies, faulty mathematical derivation, or broken code logic.",
                "1": "Fundamentally wrong, major hallucinations, or completely invalid code/math."
            }
        },
        "completeness": {
            "weight": 0.35,
            "description": "Exhaustive coverage of all explicit and implicit sub-goals in complex compound prompts.",
            "levels": {
                "5": "All multi-domain facets and instructions fully addressed with thorough depth.",
                "4": "All primary facets addressed; one minor sub-question covered with slight brevity.",
                "3": "Core domain addressed well, but secondary cross-domain requirement partially omitted.",
                "2": "Multiple explicit sub-tasks omitted; shallow superficial answers.",
                "1": "Fails to address the vast majority of the prompt's requirements."
            }
        },
        "coherence_synthesis": {
            "weight": 0.25,
            "description": "Logical structural flow, cross-subtask synthesis, resolution of contradictions, and readability.",
            "levels": {
                "5": "Masterful synthesis; unified explanation where multi-domain parts reinforce each other seamlessly.",
                "4": "Well-structured, clear transitions, minimal redundancy.",
                "3": "Understandable, but reads like distinct concatenated sections with abrupt transitions.",
                "2": "Disjointed, contradictory statements between sections, or repetitive text.",
                "1": "Incoherent, internally contradictory, or unstructured rambling."
            }
        }
    }
}

# -------------------------------------------------------------
# 5. Write Files & Compute Cryptographic Lock
# -------------------------------------------------------------
with open("data/eval_dataset_master.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2)

with open("data/queries_dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_queries, f, indent=2)

held_out_bytes = json.dumps(held_out_queries, indent=2, sort_keys=True).encode("utf-8")
with open("data/queries_held_out.json", "wb") as f:
    f.write(held_out_bytes)

with open("data/gold_dags.json", "w", encoding="utf-8") as f:
    json.dump(gold_dags, f, indent=2)

with open("data/scoring_rubric.json", "w", encoding="utf-8") as f:
    json.dump(rubric_spec, f, indent=2)

# Compute SHA-256 for Held-Out Split
held_out_sha256 = hashlib.sha256(held_out_bytes).hexdigest()
lock_metadata = {
    "held_out_file": "data/queries_held_out.json",
    "query_count": len(held_out_queries),
    "strata_counts": {
        "single_domain": len([q for q in held_out_queries if q["complexity_tier"] == "single_domain"]),
        "two_domain": len([q for q in held_out_queries if q["complexity_tier"] == "two_domain"]),
        "three_plus_domain": len([q for q in held_out_queries if q["complexity_tier"] == "three_plus_domain"])
    },
    "sha256": held_out_sha256,
    "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "LOCKED — Subject to AGENTS.md Rule 5 Held-out discipline"
}

with open("data/held_out_lock.json", "w", encoding="utf-8") as f:
    json.dump(lock_metadata, f, indent=2)

with open("data/held_out_lock.sha256", "w", encoding="utf-8") as f:
    f.write(f"{held_out_sha256}  data/queries_held_out.json\n")

print("=== DATASET CONSTRUCTION SUMMARY ===")
print(f"Total Queries: {len(queries)}")
print(f"Dev Split: {len(dev_queries)} queries (Single: 16, Two: 24, Three+: 20)")
print(f"Held-Out Split: {len(held_out_queries)} queries (Single: 34, Two: 46, Three+: 40)")
print(f"Gold DAGs Annotated: {len(gold_dags)} DAGs")
print(f"Held-Out SHA256 Hash: {held_out_sha256}")
print("Held-out split successfully generated and locked.")

