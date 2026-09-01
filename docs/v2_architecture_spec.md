# AI Search Framework v2: Target Architecture & Resolution Specification

**Phase:** v2.1 Architecture Resolution  
**Status:** Complete & User-Confirmed  
**Date:** 2026-09-01  

---

## 1. Executive Summary & Resolution Record

In Phase v2.1, all structural and operational open questions from [`Proposed_Architecture_v2_source.txt`](file:///c:/Users/ACER/Downloads/antigravity-ai-search-skills%20(1)/v2/antigravity-ai-search-v2-additions/.agents/knowledge/Proposed_Architecture_v2_source.txt) were systematically presented and resolved with user decisions:

| Open Question | Resolved Decision | Specification Detail |
| :--- | :--- | :--- |
| **Q1: Color Taxonomy** | **Option A (5 Domain Colors)** | `Color 1 (Blue)`: Coding, `Color 2 (Green)`: Math, `Color 3 (Purple)`: Logic/Reasoning, `Color 4 (Amber)`: Retrieval, `Color 5 (Slate)`: General. |
| **Q2: Recursion Depth Limit** | **Option C (`max_depth = 3`)** | Initial decomposition is Depth 0. Re-decomposition loops back up to 3 times for multi-color tasks before forcing multi-agent collaboration. |
| **Q3: Task ID Versioning** | **Option A (Hierarchical Dot Notation)** | Parent `node_1` decomposes into child subtasks `node_1.1`, `node_1.2`. Lineage and dependency rewiring are explicit. |
| **Q4: Scheduling Cost Model** | **Option A (Fixed Pricing + Concurrency Cap)** | Uses versioned `pricing_table.json` and hardware concurrency limits (`max_concurrent_slms`). |
| **Q5: Aggregator Placement** | **Option B (Two-Stage Aggregation)** | Local collaboration synthesis for multi-agent tasks + Global Aggregator SLM (&le; 8B) for terminal cross-task fusion. |

---

## 2. End-to-End v2 Proposed Workflow

```
                                    User Query
                                         |
                                         v
==================================== BRANCH A: Task Side ====================================
                               Decomposer SLM (<= 3B)
                             [Decomposes prompt into tasks]
                                         |
                                         v
                               SLM-2: Task Analyser
                            [Skill vector of each task]
                                         |
                                         v
                                SLM-3: Task Colorer
                            [Colors tasks by similarity]
                                         |
                                         +-----------------------+
                                                                 |
=================================== BRANCH B: Agent Side ========|===========================
                                  SLM Agent Pool                 |
                                         |                       |
                                         v                       |
                                Agent Analyser SLM               |
                          [Skill vector, once per agent]         |
                                         |                       |
                                         v                       |
                                 Agent Colorer SLM               |
                         [Multi-skill = bridges colors]          |
                                         |                       |
                                         +-----------+-----------+
                                                     |
                                                     v
======================================= CONVERGENCE =========================================
                                       Matching SLM
              - Within each color, matches which agent fits which task
              - Marks whether matched tasks can run in parallel or must run in series
              - Evaluates Loop Condition:
                     spans_multiple_colors AND depth < max_depth (max_depth=3)
                                         |
                    +--------------------+--------------------+
                    | (YES: Loop-Back)                        | (NO: Single-color OR Depth=3)
                    v                                         v
         [Loop to Decomposer SLM]                       Scheduling SLM
        (Subtasks get node_1.1, etc.)      - Builds execution graph from assignments
                                           - Arranges parallel vs. series by cost & availability
                                           - Multi-color tasks at depth limit get multi-agent
                                             collaboration with local mini-aggregation
                                                              |
                                                              v
                                                   Execution Graph Runner
                                              (Dispatches single/multi-agent SLMs)
                                                              |
                                                              v
                                              Global Aggregator SLM (<= 8B)
                                         (Terminal Fusion & Contradiction Resolution)
                                                              |
                                                              v
                                                        Final Response
```

---

## 3. Detailed Component Contracts

### 3.1 Branch A: Task Processing
1. **Decomposer SLM (&le; 3B):** Decomposes user prompt into initial tasks ($D=0$) or re-decomposes multi-color tasks ($D \in \{1, 2, 3\}$).
2. **SLM-2 (Task Analyser):** Computes a 5-dimensional normalized skill requirement vector:
   $$\mathbf{s}_{\text{task}} = \langle s_{\text{code}}, s_{\text{math}}, s_{\text{logic}}, s_{\text{retrieval}}, s_{\text{general}} \rangle, \quad \sum s_i = 1.0$$
3. **SLM-3 (Task Colorer):** Assigns dominant color categories where $s_i \ge \theta_{\text{color}}$ ($\theta = 0.20$). If more than one dimension exceeds threshold, the task is marked as `multi-color`.

### 3.2 Branch B: Agent Processing
1. **SLM Agent Pool:** The 5 specialized domain SLMs (`Qwen2.5-Coder-7B`, `Qwen2.5-Math-7B`, `Phi-3.5-mini`, `Llama-3.2-3B`, `Llama-3.1-8B`).
2. **Agent Analyser SLM:** Generates static capability vectors $\mathbf{a}_j$ for each pool model (cached after initial computation).
3. **Agent Colorer SLM:** Assigns color tags and bridging flags for models supporting multiple secondary capabilities.

### 3.3 Convergence & Bounded Feedback Loop
1. **Matching SLM:** Computes agent-task fit $\text{sim}(\mathbf{s}_{\text{task}}, \mathbf{a}_j)$ within color classes.
2. **Loop Condition:**
   ```python
   if task.is_multi_color and task.depth < 3:
       # Trigger re-decomposition: emit child tasks node_X.1, node_X.2
       loop_to_decomposer(task, next_depth=task.depth + 1)
   else:
       # Forward to scheduling
       forward_to_scheduling(task)
   ```
3. **Scheduling SLM:** Builds topological execution order, budgets parallel concurrency against `max_concurrent_slms`, and assigns collaborative multi-agent teams to tasks that reached `depth = 3` while multi-color.

### 3.4 Two-Stage Aggregation
1. **Local Collaboration Aggregator:** When a multi-agent team executes a complex node at depth limit, intermediate agent outputs are synthesized locally into a unified subtask answer.
2. **Global Terminal Aggregator (&le; 8B):** Synthesizes all completed subtask outputs into the final authoritative answer, explicitly verifying consistency and resolving contradictions.

---

## 4. Run-Log Schema Extension for v2

Every execution trace records:
* `task_id` (hierarchical string, e.g. `node_1.2`)
* `depth` (integer $0 \le D \le 3$)
* `skill_vector` ($\mathbf{s}_{\text{task}}$ at each iteration)
* `assigned_colors` (list of color tags)
* `loop_events` (timestamps, parent ID, trigger reasons)
* `execution_type` (`single_agent_match` vs `multi_agent_collaboration_at_limit`)
* `local_aggregation_output` (if multi-agent)
* `global_aggregation_io` (inputs and final response)

