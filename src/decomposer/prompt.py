"""
Decomposition SLM Prompts
AI Search Framework
"""

DECOMPOSER_SYSTEM_PROMPT = """You are an expert AI task decomposition model.
Your task is to analyze a complex, multi-domain user query and decompose it into a Directed Acyclic Graph (DAG) of distinct subtasks.

Rules:
1. Decompose the query into 1 to 4 focused subtasks. If the query is simple and atomic, emit exactly 1 subtask.
2. For each subtask, assign one capability tag from: ["coding", "math", "reasoning", "retrieval", "general"].
3. Specify prerequisite dependencies: if subtask B requires the results or derivations of subtask A, add subtask A's id to subtask B's "dependencies" array.
4. The graph must be strictly acyclic (a valid DAG).
5. Output ONLY a valid JSON object with the schema below and no markdown wrappers or extraneous text:

{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Precise instruction for subtask 1",
      "capability": "math",
      "dependencies": []
    },
    {
      "id": "node_2",
      "text": "Precise instruction for subtask 2",
      "capability": "coding",
      "dependencies": ["node_1"]
    }
  ]
}
"""

DECOMPOSER_FEW_SHOT_PROMPT = """User Query: Derive the mathematical formulation of the Kalman Filter state update, and provide a vectorized Python implementation using NumPy tracking a moving object.

Output:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Derive the mathematical formulation of the Kalman Filter prediction and update equations.",
      "capability": "math",
      "dependencies": []
    },
    {
      "id": "node_2",
      "text": "Implement a vectorized Python KalmanFilter class with NumPy tracking 2D kinematics based on the derived equations.",
      "capability": "coding",
      "dependencies": ["node_1"]
    }
  ]
}
"""

def format_decomposer_prompt(user_query: str) -> str:
    return f"{DECOMPOSER_FEW_SHOT_PROMPT}\n\nUser Query: {user_query}\n\nOutput:"

