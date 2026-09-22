"""
Coding Specialist Deterministic Retrieval Tool
AI Search Framework - Version 5 Step A
Provides zero-LLM deterministic implementation patterns and sandboxing primitives
from the pinned coding reference corpus.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

DEFAULT_CODING_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "corpora", "coding_reference_corpus.json"
)

class CodingRetrievalTool:
    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = os.path.abspath(corpus_path or DEFAULT_CODING_CORPUS_PATH)
        self.documents = []
        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Coding reference corpus not found at {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def query(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Deterministic keyword overlap matching over the pinned coding corpus.
        Zero LLM or probabilistic ranking.
        """
        tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text.lower()))
        scored_docs = []

        for doc in self.documents:
            doc_id = doc.get("id", "").lower()
            doc_title = doc.get("title", "").lower()
            doc_summary = doc.get("summary", "").lower()
            doc_tags = " ".join(doc.get("tags", [])).lower()

            combined_text = f"{doc_id} {doc_title} {doc_summary} {doc_tags}"
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined_text))
            overlap = len(tokens.intersection(doc_tokens))

            # Domain-specific keyword boosting and cross-domain suppression
            boost = 0
            is_mdo_query = any(k in tokens for k in ["mdo", "augmented", "lagrangian", "benchmark", "multidisciplinary"])
            is_pde_query = any(k in tokens for k in ["diffusion", "laplacian", "pde", "parquet", "kronecker", "thermodynamic"])
            is_sec_query = any(k in tokens for k in ["sandbox", "namespaces", "seccomp", "socket", "privilege"])

            if is_mdo_query:
                if "mdo" in doc_id:
                    boost += 30
                else:
                    boost -= 100
            elif is_pde_query:
                if "parquet" in doc_id or "kronecker" in doc_id or "diffusion" in doc_id:
                    boost += 30
                else:
                    boost -= 100
            elif is_sec_query:
                if "sandbox" in doc_id:
                    boost += 30
                else:
                    boost -= 100

            score = overlap + boost
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def format_grounded_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved code templates into an authoritative implementation guide.
        """
        if not retrieved_docs:
            return "No verified implementation patterns retrieved from coding reference corpus."

        blocks = ["### VERIFIED CODING IMPLEMENTATION STANDARDS & PATTERNS:"]
        for idx, doc in enumerate(retrieved_docs, 1):
            title = doc.get("title", "Untitled Pattern")
            summary = doc.get("summary", "")
            py_template = doc.get("python_template", "").strip()

            block = f"[{idx}] Standard Pattern: {title}\nSummary: {summary}"
            if py_template:
                block += f"\nAuthoritative Implementation Template:\n```python\n{py_template}\n```"
            blocks.append(block)

        blocks.append(
            "\nInstruction to Coding Specialist:\n"
            "- Implement complete, executable Python code grounded in the authoritative pattern above.\n"
            "- Ensure all array dimensions, packages (numpy, scipy, pandas, pyarrow, socket), and loops run error-free.\n"
            "- For sandboxing: use genuine Linux primitives (namespaces, seccomp, capability dropping). Do NOT confuse virtual environments (venv) with security sandboxes."
        )
        return "\n\n".join(blocks)

