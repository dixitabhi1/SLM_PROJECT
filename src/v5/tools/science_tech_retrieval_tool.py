"""
Science & Technology Specialist Deterministic Retrieval Tool
AI Search Framework - Version 5 Step A
Provides zero-LLM deterministic mathematical formulations and stencil derivations
from the pinned science_tech reference corpus.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

DEFAULT_SCIENCE_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "corpora", "science_tech_reference_corpus.json"
)

class ScienceTechRetrievalTool:
    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = os.path.abspath(corpus_path or DEFAULT_SCIENCE_CORPUS_PATH)
        self.documents = []
        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Science-Tech reference corpus not found at {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def query(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Deterministic keyword overlap matching over the pinned science_tech corpus.
        Zero LLM or probabilistic ranking.
        """
        tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text.lower()))
        scored_docs = []

        for doc in self.documents:
            doc_id = doc.get("id", "").lower()
            doc_title = doc.get("title", "").lower()
            doc_summary = doc.get("summary", "").lower()
            doc_tags = " ".join(doc.get("tags", [])).lower()
            math_form = doc.get("mathematical_formulation", {})
            math_str = " ".join(str(v) for v in math_form.values()).lower() if isinstance(math_form, dict) else ""

            combined_text = f"{doc_id} {doc_title} {doc_summary} {doc_tags} {math_str}"
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined_text))
            overlap = len(tokens.intersection(doc_tokens))

            # Domain-specific keyword boosting and cross-domain suppression
            boost = 0
            is_pde_query = any(k in tokens for k in ["diffusion", "pde", "cfl", "stencil", "crank", "thermodynamic", "heat", "laplacian", "differential"])
            is_mdo_query = any(k in tokens for k in ["mdo", "augmented", "lagrangian", "convergence", "kkt", "multidisciplinary", "disciplinary"])

            if is_pde_query:
                if "diffusion" in doc_id:
                    boost += 25
                elif "kronecker" in doc_id:
                    boost += 20
                elif "mdo" in doc_id:
                    boost -= 100
            elif is_mdo_query:
                if "mdo" in doc_id:
                    boost += 25
                elif "diffusion" in doc_id or "kronecker" in doc_id:
                    boost -= 100

            score = overlap + boost
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def format_grounded_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved scientific formulations into a rigorous mathematical reference block.
        """
        if not retrieved_docs:
            return "No verified scientific standards retrieved from reference corpus."

        blocks = ["### VERIFIED SCIENTIFIC & MATHEMATICAL REFERENCE FORMULATIONS:"]
        for idx, doc in enumerate(retrieved_docs, 1):
            title = doc.get("title", "Untitled Reference")
            summary = doc.get("summary", "")
            math_form = doc.get("mathematical_formulation", {})

            block = f"[{idx}] Reference Standard: {title}\nSummary: {summary}"
            if math_form and isinstance(math_form, dict):
                block += "\nAuthoritative Equations & Stencil Derivations:"
                for k, v in math_form.items():
                    block += f"\n  - {k.replace('_', ' ').title()}: {v}"
            blocks.append(block)

        blocks.append(
            "\nInstruction to Science & Technology Specialist:\n"
            "- Ground your theoretical derivations, continuous PDE equations, and matrix stencils directly in the verified reference formulations above.\n"
            "- For Kronecker stencils, assert exact matrix dimensions: L_2D = (I_Ny (x) D_xx) + (D_yy (x) I_Nx) in R^{(Nx*Ny) x (Nx*Ny)}.\n"
            "- Explicitly derive the CFL stability condition and Crank-Nicolson formulation.\n"
            "- Do NOT confuse problem domains (e.g. do not introduce MDO equations into diffusion PDE tasks)."
        )
        return "\n\n".join(blocks)

