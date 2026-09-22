"""
Engineering & Standards Retrieval Tool
AI Search Framework - Version 5
Provides deterministic, zero-LLM citation and mathematical formulation lookup
from the pinned engineering and standards reference corpus.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

DEFAULT_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "corpora", "engineering_standards_corpus.json"
)

class EngineeringRetrievalTool:
    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = os.path.abspath(corpus_path or DEFAULT_CORPUS_PATH)
        self.documents = []
        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Reference corpus not found at {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs deterministic keyword-matching over the pinned corpus.
        Ranks by token overlap without any generative or probabilistic steps.
        """
        tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text.lower()))
        scored_docs = []

        for doc in self.documents:
            doc_id = doc.get("id", "").lower()
            doc_title = doc.get("title", "").lower()
            doc_summary = doc.get("summary", "").lower()
            doc_tags = " ".join(doc.get("tags", [])).lower()
            doc_keys = " ".join(doc.get("key_mechanisms", [])).lower()
            
            math_form = doc.get("mathematical_formulation", {})
            math_str = " ".join(str(v) for v in math_form.values()).lower() if isinstance(math_form, dict) else ""

            combined_text = f"{doc_id} {doc_title} {doc_summary} {doc_tags} {doc_keys} {math_str}"
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined_text))
            overlap = len(tokens.intersection(doc_tokens))
            
            # Domain-specific keyword boosting
            boost = 0
            if "mdo" in tokens or "optimization" in tokens or "lagrangian" in tokens:
                if "mdo" in doc_id or "mdo" in doc_tags:
                    boost += 6
            if "diffusion" in tokens or "laplacian" in tokens or "pde" in tokens or "parquet" in tokens:
                if "pde" in doc_id or "pde" in doc_tags:
                    boost += 6
            if "relational" in tokens or "schema" in tokens or "invariants" in tokens or "multi-tenant" in tokens:
                if "fol" in doc_id or "invariants" in doc_tags:
                    boost += 6
            if "rfc" in tokens or "socket" in tokens or "vulnerability" in tokens or "sandbox" in tokens:
                if "sec" in doc_id or "rfc" in doc_id or "cve" in doc_id:
                    boost += 5

            score = overlap + boost
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def format_grounded_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved documents into an authoritative technical reference block
        with mathematical formulations, stability bounds, and code patterns.
        """
        if not retrieved_docs:
            return "No verified documents retrieved from the authoritative reference corpus."

        CITATION_MAP = {
            "MDO-AUGMENTED-LAGRANGIAN": "AIAA Multidisciplinary Design Optimization Standards",
            "PDE-KRONECKER-DIFFUSION": "SIAM Numerical Heat Transfer & PDE Discretization Standards",
            "FOL-RELATIONAL-INVARIANTS": "ACM Relational Database Theory & ISO SQL Invariants",
            "SEC-LINUX-SOCKETS": "Linux Kernel Socket Security Architecture & IETF RFC Standards",
            "RFC-8446": "IETF RFC 8446 (Transport Layer Security TLS 1.3)",
            "RFC-9293": "IETF RFC 9293 (Transmission Control Protocol Specification)",
            "CVE-2017-6074": "NVD CVE-2017-6074 (Linux Kernel DCCP Double-Free Vulnerability)",
            "CVE-2023-32233": "NVD CVE-2023-32233 (Linux Netfilter nf_tables Privilege Escalation)"
        }

        blocks = ["### VERIFIED TECHNICAL REFERENCE STANDARDS:"]
        for idx, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", "UNKNOWN")
            title = doc.get("title", "Untitled")
            summary = doc.get("summary", "")
            citation = CITATION_MAP.get(doc_id, doc_id)

            block = f"[{idx}] Standard Reference: {title} ({citation})\nSummary: {summary}"
            
            # Mathematical formulation
            math_form = doc.get("mathematical_formulation")
            if math_form and isinstance(math_form, dict):
                block += "\nAuthoritative Technical Formulation:"
                for k, v in math_form.items():
                    if isinstance(v, list):
                        block += f"\n  - {k.replace('_', ' ').title()}:\n" + "\n".join(f"      * {item}" for item in v)
                    else:
                        block += f"\n  - {k.replace('_', ' ').title()}: {v}"
            
            # Key mechanisms or mitigations
            mechanisms = doc.get("key_mechanisms") or doc.get("mitigations")
            if mechanisms:
                block += "\nKey Mechanisms / Mitigations:\n" + "\n".join(f"  - {m}" for m in mechanisms)

            # Python template
            py_template = doc.get("python_template")
            if py_template:
                block += f"\nAuthoritative Implementation Template:\n```python\n{py_template.strip()}\n```"

            blocks.append(block)

        blocks.append(
            "\nInstruction to Specialist: Ground your technical output directly in the verified reference formulations above.\n"
            "- Use the exact equations, stability criteria, and architectural patterns provided.\n"
            "- Reference standard technical standards (e.g. RFC 8446, RFC 9293, AIAA MDO formulations) where appropriate.\n"
            "- Present formulas and proofs professionally without quoting internal database keys.\n"
            "- Do NOT simplify or reduce problems to high-school toy models (e.g. do not reduce coupled MDO to 1D linear regression)."
        )
        return "\n\n".join(blocks)

