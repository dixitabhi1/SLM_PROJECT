"""
Deterministic Retrieval Tool for Factual Specialists
AI Search Framework - Version 4 Step 2
Provides grounded, zero-LLM citation lookup from pinned reference corpora.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

DEFAULT_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "corpora", "security_standards_corpus.json"
)

class DeterministicRetrievalTool:
    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = os.path.abspath(corpus_path or DEFAULT_CORPUS_PATH)
        self.documents = []
        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Reference corpus not found at {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def query(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
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
            doc_keys = " ".join(doc.get("key_mechanisms", [])).lower()
            combined_text = f"{doc_id} {doc_title} {doc_summary} {doc_keys}"

            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", combined_text))
            overlap = len(tokens.intersection(doc_tokens))
            
            # Boost exact ID matches (e.g. 'rfc', 'cve', 'socket')
            boost = 0
            if "rfc" in tokens and "rfc" in doc_id:
                boost += 3
            if "socket" in tokens and ("socket" in doc_id or "socket" in doc_title):
                boost += 3
            if "vulnerability" in tokens and "cve" in doc_id:
                boost += 3
            if "linux" in tokens and "linux" in combined_text:
                boost += 2

            score = overlap + boost
            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def format_grounded_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved documents into an immutable, grounded reference block
        to inject into the specialist prompt.
        """
        if not retrieved_docs:
            return "No verified documents retrieved from the authoritative reference corpus."

        blocks = ["### VERIFIED REFERENCE CORPUS (Authoritative Source Data):"]
        for idx, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", "UNKNOWN")
            title = doc.get("title", "Untitled")
            summary = doc.get("summary", "")
            mechanisms = doc.get("key_mechanisms") or doc.get("mitigations") or []

            block = f"[{idx}] Source ID: {doc_id} - '{title}'\nSummary: {summary}"
            if mechanisms:
                block += "\nKey Specifications/Mitigations:\n" + "\n".join(f"  - {m}" for m in mechanisms)
            blocks.append(block)

        blocks.append("\nInstruction to Specialist: Ground your technical response directly in the verified source IDs and specifications above. Cite exact RFC numbers and CVE identifiers verbatim.")
        return "\n\n".join(blocks)

