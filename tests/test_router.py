"""
Unit Tests for Capability Router
"""

from src.router.capability_router import CapabilityRouter

def test_router_coding():
    router = CapabilityRouter()
    res = router.route("Implement a Python async rate limiter using a sliding window algorithm.")
    assert res["primary_capability"] == "coding"
    assert res["confidence"] > 0.50

def test_router_math():
    router = CapabilityRouter()
    res = router.route("Derive the eigenvalues and eigenvectors of a symmetric covariance matrix.")
    assert res["primary_capability"] == "math"
    assert res["confidence"] > 0.40

def test_router_reasoning():
    router = CapabilityRouter()
    res = router.route("Analyze the validity of the syllogism and evaluate formal logical fallacies.")
    assert res["primary_capability"] == "reasoning"
    assert res["confidence"] > 0.40

def test_router_retrieval():
    router = CapabilityRouter()
    res = router.route("Retrieve RFC specifications for TLS 1.3 cryptographic handshake protocols.")
    assert res["primary_capability"] == "retrieval"

def test_router_replication():
    router = CapabilityRouter(replication_cutoff=0.99)
    # Cross-domain prompt triggering replication
    res = router.route("Analyze the mathematical proof and write a Python script.")
    assert res["replicate"] is True
    assert res["secondary_capability"] is not None

