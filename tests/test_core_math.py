import numpy as np
import pytest

from twmacl.trust import TrustMatrix
from twmacl.diagnostics import entropy_per_agent, asymmetry_index
from twmacl.portfolio import project_l1_leverage
from twmacl.policies import LinearGaussianPolicy, ActionResult

def test_trust_matrix_initialization():
    tm = TrustMatrix(num_agents=3, alpha=0.1, lambda_=2.0)
    assert tm.raw.shape == (3, 3)
    assert np.all(tm.raw == 0.0)

def test_trust_matrix_update_and_normalize():
    tm = TrustMatrix(num_agents=3, alpha=0.5, lambda_=1.0)
    
    # Predictions:
    # Agent 0: [1, 1]
    # Agent 1: [2, 2] (better)
    # Agent 2: [3, 3] (worse)
    predictions = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])
    realized_return = np.array([1.9, 1.9])
    
    tm.update(predictions, realized_return)
    
    # L1 errors:
    # Agent 0: |1-1.9| + |1-1.9| = 1.8
    # Agent 1: |2-1.9| + |2-1.9| = 0.2
    # Agent 2: |3-1.9| + |3-1.9| = 2.2
    
    # Raw scores (alpha=0.5, prev=0):
    # Agent 0: 0.5 * exp(-1 * 1.8) = 0.5 * 0.165 = 0.0826
    # Agent 1: 0.5 * exp(-1 * 0.2) = 0.5 * 0.818 = 0.4093
    # Agent 2: 0.5 * exp(-1 * 2.2) = 0.5 * 0.110 = 0.0554
    
    assert tm.raw[0, 1] > tm.raw[0, 0] # Agent 1 should be trusted more than Agent 0 by Agent 0
    assert tm.raw[0, 1] > tm.raw[0, 2] # Agent 1 should be trusted more than Agent 2 by Agent 0
    
    norm = tm.normalized()
    assert norm.shape == (3, 3)
    assert np.allclose(np.diag(norm), 0.0) # Self trust is 0
    assert np.allclose(np.sum(norm, axis=1), 1.0) # Rows sum to 1
    
    # Agent 0 trusts Agent 1 more than Agent 2
    assert norm[0, 1] > norm[0, 2]

def test_entropy_per_agent():
    # 3 agents
    normalized_trust = np.array([
        [0.0, 0.5, 0.5], # Agent 0 trusts 1 and 2 equally (max entropy)
        [0.0, 0.0, 1.0], # Agent 1 trusts only 2 (min entropy)
        [0.2, 0.8, 0.0]  # Agent 2 has skewed trust
    ])
    entropy = entropy_per_agent(normalized_trust)
    
    assert np.isclose(entropy[0], -np.log(0.5))
    assert np.isclose(entropy[1], 0.0)
    assert entropy[2] > 0.0 and entropy[2] < entropy[0]

def test_asymmetry_index():
    # Symmetric trust
    normalized_trust_sym = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    assert np.isclose(asymmetry_index(normalized_trust_sym), 0.0)
    
    # Asymmetric trust
    normalized_trust_asym = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    # Pairs: (0,1): |1-0|=1, (0,2): |0-1|=1, (1,2): |1-0|=1. Total diff = 3 (across 3 unordered pairs, but code counts 6 ordered pairs and divides by 6). Total=6, count=6 -> index=1.0
    assert np.isclose(asymmetry_index(normalized_trust_asym), 1.0)

def test_project_l1_leverage():
    # Already within cap
    w1 = np.array([0.5, -0.2, 0.1])
    assert np.allclose(project_l1_leverage(w1, 2.0), w1)
    
    # Exactly at cap
    w2 = np.array([1.0, -1.0])
    assert np.allclose(project_l1_leverage(w2, 2.0), w2)
    
    # Exceeds cap (L1=3.0) -> should scale by 2/3
    w3 = np.array([1.5, -1.0, 0.5])
    expected = w3 * (2.0 / 3.0)
    assert np.allclose(project_l1_leverage(w3, 2.0), expected)
    
    # All zeros
    w4 = np.array([0.0, 0.0])
    assert np.allclose(project_l1_leverage(w4, 2.0), w4)

def test_policy_reward_normalization():
    rng = np.random.default_rng(42)
    policy = LinearGaussianPolicy(
        num_assets=2,
        obs_dim=3,
        leverage_cap=2.0,
        learning_rate=0.01,
        exploration_std=0.1,
        reward_baseline_decay=0.95,
        rng=rng
    )
    
    # First reward
    n1 = policy._normalize_reward(100.0)
    assert n1 == 0.0 # < 2 samples
    
    # Second reward
    n2 = policy._normalize_reward(110.0)
    # Mean=(100+110)/2=105. Std=sqrt(((100-105)^2 + (110-105)^2)/1)=7.07
    # n2 = (110 - 105) / 7.07 = 0.707
    assert np.isclose(n2, 0.7071, atol=0.01)
    
    # Test reset
    policy.reset_normalizer()
    n3 = policy._normalize_reward(1000.0)
    assert n3 == 0.0 # Should be back to initial state
